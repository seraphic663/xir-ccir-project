#!/usr/bin/env python3
"""Build deterministic query-disjoint train/dev/test data for early stopping.

The LRAT paper defines supervision inside each Search -> Browse event.  This
split therefore keeps original training rows intact, but assigns *complete*
normalized-query groups to exactly one of train, dev, or locked test.  Eval
rows aggregate all positives and negatives observed for a selected query;
an identifier that is positive anywhere in the group is never emitted as a
negative.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import statistics
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_key(query_key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{query_key}".encode()).hexdigest()


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def validate_row(row: Any, line_number: int) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: row must be a JSON object")
    query = row.get("query")
    if not isinstance(query, str):
        raise ValueError(f"line {line_number}: query must be a string")
    for text_key, id_key in (("pos", "pos_id"), ("neg", "neg_id")):
        texts, identifiers = row.get(text_key), row.get(id_key)
        if not isinstance(texts, list) or not isinstance(identifiers, list):
            raise ValueError(f"line {line_number}: {text_key}/{id_key} must be lists")
        if not texts or len(texts) != len(identifiers):
            raise ValueError(f"line {line_number}: {text_key}/{id_key} empty or misaligned")
    return query, norm_query(query)


def aggregate_eval_row(group: dict[str, Any]) -> dict[str, Any]:
    positives = group["positives"]
    negatives = {key: value for key, value in group["negatives"].items() if key not in positives}
    if not positives or not negatives:
        raise ValueError(f"query group lacks eval candidates: {group['normalized_query_sha256']}")
    return {
        "query": group["query"],
        "pos_id": list(positives),
        "pos": list(positives.values()),
        "neg_id": list(negatives),
        "neg": list(negatives.values()),
    }


def prepare(
    source: Path,
    output_root: Path,
    *,
    dev_queries: int = 1500,
    test_queries: int = 500,
    salt: str = "ccir-early-stop-v1",
    write_train: bool = False,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    if dev_queries < 1 or test_queries < 1 or not salt:
        raise ValueError("dev_queries, test_queries, and salt must be non-empty/positive")

    source_digest = hashlib.sha256()
    group_rows: collections.Counter[str] = collections.Counter()
    source_rows = 0
    empty_query_rows = 0
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            source_rows += 1
            source_digest.update(line)
            row = json.loads(line)
            _, query_key = validate_row(row, line_number)
            empty_query_rows += int(not query_key)
            group_rows[query_key] += 1

    query_keys = [key for key in group_rows if key]
    required = dev_queries + test_queries
    if required > len(query_keys):
        raise ValueError(f"requested {required} query groups, only {len(query_keys)} non-empty groups available")
    ordered = sorted(query_keys, key=lambda key: (split_key(key, salt), key))
    dev_keys = set(ordered[:dev_queries])
    test_keys = set(ordered[dev_queries:required])
    heldout_keys = dev_keys | test_keys

    selected: dict[str, dict[str, Any]] = {}
    for key in heldout_keys:
        selected[key] = {
            "query": None,
            "source_lines": [],
            "positives": {},
            "negatives": {},
            "normalized_query_sha256": hashlib.sha256(key.encode()).hexdigest(),
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent))
    try:
        train_path = staging / "train.jsonl"
        train_handle = train_path.open("xb") if write_train else None
        train_rows = 0
        heldout_source_rows = 0
        try:
            with source.open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    query, query_key = validate_row(row, line_number)
                    if query_key not in heldout_keys:
                        train_rows += 1
                        if train_handle is not None:
                            train_handle.write(line if line.endswith(b"\n") else line + b"\n")
                        continue
                    heldout_source_rows += 1
                    group = selected[query_key]
                    if group["query"] is None:
                        group["query"] = query
                    group["source_lines"].append(line_number)
                    for identifier, text in zip(row["pos_id"], row["pos"]):
                        group["positives"].setdefault(str(identifier), text)
                    for identifier, text in zip(row["neg_id"], row["neg"]):
                        group["negatives"].setdefault(str(identifier), text)
        finally:
            if train_handle is not None:
                train_handle.close()

        if set(selected) & ({key for key in group_rows if key not in heldout_keys}):
            raise AssertionError("normalized query split overlap")
        if train_rows + heldout_source_rows != source_rows:
            raise AssertionError((train_rows, heldout_source_rows, source_rows))

        split_summaries: dict[str, Any] = {}
        for label, keys in (("dev", dev_keys), ("test", test_keys)):
            output = staging / f"{label}.jsonl"
            source_count_values: list[float] = []
            candidate_values: list[float] = []
            positive_values: list[float] = []
            negative_values: list[float] = []
            provenance = []
            with output.open("x", encoding="utf-8") as destination:
                for key in sorted(keys, key=lambda value: (split_key(value, salt), value)):
                    group = selected[key]
                    eval_row = aggregate_eval_row(group)
                    destination.write(json.dumps(eval_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    source_count_values.append(len(group["source_lines"]))
                    positive_values.append(len(eval_row["pos_id"]))
                    negative_values.append(len(eval_row["neg_id"]))
                    candidate_values.append(len(eval_row["pos_id"]) + len(eval_row["neg_id"]))
                    provenance.append(
                        {
                            "normalized_query_sha256": group["normalized_query_sha256"],
                            "source_lines": group["source_lines"],
                            "positive_count": len(eval_row["pos_id"]),
                            "negative_count": len(eval_row["neg_id"]),
                        }
                    )
            split_summaries[label] = {
                "query_groups": len(keys),
                "source_rows_excluded": sum(map(int, source_count_values)),
                "source_rows_per_query": describe(source_count_values),
                "positives_per_query": describe(positive_values),
                "negatives_per_query": describe(negative_values),
                "candidates_per_query": describe(candidate_values),
                "output": {
                    "path": output.name,
                    "rows": len(keys),
                    "bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                },
                "provenance": provenance,
            }

        train_output = None
        if write_train:
            train_output = {
                "path": train_path.name,
                "rows": train_rows,
                "bytes": train_path.stat().st_size,
                "sha256": file_sha256(train_path),
            }
        report = {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": {
                "unit": "complete normalized-query group",
                "normalization": "strip, collapse whitespace, Unicode casefold",
                "ordering": "ascending SHA-256(salt + NUL + normalized_query), then normalized_query",
                "assignment": "first dev_queries groups -> dev; next test_queries -> locked test; remainder -> train",
                "salt": salt,
                "locked_test_policy": "do not use test metrics for checkpoint, hyperparameter, or method selection",
                "eval_aggregation": "union identifiers across source rows; any positive identifier is removed from negatives",
            },
            "source": {
                "path": str(source),
                "rows": source_rows,
                "bytes": source.stat().st_size,
                "sha256": source_digest.hexdigest(),
                "nonempty_normalized_query_groups": len(query_keys),
                "empty_query_rows_retained_in_train": empty_query_rows,
                "rows_per_nonempty_query": describe([float(group_rows[key]) for key in query_keys]),
            },
            "train": {
                "rows": train_rows,
                "query_groups_excluded": len(heldout_keys),
                "source_rows_excluded": heldout_source_rows,
                "normalized_query_overlap_with_dev_or_test": 0,
                "output": train_output,
            },
            "dev": split_summaries["dev"],
            "test": split_summaries["test"],
        }
        manifest = staging / "manifest.json"
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output_root)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dev-queries", type=int, default=1500)
    parser.add_argument("--test-queries", type=int, default=500)
    parser.add_argument("--salt", default="ccir-early-stop-v1")
    parser.add_argument("--write-train", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.source,
        args.output_root,
        dev_queries=args.dev_queries,
        test_queries=args.test_queries,
        salt=args.salt,
        write_train=args.write_train,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
