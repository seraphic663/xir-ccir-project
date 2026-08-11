#!/usr/bin/env python3
"""Map released LRAT training pairs back to their source trajectory events.

The released default training JSONL intentionally contains no trajectory id.
This tool reconstructs provenance with the exact intermediate query, positive
document id, and Qwen tokenizer reasoning length.  It never extracts the
trajectory archive and never reads a locked evaluation split.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tarfile
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


BROWSE_TOOLS = {"get_document", "visit"}
NEGATIVE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explicit_not_relevant", re.compile(r"\b(?:not|isn['’]t|wasn['’]t)\s+(?:directly\s+)?relevant\b", re.I)),
    ("explicit_irrelevant", re.compile(r"\birrelevant\b", re.I)),
    ("explicit_not_helpful", re.compile(r"\b(?:not|doesn['’]t|didn['’]t)\s+(?:seem\s+)?helpful\b", re.I)),
    ("explicit_not_useful", re.compile(r"\b(?:not|isn['’]t|wasn['’]t)\s+(?:very\s+)?useful\b", re.I)),
    ("explicit_no_information", re.compile(r"\b(?:no|does not|doesn['’]t)\s+(?:relevant\s+)?information\b", re.I)),
    ("explicit_does_not_mention", re.compile(r"\bdoes\s+not\s+mention\b|\bdoesn['’]t\s+mention\b", re.I)),
    ("explicit_not_what_needed", re.compile(r"\bnot\s+(?:quite\s+)?what\s+(?:we|i)\s+need", re.I)),
    ("explicit_no_answer", re.compile(r"\bdoes\s+not\s+answer\b|\bdoesn['’]t\s+answer\b", re.I)),
)
WORD_RE = re.compile(r"\w+", re.UNICODE)
WS_RE = re.compile(r"\s+")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_key(query: str, positive_id: str) -> str:
    return hashlib.sha256(f"{query}\0{positive_id}".encode()).hexdigest()


def normalized_query_hash(query: str) -> str:
    normalized = WS_RE.sub(" ", query.strip()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def text_tokens(value: Any) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return WORD_RE.findall(normalized)


def answer_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    if value is None:
        return []
    return [str(value)]


def answer_features(answer: Any, final_output: str) -> dict[str, Any]:
    candidates = [text_tokens(item) for item in answer_values(answer)]
    candidates = [item for item in candidates if item]
    final_tokens = text_tokens(final_output)
    final_joined = " ".join(final_tokens)
    final_set = set(final_tokens)
    exact = any(" ".join(item) in final_joined for item in candidates)
    subset = any(set(item) <= final_set for item in candidates)
    coverage = 0.0
    for item in candidates:
        coverage = max(coverage, len(set(item) & final_set) / len(set(item)))
    return {
        "gold_answer_present": bool(candidates),
        "final_output_present": bool(final_tokens),
        "answer_exact_substring": exact,
        "answer_token_subset": subset,
        "answer_token_coverage": coverage,
    }


def parse_arguments(step: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(step.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_search(step: dict[str, Any]) -> tuple[str, list[str]]:
    arguments = parse_arguments(step)
    query_value = arguments.get("query", [""])
    query = query_value[0] if isinstance(query_value, list) and query_value else str(query_value or "")
    documents = []
    for line in str(step.get("output") or "").splitlines():
        if line.startswith("DocID:"):
            documents.append(line.split(":", 1)[1].strip())
    return query, documents


def browse_document_id(step: dict[str, Any]) -> str:
    value = parse_arguments(step).get("docid")
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).split(":")[-1] if value is not None else ""


def next_reasoning(steps: list[dict[str, Any]], index: int) -> str:
    if index + 1 >= len(steps) or steps[index + 1].get("type") != "reasoning":
        return ""
    value = steps[index + 1].get("output", "")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def negative_cues(text: str) -> list[str]:
    return [name for name, pattern in NEGATIVE_CUES if pattern.search(text)]


def iter_archive_events(archive: Path) -> Iterable[dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            value = json.load(extracted)
            if not isinstance(value, dict):
                continue
            raw_steps = value.get("result") or []
            if not isinstance(raw_steps, list):
                raw_steps = []
            steps = [item if isinstance(item, dict) else {} for item in raw_steps]
            trajectory_source_documents = [
                str(item) for item in (value.get("retrieved_docids") or [])
            ]
            for source_step in steps:
                if source_step.get("type") == "tool_call" and source_step.get("tool_name") == "search":
                    _, source_documents = parse_search(source_step)
                    trajectory_source_documents.extend(source_documents)
                elif source_step.get("type") == "tool_call" and source_step.get("tool_name") in BROWSE_TOOLS:
                    source_document = browse_document_id(source_step)
                    if source_document:
                        trajectory_source_documents.append(source_document)
            trajectory_source_documents = list(dict.fromkeys(trajectory_source_documents))
            total_searches = sum(
                step.get("type") == "tool_call" and step.get("tool_name") == "search"
                for step in steps
            )
            total_browses = sum(
                step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS
                for step in steps
            )
            final_output = ""
            for step in steps:
                if step.get("type") == "output_text":
                    final_output = str(step.get("output") or "")
            answer_info = answer_features(value.get("answer"), final_output)
            parts = PurePosixPath(member.name).parts
            retriever = parts[1] if len(parts) > 2 else ""
            current_query = ""
            current_documents: list[str] = []
            searched_documents: list[str] = []
            search_index = -1
            browse_index = -1
            event_index = 0
            for index, step in enumerate(steps):
                if step.get("type") == "tool_call" and step.get("tool_name") == "search":
                    current_query, current_documents = parse_search(step)
                    searched_documents.extend(current_documents)
                    search_index += 1
                    browse_index = -1
                    continue
                if not (
                    step.get("type") == "tool_call"
                    and step.get("tool_name") in BROWSE_TOOLS
                ):
                    continue
                browse_index += 1
                document_id = browse_document_id(step)
                reasoning = next_reasoning(steps, index)
                try:
                    retrieved_rank = current_documents.index(document_id) + 1
                except ValueError:
                    retrieved_rank = None
                after_reasoning = steps[index + 2] if index + 2 < len(steps) else {}
                searched_so_far = list(dict.fromkeys(searched_documents))
                yield {
                    "key_sha256": exact_key(current_query, document_id),
                    "query": current_query,
                    "pos_id": document_id,
                    "reasoning_text": reasoning,
                    "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
                    "reasoning_chars": len(reasoning),
                    "negative_cues": negative_cues(reasoning),
                    "traj_path": member.name,
                    "retriever": retriever,
                    "seed_query_id": str(value.get("query_id", "")),
                    "trajectory_status": value.get("status"),
                    "trajectory_steps": len(steps),
                    "trajectory_searches": total_searches,
                    "trajectory_browses": total_browses,
                    "search_idx": search_index,
                    "browse_idx_in_search": browse_index,
                    "event_idx": event_index,
                    "retrieved_rank": retrieved_rank,
                    "search_result_count": len(current_documents),
                    "searched_doc_ids_so_far": searched_so_far,
                    "trajectory_source_doc_ids": trajectory_source_documents,
                    "next_step_type": after_reasoning.get("type"),
                    "next_tool_name": after_reasoning.get("tool_name"),
                    "gold_answer": value.get("answer"),
                    "final_output_excerpt": final_output[:1200],
                    **answer_info,
                }
                event_index += 1


def tokenize_event_lengths(
    events: list[dict[str, Any]],
    tokenizer_path: Path,
    *,
    batch_size: int = 512,
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        encodings = tokenizer.backend_tokenizer.encode_batch(
            [event["reasoning_text"] for event in batch],
            add_special_tokens=False,
        )
        for event, encoding in zip(batch, encodings):
            event["reasoning_len"] = len(encoding.ids)


def describe(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    valid = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not valid:
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "p99": None, "max": None, "mean": None}

    def at(fraction: float) -> float:
        return valid[min(len(valid) - 1, int((len(valid) - 1) * fraction))]

    return {
        "count": len(valid),
        "min": valid[0],
        "p10": at(0.10),
        "median": statistics.median(valid),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": valid[-1],
        "mean": statistics.fmean(valid),
    }


def load_pair_minimal(path: Path, expected_rows: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row.get("query")
            positive_ids = row.get("pos_id")
            if not isinstance(query, str):
                raise ValueError(f"line {line_number}: query must be a string")
            if not isinstance(positive_ids, list) or len(positive_ids) != 1:
                raise ValueError(f"line {line_number}: expected exactly one pos_id")
            reasoning_len = row.get("reasoning_len")
            if not isinstance(reasoning_len, int) or reasoning_len < 0:
                raise ValueError(f"line {line_number}: invalid reasoning_len")
            reweight_rate = row.get("reweight_rate")
            if not isinstance(reweight_rate, (int, float)) or not math.isfinite(reweight_rate) or reweight_rate <= 0:
                raise ValueError(f"line {line_number}: invalid reweight_rate")
            positive_id = str(positive_ids[0])
            negative_ids = row.get("neg_id")
            if not isinstance(negative_ids, list) or not all(
                isinstance(value, (str, int)) and str(value) for value in negative_ids
            ):
                raise ValueError(f"line {line_number}: invalid neg_id")
            negative_ids = [str(value) for value in negative_ids]
            rows.append(
                {
                    "row_index": line_number - 1,
                    "source_line": line_number,
                    "query": query,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "normalized_query_sha256": normalized_query_hash(query),
                    "pos_id": positive_id,
                    "key_sha256": exact_key(query, positive_id),
                    "reasoning_len": reasoning_len,
                    "reweight_rate": float(reweight_rate),
                    "satisfied": bool(row.get("satisfied")),
                    "negative_count": len(negative_ids),
                    "_negative_ids": negative_ids,
                }
            )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} pair rows, found {len(rows)}")
    return rows


def shorten_event(event: dict[str, Any]) -> dict[str, Any]:
    searched_doc_ids = event.get("searched_doc_ids_so_far", [])
    trajectory_source_doc_ids = event.get("trajectory_source_doc_ids", [])
    shortened = {
        key: value
        for key, value in event.items()
        if key not in {
            "query",
            "reasoning_text",
            "searched_doc_ids_so_far",
            "trajectory_source_doc_ids",
        }
    }
    shortened["searched_doc_ids_so_far_count"] = len(searched_doc_ids)
    shortened["searched_doc_ids_so_far_sha256"] = hashlib.sha256(
        "\0".join(searched_doc_ids).encode()
    ).hexdigest()
    shortened["trajectory_source_doc_ids_count"] = len(trajectory_source_doc_ids)
    shortened["trajectory_source_doc_ids_sha256"] = hashlib.sha256(
        "\0".join(trajectory_source_doc_ids).encode()
    ).hexdigest()
    shortened["reasoning_excerpt"] = event["reasoning_text"][:1200]
    return shortened


def map_pairs(
    pairs: list[dict[str, Any]], events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events_by_key: dict[str, list[int]] = collections.defaultdict(list)
    event_signatures: collections.Counter[tuple[str, int]] = collections.Counter()
    for index, event in enumerate(events):
        events_by_key[event["key_sha256"]].append(index)
        event_signatures[(event["key_sha256"], event["reasoning_len"])] += 1
    pair_signatures = collections.Counter(
        (pair["key_sha256"], pair["reasoning_len"]) for pair in pairs
    )

    mapped = []
    buckets: collections.Counter[str] = collections.Counter()
    negative_traceability: collections.Counter[str] = collections.Counter()
    for pair in pairs:
        candidates = events_by_key.get(pair["key_sha256"], [])
        exact = [
            index
            for index in candidates
            if events[index]["reasoning_len"] == pair["reasoning_len"]
        ]
        negative_ids = pair["_negative_ids"]
        negative_set = set(negative_ids)
        traceable = [
            index
            for index in exact
            if negative_set <= set(events[index].get("trajectory_source_doc_ids", []))
        ]
        signature = (pair["key_sha256"], pair["reasoning_len"])
        if not candidates:
            bucket = "mismatch"
            mismatch_reason = "no_query_pos_candidate"
        elif not exact:
            bucket = "mismatch"
            mismatch_reason = "reasoning_length_mismatch"
        elif not traceable:
            bucket = "mismatch"
            mismatch_reason = "negative_ids_not_retrieved_before_event"
        elif (
            len(traceable) == 1
            and event_signatures[signature] == 1
            and pair_signatures[signature] == 1
        ):
            bucket = "stable"
            mismatch_reason = None
        else:
            bucket = "ambiguous"
            mismatch_reason = "reused_exact_signature"
        candidate_pool = traceable or exact or candidates
        unmatched_negative_ids = []
        if exact and not traceable:
            searched_union = set()
            for index in exact:
                searched_union.update(events[index].get("trajectory_source_doc_ids", []))
            unmatched_negative_ids = sorted(negative_set - searched_union)
        public_pair = {key: value for key, value in pair.items() if not key.startswith("_")}
        record = {
            **public_pair,
            "bucket": bucket,
            "mismatch_reason": mismatch_reason,
            "query_pos_candidate_count": len(candidates),
            "exact_candidate_count": len(exact),
            "negative_traceable_candidate_count": len(traceable),
            "negative_ids_sha256": hashlib.sha256(
                "\0".join(negative_ids).encode()
            ).hexdigest(),
            "negative_ids_traceable": bool(traceable),
            "unmatched_negative_ids": unmatched_negative_ids[:20],
            "candidate_trajectory_paths": [
                events[index]["traj_path"] for index in candidate_pool[:20]
            ],
        }
        if bucket == "stable":
            record["event"] = shorten_event(events[traceable[0]])
        mapped.append(record)
        buckets[bucket] += 1
        negative_traceability["traceable" if traceable else "untraceable"] += 1

    report = {
        "pair_rows": len(pairs),
        "trajectory_events": len(events),
        "trajectory_event_keys": len(events_by_key),
        "buckets": dict(buckets),
        "stable_ratio": buckets["stable"] / len(pairs) if pairs else 0.0,
        "negative_ids_traceability": dict(negative_traceability),
    }
    return mapped, report


def build_feature_report(
    provenance: list[dict[str, Any]], mapping_summary: dict[str, Any]
) -> dict[str, Any]:
    stable = [record for record in provenance if record["bucket"] == "stable"]
    events = [record["event"] for record in stable]
    flags = collections.Counter()
    for event in events:
        flags["answer_unmatched"] += int(not event["answer_token_subset"])
        flags["continue_search"] += int(event["next_tool_name"] == "search")
        flags["low_rank_8_10"] += int(
            isinstance(event["retrieved_rank"], int) and event["retrieved_rank"] >= 8
        )
        flags["rank_missing"] += int(event["retrieved_rank"] is None)
        flags["empty_reasoning"] += int(event["reasoning_len"] == 0)
        flags["explicit_negative_reasoning"] += int(bool(event["negative_cues"]))
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "mapping": mapping_summary,
        "pair_fields": {
            "satisfied": dict(collections.Counter(record["satisfied"] for record in provenance)),
            "reasoning_len": describe(record["reasoning_len"] for record in provenance),
            "reweight_rate": describe(record["reweight_rate"] for record in provenance),
            "negative_count": describe(record["negative_count"] for record in provenance),
        },
        "stable_features": {
            "rows": len(stable),
            "retriever": dict(collections.Counter(event["retriever"] for event in events)),
            "trajectory_steps": describe(event["trajectory_steps"] for event in events),
            "trajectory_searches": describe(event["trajectory_searches"] for event in events),
            "trajectory_browses": describe(event["trajectory_browses"] for event in events),
            "search_idx": describe(event["search_idx"] for event in events),
            "browse_idx_in_search": describe(event["browse_idx_in_search"] for event in events),
            "retrieved_rank": describe(event["retrieved_rank"] for event in events),
            "next_action": {
                f"{action[0]}:{action[1]}": count
                for action, count in collections.Counter(
                    (event["next_step_type"], event["next_tool_name"]) for event in events
                ).items()
            },
            "candidate_flags": dict(flags),
        },
        "interpretation_limits": [
            "stable means deterministic provenance, not a relevance judgment",
            "mismatch and ambiguous rows must not be force-mapped",
            "continuing search does not by itself imply irrelevance",
            "answer token mismatch is a noisy trajectory-level proxy",
            "locked test data was not read",
        ],
    }


def build(
    archive: Path,
    pairs_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    *,
    expected_rows: int | None,
    expected_archive_sha256: str | None,
    expected_pairs_sha256: str | None,
    expected_tokenizer_sha256: str | None,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    for path in (archive, pairs_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(path)
    archive_sha256 = file_sha256(archive)
    pairs_sha256 = file_sha256(pairs_path)
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    tokenizer_sha256 = file_sha256(tokenizer_json)
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        raise ValueError("trajectory archive SHA-256 mismatch")
    if expected_pairs_sha256 and pairs_sha256 != expected_pairs_sha256:
        raise ValueError("pair data SHA-256 mismatch")
    if expected_tokenizer_sha256 and tokenizer_sha256 != expected_tokenizer_sha256:
        raise ValueError("tokenizer.json SHA-256 mismatch")

    events = list(iter_archive_events(archive))
    tokenize_event_lengths(events, tokenizer_path)
    pairs = load_pair_minimal(pairs_path, expected_rows)
    provenance, mapping_summary = map_pairs(pairs, events)
    feature_report = build_feature_report(provenance, mapping_summary)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    try:
        provenance_path = staging / "provenance.jsonl"
        bucket_handles = {
            bucket: (staging / f"{bucket}.row_indices.txt").open("x", encoding="utf-8")
            for bucket in ("stable", "mismatch", "ambiguous")
        }
        try:
            with provenance_path.open("x", encoding="utf-8") as handle:
                for record in provenance:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    bucket_handles[record["bucket"]].write(
                        f"{record['row_index']}\n"
                    )
        finally:
            for handle in bucket_handles.values():
                handle.close()
        feature_path = staging / "feature_report.json"
        feature_path.write_text(
            json.dumps(feature_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs = {}
        for path in sorted(staging.iterdir()):
            if path.name == "manifest.json":
                continue
            outputs[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": {
                "mapping_key": "exact intermediate query + NUL + positive document id",
                "stable_signature": "mapping key + tokenizer reasoning length, unique on both sides",
                "buckets": ["stable", "mismatch", "ambiguous"],
                "locked_test_used": False,
            },
            "inputs": {
                "trajectory_archive": {
                    "path": str(archive.resolve()),
                    "bytes": archive.stat().st_size,
                    "sha256": archive_sha256,
                },
                "pairs": {
                    "path": str(pairs_path.resolve()),
                    "bytes": pairs_path.stat().st_size,
                    "sha256": pairs_sha256,
                    "rows": len(pairs),
                },
                "tokenizer": {
                    "path": str(tokenizer_path.resolve()),
                    "tokenizer_json_sha256": tokenizer_sha256,
                    "files": {
                        path.name: {
                            "bytes": path.stat().st_size,
                            "sha256": file_sha256(path),
                        }
                        for path in sorted(tokenizer_path.iterdir())
                        if path.is_file()
                        and path.name
                        in {
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json",
                            "merges.txt",
                            "added_tokens.json",
                            "special_tokens_map.json",
                        }
                    },
                },
            },
            "mapping": mapping_summary,
            "outputs": outputs,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-pairs-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    args = parser.parse_args()
    result = build(
        args.trajectory_archive,
        args.pairs,
        args.tokenizer,
        args.output_root,
        expected_rows=args.expected_rows,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_pairs_sha256=args.expected_pairs_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
