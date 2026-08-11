#!/usr/bin/env python3
"""Validate and stage the exact organizer-provided training pairs used by M10.

This program never calls a model, network service, or external API. Optional
trajectory provenance and corpus checks make every pair row traceable to the
organizer-provided raw sources; without both, the manifest remains explicitly
blocked on the competition's interpretation of the released pair file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterator


EXPECTED_SHA256 = "dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9"
EXPECTED_ROWS = 96_504
EXPECTED_BYTES = 3_883_089_616
EXPECTED_CORPUS_SHA256 = "4d795938bb89cbd7e7467a8da4e772f7ae95e6b533181aeace2a5e3fd3de6393"
EXPECTED_CORPUS_BYTES = 18_496_937_987
REQUIRED_FIELDS = {
    "query",
    "pos",
    "pos_id",
    "neg",
    "neg_id",
    "reasoning_len",
    "satisfied",
    "reweight_rate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_row(row: dict[str, Any], line_number: int) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
    if not isinstance(row["query"], str):
        raise ValueError(f"line {line_number}: query must be a string")
    for text_field, id_field in (("pos", "pos_id"), ("neg", "neg_id")):
        texts = row[text_field]
        identifiers = row[id_field]
        if not isinstance(texts, list) or not texts or not all(
            isinstance(value, str) and value for value in texts
        ):
            raise ValueError(f"line {line_number}: {text_field} must be non-empty strings")
        if not isinstance(identifiers, list) or len(identifiers) != len(texts) or not all(
            isinstance(value, (str, int)) and str(value) for value in identifiers
        ):
            raise ValueError(f"line {line_number}: {id_field} must align with {text_field}")
    if not isinstance(row["reasoning_len"], int) or isinstance(row["reasoning_len"], bool) or row["reasoning_len"] < 0:
        raise ValueError(f"line {line_number}: invalid reasoning_len")
    if not isinstance(row["satisfied"], bool):
        raise ValueError(f"line {line_number}: satisfied must be boolean")
    weight = row["reweight_rate"]
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"line {line_number}: reweight_rate must be finite and positive")


def iter_pairs(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            validate_row(row, line_number)
            yield line_number, row


def build_corpus_index(corpus: Path, database: Path) -> tuple[int, str]:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL)")
    rows = 0
    with corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"corpus line {line_number}: invalid JSON") from error
            if not isinstance(document, dict):
                raise ValueError(f"corpus line {line_number}: expected an object")
            doc_id = document.get("doc_id")
            title = document.get("title")
            content = document.get("content")
            if not isinstance(doc_id, (str, int)) or not isinstance(title, str) or not isinstance(content, str):
                raise ValueError(f"corpus line {line_number}: invalid doc_id/title/content")
            try:
                connection.execute(
                    "INSERT INTO documents(doc_id, title, content) VALUES (?, ?, ?)",
                    (str(doc_id), title, content),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"corpus line {line_number}: duplicate doc_id {doc_id}") from error
            rows += 1
            if rows % 10_000 == 0:
                connection.commit()
    connection.commit()
    connection.close()
    return rows, sha256_file(corpus)


def expected_document_text(title: str, content: str) -> str:
    return f"{title}\n{content}" if title else content


def validate_corpus_links(pairs: Path, database: Path) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    checked = 0
    try:
        for line_number, row in iter_pairs(pairs):
            for text_field, id_field in (("pos", "pos_id"), ("neg", "neg_id")):
                for text, doc_id in zip(row[text_field], row[id_field]):
                    result = connection.execute(
                        "SELECT title, content FROM documents WHERE doc_id = ?",
                        (str(doc_id),),
                    ).fetchone()
                    if result is None:
                        raise ValueError(f"line {line_number}: document {doc_id} is absent from corpus")
                    if text != expected_document_text(result[0], result[1]):
                        raise ValueError(f"line {line_number}: text differs from corpus document {doc_id}")
                    checked += 1
    finally:
        connection.close()
    return checked


def collect_expected_documents(pairs: Path) -> tuple[dict[str, tuple[bytes, int]], int]:
    """Collect the exact text identity required for every referenced corpus id."""
    expected: dict[str, tuple[bytes, int]] = {}
    references = 0
    for line_number, row in iter_pairs(pairs):
        for text_field, id_field in (("pos", "pos_id"), ("neg", "neg_id")):
            for text, doc_id in zip(row[text_field], row[id_field]):
                identifier = str(doc_id)
                identity = (hashlib.sha256(text.encode()).digest(), len(text.encode()))
                previous = expected.setdefault(identifier, identity)
                if previous != identity:
                    raise ValueError(
                        f"line {line_number}: document {identifier} has conflicting pair text"
                    )
                references += 1
    return expected, references


def corpus_document(record: dict[str, Any], line_number: int) -> tuple[str, str, str]:
    """Normalize the paper corpus and the competition-described corpus schemas."""
    if "id" in record and "contents" in record:
        doc_id = record["id"]
        text = record["contents"]
        schema = "id_contents"
    elif {"doc_id", "title", "content"} <= set(record):
        doc_id = record["doc_id"]
        title = record["title"]
        content = record["content"]
        if not isinstance(title, str) or not isinstance(content, str):
            raise ValueError(f"corpus line {line_number}: invalid title/content")
        text = expected_document_text(title, content)
        schema = "doc_id_title_content"
    else:
        raise ValueError(
            f"corpus line {line_number}: expected id/contents or doc_id/title/content"
        )
    if not isinstance(doc_id, (str, int)) or isinstance(doc_id, bool) or not str(doc_id):
        raise ValueError(f"corpus line {line_number}: invalid document id")
    if not isinstance(text, str) or not text:
        raise ValueError(f"corpus line {line_number}: invalid document text")
    return str(doc_id), text, schema


def validate_corpus_stream(
    pairs: Path,
    corpus: Path,
    *,
    expected_sha256: str | None,
    expected_bytes: int | None,
) -> dict[str, Any]:
    """Validate all pair references in one corpus pass without a full SQLite copy."""
    if expected_bytes is not None and corpus.stat().st_size != expected_bytes:
        raise ValueError(
            f"corpus byte-size mismatch: {corpus.stat().st_size} != {expected_bytes}"
        )
    expected, references = collect_expected_documents(pairs)
    remaining = set(expected)
    seen_referenced: set[str] = set()
    schemas: dict[str, int] = {}
    corpus_rows = 0
    digest = hashlib.sha256()
    with corpus.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"corpus line {line_number}: invalid JSON/UTF-8") from error
            if not isinstance(record, dict):
                raise ValueError(f"corpus line {line_number}: expected an object")
            doc_id, text, schema = corpus_document(record, line_number)
            corpus_rows += 1
            schemas[schema] = schemas.get(schema, 0) + 1
            if doc_id not in expected:
                continue
            if doc_id in seen_referenced:
                raise ValueError(f"corpus line {line_number}: duplicate referenced id {doc_id}")
            identity = (hashlib.sha256(text.encode()).digest(), len(text.encode()))
            if identity != expected[doc_id]:
                raise ValueError(
                    f"corpus line {line_number}: text differs from pair document {doc_id}"
                )
            seen_referenced.add(doc_id)
            remaining.discard(doc_id)
    corpus_sha256 = digest.hexdigest()
    if expected_sha256 and corpus_sha256 != expected_sha256:
        raise ValueError(f"corpus SHA-256 mismatch: {corpus_sha256}")
    if remaining:
        examples = sorted(remaining)[:20]
        raise ValueError(
            f"{len(remaining)} referenced documents are absent from corpus; examples={examples}"
        )
    return {
        "path": str(corpus.resolve()),
        "sha256": corpus_sha256,
        "bytes": corpus.stat().st_size,
        "rows": corpus_rows,
        "schemas": schemas,
        "linked_pair_document_references": references,
        "linked_unique_documents": len(expected),
        "missing_unique_documents": 0,
        "text_mismatches": 0,
    }


def validate_provenance(pairs: Path, provenance: Path) -> dict[str, int]:
    counts = {"stable": 0, "ambiguous": 0, "mismatch": 0}
    negative_ids_traceable = 0
    with provenance.open(encoding="utf-8") as provenance_handle:
        for row_index, (line_number, row) in enumerate(iter_pairs(pairs)):
            line = provenance_handle.readline()
            if not line:
                raise ValueError(f"provenance ended before pair line {line_number}")
            record = json.loads(line)
            if record.get("row_index") != row_index:
                raise ValueError(f"line {line_number}: provenance row_index mismatch")
            query_hash = hashlib.sha256(row["query"].encode()).hexdigest()
            if record.get("query_sha256") != query_hash:
                raise ValueError(f"line {line_number}: provenance query mismatch")
            if str(record.get("pos_id")) != str(row["pos_id"][0]):
                raise ValueError(f"line {line_number}: provenance positive id mismatch")
            if record.get("reasoning_len") != row["reasoning_len"]:
                raise ValueError(f"line {line_number}: provenance reasoning length mismatch")
            if record.get("negative_count") != len(row["neg_id"]):
                raise ValueError(f"line {line_number}: provenance negative count mismatch")
            expected_negative_sha = hashlib.sha256(
                "\0".join(str(value) for value in row["neg_id"]).encode()
            ).hexdigest()
            if record.get("negative_ids_sha256") != expected_negative_sha:
                raise ValueError(f"line {line_number}: provenance negative id mismatch")
            if record.get("negative_ids_traceable") is not True:
                raise ValueError(f"line {line_number}: negative ids are not trajectory-traceable")
            negative_ids_traceable += 1
            bucket = record.get("bucket")
            if bucket not in counts:
                raise ValueError(f"line {line_number}: unsupported provenance bucket {bucket!r}")
            counts[bucket] += 1
            if bucket == "stable" and not isinstance(record.get("event", {}).get("traj_path"), str):
                raise ValueError(f"line {line_number}: stable provenance lacks trajectory path")
            if bucket == "ambiguous" and not record.get("candidate_trajectory_paths"):
                raise ValueError(f"line {line_number}: ambiguous provenance lacks candidate paths")
        if provenance_handle.readline():
            raise ValueError("provenance contains more rows than pairs")
    if counts["mismatch"]:
        raise ValueError(f"{counts['mismatch']} pair rows cannot be traced to a trajectory")
    counts["negative_ids_traceable"] = negative_ids_traceable
    return counts


def validate_pairs(path: Path, expected_rows: int | None = EXPECTED_ROWS) -> dict[str, Any]:
    rows = 0
    empty_query_lines = []
    satisfied_true = 0
    min_negatives = None
    max_negatives = None
    lengths_and_weights = []
    for line_number, row in iter_pairs(path):
        rows += 1
        if not row["query"].strip():
            empty_query_lines.append(line_number)
        satisfied_true += int(row["satisfied"])
        negatives = len(row["neg"])
        lengths_and_weights.append((row["reasoning_len"], float(row["reweight_rate"])))
        min_negatives = negatives if min_negatives is None else min(min_negatives, negatives)
        max_negatives = negatives if max_negatives is None else max(max_negatives, negatives)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {rows}")
    positive_lengths = [
        float(length)
        for length, _ in lengths_and_weights
        if isinstance(length, (int, float)) and not isinstance(length, bool) and length > 0
    ]
    half_life = statistics.median(positive_lengths) if positive_lengths else 1.0
    raw_weights = [
        1 - math.exp(-length * math.log(2.0) / half_life)
        for length in positive_lengths
    ]
    mean_raw_weight = sum(raw_weights) / len(raw_weights) if raw_weights else 1.0
    maximum_weight_error = 0.0
    for length, observed in lengths_and_weights:
        if length > 0:
            raw = 1 - math.exp(-float(length) * math.log(2.0) / half_life)
            expected_weight = raw / mean_raw_weight if mean_raw_weight else 1.0
        else:
            expected_weight = 1.0
        maximum_weight_error = max(maximum_weight_error, abs(observed - expected_weight))
    if maximum_weight_error > 1e-12:
        raise ValueError(
            f"reweight_rate is not reproducible from reasoning_len: max error {maximum_weight_error}"
        )
    return {
        "rows": rows,
        "empty_query_lines": empty_query_lines,
        "satisfied_true": satisfied_true,
        "satisfied_false": rows - satisfied_true,
        "minimum_negatives": min_negatives,
        "maximum_negatives": max_negatives,
        "reweight_derivation": {
            "formula": "(1-exp(-reasoning_len*ln(2)/median_positive_reasoning_len))/mean_raw_weight",
            "median_positive_reasoning_len": half_life,
            "mean_raw_weight": mean_raw_weight,
            "maximum_absolute_error": maximum_weight_error,
            "verified": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trajectory-provenance", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--expected-corpus-sha256", default=EXPECTED_CORPUS_SHA256)
    parser.add_argument("--expected-corpus-bytes", type=int, default=EXPECTED_CORPUS_BYTES)
    parser.add_argument("--require-full-traceability", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"training-pair byte size mismatch: {source.stat().st_size}")
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_SHA256:
        raise ValueError(f"training-pair SHA-256 mismatch: {source_sha}")
    statistics = validate_pairs(source)
    rows = statistics["rows"]

    provenance_counts = None
    if args.trajectory_provenance:
        provenance_counts = validate_provenance(source, args.trajectory_provenance)

    corpus_report = None
    if args.corpus:
        corpus_report = validate_corpus_stream(
            source,
            args.corpus,
            expected_sha256=args.expected_corpus_sha256 or None,
            expected_bytes=args.expected_corpus_bytes,
        )

    full_traceability = provenance_counts is not None and corpus_report is not None
    if args.require_full_traceability and not full_traceability:
        raise ValueError("--require-full-traceability needs both --trajectory-provenance and --corpus")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output != source:
        if output.exists():
            raise FileExistsError(output)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            staging = Path(handle.name)
        try:
            shutil.copyfile(source, staging)
            os.replace(staging, output)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
    if sha256_file(output) != EXPECTED_SHA256:
        raise RuntimeError("staged training data differs from validated input")

    status = "full_raw_traceability_verified" if full_traceability else "blocked_pending_full_raw_traceability"
    report = {
        "schema_version": 1,
        "mode": "exact_m10_source_verification",
        "status": status,
        "input": {
            "path": str(source),
            "rows": rows,
            "bytes": source.stat().st_size,
            "sha256": source_sha,
            "schema_statistics": statistics,
        },
        "output": {
            "path": str(output),
            "rows": rows,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "trajectory_provenance": {
            "path": str(args.trajectory_provenance.resolve()),
            "sha256": sha256_file(args.trajectory_provenance),
            "buckets": provenance_counts,
        } if args.trajectory_provenance else None,
        "corpus": corpus_report,
        "external_data_used": False,
        "external_api_used": False,
    }
    manifest = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
