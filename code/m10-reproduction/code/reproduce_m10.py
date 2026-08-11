#!/usr/bin/env python3
"""Canonical, single-entry reproduction of the submitted M10 retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PAIR_SHA256 = "dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9"
TRAJECTORY_SHA256 = "fb8ca29a7807e334fa0eab2d22fd3c3d52852c2f42f534969c4b1605578617a9"
CORPUS_SHA256 = "4d795938bb89cbd7e7467a8da4e772f7ae95e6b533181aeace2a5e3fd3de6393"
CORPUS_BYTES = 18_496_937_987
BASE_MODEL_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
TRAIN_SHA256 = "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def build_commands(
    *,
    python: str,
    code_root: Path,
    pairs: Path,
    trajectories: Path,
    corpus: Path,
    base_model: Path,
    output_root: Path,
    dry_run: bool,
) -> list[list[str]]:
    work = output_root / "work"
    provenance = work / "provenance"
    verified = work / "verified"
    split = work / "m10" / "early_stop_v1"
    commands = [
        [
            python,
            str(code_root / "build_provenance.py"),
            "--trajectory-archive",
            str(trajectories),
            "--pairs",
            str(pairs),
            "--tokenizer",
            str(base_model),
            "--output-root",
            str(provenance),
            "--expected-rows",
            "96504",
            "--expected-archive-sha256",
            TRAJECTORY_SHA256,
            "--expected-pairs-sha256",
            PAIR_SHA256,
            "--expected-tokenizer-sha256",
            TOKENIZER_SHA256,
        ],
        [
            python,
            str(code_root / "preprocess.py"),
            "--input",
            str(pairs),
            "--output",
            str(verified / "LRAT-training-pairs.jsonl"),
            "--manifest",
            str(verified / "preprocess_manifest.json"),
            "--trajectory-provenance",
            str(provenance / "provenance.jsonl"),
            "--corpus",
            str(corpus),
            "--expected-corpus-sha256",
            CORPUS_SHA256,
            "--expected-corpus-bytes",
            str(CORPUS_BYTES),
            "--require-full-traceability",
        ],
        [
            python,
            str(code_root / "prepare_split.py"),
            "--source",
            str(verified / "LRAT-training-pairs.jsonl"),
            "--output-root",
            str(split),
            "--dev-queries",
            "1500",
            "--test-queries",
            "500",
            "--salt",
            "ccir-early-stop-v1",
            "--write-train",
        ],
        [
            python,
            str(code_root / "train.py"),
            "--base-model",
            str(base_model),
            "--source-pairs",
            str(verified / "LRAT-training-pairs.jsonl"),
            "--source-manifest",
            str(verified / "preprocess_manifest.json"),
            "--train-data",
            str(split / "train.jsonl"),
            "--split-manifest",
            str(split / "manifest.json"),
            "--output-dir",
            str(output_root / "training"),
            "--cache-dir",
            str(output_root / "cache"),
        ],
    ]
    if dry_run:
        commands[-1].append("--dry-run")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the only supported M10 reproduction path from pinned official inputs."
    )
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1",
        help="Exactly two CUDA device ids; device identity may vary, process count may not.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all identity/provenance/split checks and print the fixed training command.",
    )
    args = parser.parse_args()

    devices = [value.strip() for value in args.cuda_visible_devices.split(",") if value.strip()]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("--cuda-visible-devices must name exactly two distinct devices")

    pairs = args.pairs.resolve()
    trajectories = args.trajectories.resolve()
    corpus = args.corpus.resolve()
    base_model = args.base_model.resolve()
    output_root = args.output_root.resolve()
    code_root = Path(__file__).resolve().parent

    if output_root.exists():
        raise FileExistsError(f"refusing to reuse or overwrite output root: {output_root}")

    assert_file(pairs, PAIR_SHA256)
    assert_file(trajectories, TRAJECTORY_SHA256)
    if corpus.stat().st_size != CORPUS_BYTES:
        raise ValueError(f"corpus byte-size mismatch: {corpus.stat().st_size} != {CORPUS_BYTES}")
    assert_file(corpus, CORPUS_SHA256)
    assert_file(base_model / "model.safetensors", BASE_MODEL_SHA256)
    assert_file(base_model / "tokenizer.json", TOKENIZER_SHA256)

    output_root.mkdir(parents=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "model_id": "M10",
        "started_at": datetime.now().astimezone().isoformat(),
        "dry_run": args.dry_run,
        "cuda_visible_devices": devices,
        "input_sha256": {
            "pairs": PAIR_SHA256,
            "trajectories": TRAJECTORY_SHA256,
            "corpus": CORPUS_SHA256,
            "base_model": BASE_MODEL_SHA256,
            "tokenizer": TOKENIZER_SHA256,
            "expected_train_jsonl": TRAIN_SHA256,
        },
    }
    record_path = output_root / "CANONICAL_REPRODUCTION.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    commands = build_commands(
        python=sys.executable,
        code_root=code_root,
        pairs=pairs,
        trajectories=trajectories,
        corpus=corpus,
        base_model=base_model,
        output_root=output_root,
        dry_run=args.dry_run,
    )
    for command in commands:
        run(command, environment=environment)

    final_weight = output_root / "training" / "model.safetensors"
    result = {
        **record,
        "completed_at": datetime.now().astimezone().isoformat(),
        "status": "dry_run_verified" if args.dry_run else "training_completed",
        "model_safetensors_sha256": (
            sha256_file(final_weight) if final_weight.is_file() else None
        ),
        "acceptance": (
            "Training is accepted by the documented dev1500 metric tolerance, "
            "not by byte-identical checkpoint reproduction."
        ),
    }
    record_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
