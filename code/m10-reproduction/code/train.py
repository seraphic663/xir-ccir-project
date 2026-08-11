#!/usr/bin/env python3
"""Reproduce M10 from an audited query-disjoint subset of official LRAT pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9"
EXPECTED_TRAIN_SHA256 = "158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9"
EXPECTED_CORPUS_SHA256 = "4d795938bb89cbd7e7467a8da4e772f7ae95e6b533181aeace2a5e3fd3de6393"
EXPECTED_TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
EXPECTED_SOURCE_ROWS = 96_504
EXPECTED_ROWS = 94_113


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("model_id") != "M10":
        raise ValueError("unsupported config")
    return value


def flag(arguments: list[str], name: str, enabled: bool) -> None:
    if enabled:
        arguments.append(name)


def build_command(
    config: dict[str, Any],
    *,
    base_model: Path,
    train_data: Path,
    output_dir: Path,
    cache_dir: Path,
    nproc_per_node: int,
    resume_from_checkpoint: Path | None,
) -> list[str]:
    training = config["training"]
    inference = config["inference"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={training['master_port']}",
        "-m",
        "FlagEmbedding.finetune.embedder.decoder_only.base",
        "--model_name_or_path",
        str(base_model),
        "--train_data",
        str(train_data),
        "--cache_path",
        str(cache_dir),
        "--train_group_size",
        str(training["train_group_size"]),
        "--query_max_len",
        str(training["query_max_len"]),
        "--passage_max_len",
        str(training["passage_max_len"]),
        "--pad_to_multiple_of",
        str(training["pad_to_multiple_of"]),
        "--query_instruction_for_retrieval",
        inference["query_instruction"],
        "--query_instruction_format",
        inference["query_instruction_format"],
        "--knowledge_distillation",
        str(training["knowledge_distillation"]),
        "--output_dir",
        str(output_dir),
        "--learning_rate",
        str(training["learning_rate"]),
        "--num_train_epochs",
        str(training["num_train_epochs"]),
        "--max_steps",
        str(training["max_steps"]),
        "--per_device_train_batch_size",
        str(training["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--dataloader_drop_last",
        "True",
        "--warmup_ratio",
        str(training["warmup_ratio"]),
        "--gradient_checkpointing_kwargs",
        '{"use_reentrant": false}',
        "--logging_steps",
        str(training["logging_steps"]),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(training["save_steps"]),
        "--save_total_limit",
        str(training["save_total_limit"]),
        "--temperature",
        str(training["temperature"]),
        "--sentence_pooling_method",
        training["sentence_pooling_method"],
        "--normalize_embeddings",
        str(training["normalize_embeddings"]),
        "--seed",
        str(training["seed"]),
        "--data_seed",
        str(training["data_seed"]),
        "--ddp_find_unused_parameters",
        "False",
        "--report_to",
        "none",
    ]
    flag(command, "--bf16", training["bf16"])
    flag(command, "--gradient_checkpointing", training["gradient_checkpointing"])
    flag(command, "--negatives_cross_device", training["negatives_cross_device"])
    if resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", str(resume_from_checkpoint)])
    return command


def validate_source_manifest(path: Path, source_pairs: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "full_raw_traceability_verified":
        raise ValueError("source manifest has not passed full raw traceability")
    source = value.get("input") or {}
    output = value.get("output") or {}
    if source.get("rows") != EXPECTED_SOURCE_ROWS or output.get("rows") != EXPECTED_SOURCE_ROWS:
        raise ValueError("source manifest row count mismatch")
    if source.get("sha256") != EXPECTED_SOURCE_SHA256 or output.get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("source manifest official-pair identity mismatch")
    if Path(output.get("path", "")).resolve() != source_pairs:
        raise ValueError("source-pairs path differs from validated preprocess output")
    derivation = (source.get("schema_statistics") or {}).get("reweight_derivation") or {}
    if derivation.get("verified") is not True or derivation.get("maximum_absolute_error", 1.0) > 1e-12:
        raise ValueError("source manifest reweight derivation is not verified")
    provenance = value.get("trajectory_provenance") or {}
    buckets = provenance.get("buckets") or {}
    if buckets.get("mismatch", 0) != 0:
        raise ValueError("source manifest contains trajectory mismatches")
    if buckets.get("negative_ids_traceable") != EXPECTED_SOURCE_ROWS:
        raise ValueError("source manifest lacks full negative-document traceability")
    if buckets.get("stable", 0) + buckets.get("ambiguous", 0) != EXPECTED_SOURCE_ROWS:
        raise ValueError("source manifest trajectory coverage mismatch")
    corpus = value.get("corpus") or {}
    if corpus.get("sha256") != EXPECTED_CORPUS_SHA256:
        raise ValueError("source manifest offline-corpus identity mismatch")
    if corpus.get("missing_unique_documents") != 0 or corpus.get("text_mismatches") != 0:
        raise ValueError("source manifest offline-corpus coverage is incomplete")
    return value


def validate_split_manifest(
    path: Path,
    train_data: Path,
    source_pairs: Path,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    source = value.get("source") or {}
    train = value.get("train") or {}
    output = train.get("output") or {}
    contract = value.get("contract") or {}
    if source.get("rows") != 96_504 or source.get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("split manifest official-source identity mismatch")
    if train.get("rows") != EXPECTED_ROWS or train.get("normalized_query_overlap_with_dev_or_test") != 0:
        raise ValueError("split manifest train identity or isolation mismatch")
    if output.get("sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError("split manifest training-data SHA-256 mismatch")
    if Path(path.parent / output.get("path", "")).resolve() != train_data:
        raise ValueError("train-data path differs from split manifest output")
    if contract.get("salt") != "ccir-early-stop-v1":
        raise ValueError("split manifest salt mismatch")
    if (value.get("dev") or {}).get("query_groups") != 1_500:
        raise ValueError("split manifest dev group count mismatch")
    if (value.get("test") or {}).get("query_groups") != 500:
        raise ValueError("split manifest locked-test group count mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--source-pairs", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    base_model = args.base_model.resolve()
    source_pairs = args.source_pairs.resolve()
    source_manifest = args.source_manifest.resolve()
    train_data = args.train_data.resolve()
    split_manifest = args.split_manifest.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    base_weight = base_model / "model.safetensors"
    if sha256_file(base_weight) != config["base_model"]["model_safetensors_sha256"]:
        raise ValueError("base-model model.safetensors SHA-256 mismatch")
    if sha256_file(base_model / "tokenizer.json") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("base-model tokenizer.json SHA-256 mismatch")
    if sha256_file(source_pairs) != EXPECTED_SOURCE_SHA256:
        raise ValueError("official source-pairs SHA-256 mismatch")
    if sha256_file(train_data) != config["training_data"]["sha256"]:
        raise ValueError("training-data SHA-256 mismatch")
    source_evidence = validate_source_manifest(source_manifest, source_pairs)
    split_evidence = validate_split_manifest(split_manifest, train_data, source_pairs)
    nproc = int(config["training"]["nproc_per_node"])
    if nproc != 2:
        raise ValueError("canonical M10 reproduction requires exactly two training processes")

    command = build_command(
        config,
        base_model=base_model,
        train_data=train_data,
        output_dir=output_dir,
        cache_dir=cache_dir,
        nproc_per_node=nproc,
        resume_from_checkpoint=args.resume_from_checkpoint.resolve() if args.resume_from_checkpoint else None,
    )
    launch_record = {
        "config": str(config_path.resolve()),
        "base_model": str(base_model),
        "base_model_sha256": sha256_file(base_weight),
        "source_pairs": str(source_pairs),
        "source_pairs_sha256": sha256_file(source_pairs),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "train_data": str(train_data),
        "train_data_sha256": sha256_file(train_data),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "compliance_gate": "full_raw_traceability_and_query_disjoint_split_verified",
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "command": command,
        "external_data_used": False,
        "external_api_used": False,
    }
    if args.dry_run:
        print(json.dumps(launch_record, ensure_ascii=False, indent=2))
        return
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output_dir}")
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RUNNING").write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(launch_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    vendor = Path(__file__).resolve().parent / "vendor" / "FlagEmbedding"
    if not vendor.is_dir():
        raise FileNotFoundError(f"vendored FlagEmbedding source not found: {vendor}")
    environment["PYTHONPATH"] = str(vendor) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("OMP_NUM_THREADS", str(config["training"]["omp_num_threads"]))
    try:
        subprocess.run(command, check=True, env=environment)
        final_weight = output_dir / "model.safetensors"
        if not final_weight.is_file():
            raise RuntimeError("trainer returned without final model.safetensors")
        (output_dir / "COMPLETED").write_text(
            f"model_safetensors_sha256={sha256_file(final_weight)}\n",
            encoding="utf-8",
        )
    except BaseException as error:
        (output_dir / "FAILED").write_text(f"error={type(error).__name__}: {error}\n", encoding="utf-8")
        raise
    finally:
        (output_dir / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
