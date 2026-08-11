#!/usr/bin/env python3
"""Encode query or document JSONL with the published competition contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported config")
    return value


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        sequence_lengths,
    ]


def iter_batches(path: Path, text_field: str, id_field: str, batch_size: int) -> Iterator[tuple[list[str], list[str]]]:
    identifiers: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(f"line {line_number}: missing string field {text_field!r}")
            identifiers.append(str(row.get(id_field, line_number - 1)))
            texts.append(row[text_field])
            if len(texts) == batch_size:
                yield identifiers, texts
                identifiers, texts = [], []
    if texts:
        yield identifiers, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("query", "document"))
    parser.add_argument(
        "--contract",
        choices=("competition", "training"),
        default="competition",
        help="Use the published evaluator contract by default; training reproduces the fine-tuning lengths/template.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--allow-nonlocal-files", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    all_config = load_config(args.config)
    config = (
        all_config["competition_inference"]
        if args.contract == "competition"
        else all_config["inference"]
    )
    checkpoint = args.checkpoint.resolve()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        add_eos_token=True,
        local_files_only=not args.allow_nonlocal_files,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        local_files_only=not args.allow_nonlocal_files,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()
    max_length = config["query_max_len"] if args.mode == "query" else config["passage_max_len"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    with args.output.open("x", encoding="utf-8") as output, torch.inference_mode():
        for identifiers, texts in iter_batches(args.input, args.text_field, args.id_field, args.batch_size):
            if args.mode == "query":
                if args.contract == "competition":
                    texts = [config["query_prefix"] + text for text in texts]
                else:
                    texts = [
                        config["query_instruction_format"].format(
                            config["query_instruction"], text
                        )
                        for text in texts
                    ]
            tokens = tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(args.device)
            hidden = model(**tokens, return_dict=True).last_hidden_state
            embeddings = last_token_pool(hidden, tokens["attention_mask"])
            if config["normalize"]:
                embeddings = functional.normalize(embeddings, p=2, dim=1)
            for identifier, embedding in zip(identifiers, embeddings.float().cpu().tolist()):
                output.write(json.dumps({"id": identifier, "embedding": embedding}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
