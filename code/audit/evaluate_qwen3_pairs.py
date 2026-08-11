#!/usr/bin/env python3
"""Evaluate a Qwen3 embedding model on JSONL query/positive/negative groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


def encode(
    texts: list[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        chunks.append(embeddings.cpu())
    return torch.cat(chunks, dim=0)


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            query = row.get("query")
            positives = row.get("pos")
            negatives = row.get("neg")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"line {line_number}: missing query")
            if not isinstance(positives, list) or not positives:
                raise ValueError(f"line {line_number}: missing positives")
            if not isinstance(negatives, list) or not negatives:
                raise ValueError(f"line {line_number}: missing negatives")
            rows.append({"query": query, "pos": positives, "neg": negatives})
    if not rows:
        raise ValueError("input contains no evaluable rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("--batch-size and --max-length must be positive")

    rows = load_rows(args.input, args.limit)
    queries = [f"Instruct: {args.instruction}\nQuery:{row['query']}" for row in rows]
    passages: list[str] = []
    spans: list[tuple[int, int, int]] = []
    for row in rows:
        start = len(passages)
        positives = [str(value) for value in row["pos"]]
        negatives = [str(value) for value in row["neg"]]
        passages.extend(positives)
        passages.extend(negatives)
        spans.append((start, len(passages), len(positives)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    loaded_at = time.perf_counter()

    query_embeddings = encode(
        queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    passage_embeddings = encode(
        passages,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    encoded_at = time.perf_counter()

    ranks: list[int] = []
    details: list[dict] = []
    for index, (start, end, positive_count) in enumerate(spans):
        scores = query_embeddings[index] @ passage_embeddings[start:end].T
        order = torch.argsort(scores, descending=True)
        positive_positions = torch.nonzero(order < positive_count, as_tuple=False).flatten()
        rank = int(positive_positions.min().item()) + 1
        ranks.append(rank)
        best_positive = float(scores[:positive_count].max().item())
        best_negative = float(scores[positive_count:].max().item())
        details.append(
            {
                "row_index": index,
                "query_sha256": hashlib.sha256(rows[index]["query"].encode()).hexdigest(),
                "candidate_count": end - start,
                "positive_count": positive_count,
                "best_positive_rank": rank,
                "best_positive_score": best_positive,
                "best_negative_score": best_negative,
                "positive_negative_margin": best_positive - best_negative,
            }
        )

    metrics = {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_positive_rank": sum(ranks) / len(ranks),
        "median_positive_rank": float(median(ranks)),
        "max_positive_rank": max(ranks),
    }
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result = {
        "model": str(args.model),
        "input": str(args.input),
        "rows": len(rows),
        "passages": len(passages),
        "embedding_dimension": int(query_embeddings.shape[1]),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "instruction": args.instruction,
        "device": str(device),
        "dtype": str(dtype),
        "model_load_seconds": loaded_at - started,
        "encoding_seconds": encoded_at - loaded_at,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_memory,
        "metrics": metrics,
        "rank_histogram": {
            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
        },
        "details": details,
    }
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("non-finite metric produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
