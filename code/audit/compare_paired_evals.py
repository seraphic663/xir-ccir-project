#!/usr/bin/env python3
"""Compare paired retrieval evaluations and apply a predeclared gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path


def load_ranks(path: Path) -> tuple[dict, list[int], list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    details = value.get("details")
    if not isinstance(details, list) or not details:
        raise ValueError(f"{path}: missing details")
    ranks: list[int] = []
    query_hashes: list[str] = []
    for expected, item in enumerate(details):
        if item.get("row_index") != expected:
            raise ValueError(f"{path}: non-contiguous row_index at {expected}")
        rank = item.get("best_positive_rank")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{path}: invalid rank at {expected}")
        query_hash = item.get("query_sha256")
        if not isinstance(query_hash, str) or len(query_hash) != 64:
            raise ValueError(f"{path}: invalid query hash at {expected}")
        ranks.append(rank)
        query_hashes.append(query_hash)
    if value.get("rows") != len(ranks):
        raise ValueError(f"{path}: row count mismatch")
    return value, ranks, query_hashes


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(sorted_values: list[float], fraction: float) -> float:
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def paired_delta(raw_ranks: list[int], clean_ranks: list[int], bootstrap_samples: int, seed: int) -> dict:
    if len(raw_ranks) != len(clean_ranks):
        raise ValueError("paired evaluations have different row counts")
    n = len(raw_ranks)
    per_query = {
        "recall_at_1": [float(c <= 1) - float(r <= 1) for r, c in zip(raw_ranks, clean_ranks)],
        "recall_at_5": [float(c <= 5) - float(r <= 5) for r, c in zip(raw_ranks, clean_ranks)],
        "recall_at_10": [float(c <= 10) - float(r <= 10) for r, c in zip(raw_ranks, clean_ranks)],
        "mrr": [(1.0 / c) - (1.0 / r) for r, c in zip(raw_ranks, clean_ranks)],
    }
    rng = random.Random(seed)
    sampled = {name: [] for name in per_query}
    for _ in range(bootstrap_samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        for name, values in per_query.items():
            sampled[name].append(sum(values[index] for index in indexes) / n)
    result = {}
    for name, values in per_query.items():
        distribution = sorted(sampled[name])
        result[name] = {
            "delta": mean(values),
            "ci95_low": percentile(distribution, 0.025),
            "ci95_high": percentile(distribution, 0.975),
        }
        if not all(math.isfinite(number) for number in result[name].values()):
            raise RuntimeError(f"non-finite result for {name}")
    result["queries_improved_rank"] = sum(c < r for r, c in zip(raw_ranks, clean_ranks))
    result["queries_tied_rank"] = sum(c == r for r, c in zip(raw_ranks, clean_ranks))
    result["queries_degraded_rank"] = sum(c > r for r, c in zip(raw_ranks, clean_ranks))
    return result


def compare(pairs: list[list[str]], output: Path, bootstrap_samples: int, seed: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if bootstrap_samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    comparisons = []
    for index, (label, raw_name, clean_name) in enumerate(pairs):
        raw_path, clean_path = Path(raw_name), Path(clean_name)
        raw_value, raw_ranks, raw_hashes = load_ranks(raw_path)
        clean_value, clean_ranks, clean_hashes = load_ranks(clean_path)
        if raw_value.get("input") != clean_value.get("input"):
            raise ValueError(f"{label}: evaluation inputs differ")
        if raw_hashes != clean_hashes or len(set(raw_hashes)) != len(raw_hashes):
            raise ValueError(f"{label}: paired query identities differ or are duplicated")
        result = paired_delta(raw_ranks, clean_ranks, bootstrap_samples, seed + index)
        result.update({"label": label, "raw_eval": str(raw_path.resolve()), "cleaned_eval": str(clean_path.resolve()), "rows": len(raw_ranks), "input": raw_value.get("input")})
        comparisons.append(result)
    by_label = {item["label"]: item for item in comparisons}
    required = {"checkpoint-500", "final-1000"}
    if set(by_label) != required:
        raise ValueError(f"required pair labels are {sorted(required)}")
    directional = all(item[metric]["delta"] > 0 for item in comparisons for metric in ("recall_at_1", "mrr"))
    final = by_label["final-1000"]
    robust_final = final["recall_at_1"]["ci95_low"] >= 0 and final["mrr"]["ci95_low"] >= 0
    no_final_topk_regression = final["recall_at_5"]["delta"] >= 0 and final["recall_at_10"]["delta"] >= 0
    gate = {
        "passed": directional and robust_final and no_final_topk_regression,
        "criteria": {
            "checkpoint_500_and_final_1000_r1_and_mrr_positive": directional,
            "final_1000_r1_and_mrr_ci95_lower_nonnegative": robust_final,
            "final_1000_r5_and_r10_nonnegative": no_final_topk_regression,
        },
        "interpretation": "Training loss is not a gate. Passing authorizes one cleaned full epoch from M00; failure stops full training.",
    }
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "comparisons": comparisons,
        "gate": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "RAW", "CLEANED"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    print(json.dumps(compare(args.pair, args.output, args.bootstrap_samples, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
