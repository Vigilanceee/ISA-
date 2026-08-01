#!/usr/bin/env python3
"""Merge resumable 24-state Monte Carlo shards and report uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    by_seed: dict[int, dict[str, str]] = {}
    for path in sorted(root.glob("shard_*/samples.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seed = int(row["seed"])
                if seed in by_seed and by_seed[seed] != row:
                    raise ValueError(f"Conflicting duplicate seed {seed}")
                by_seed[seed] = row

    expected = set(range(args.base_seed, args.base_seed + args.expected_seeds))
    found = set(by_seed)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise RuntimeError(
            f"Monte Carlo shards incomplete: missing={missing[:16]}, extra={extra[:16]}"
        )

    rows = [by_seed[seed] for seed in sorted(by_seed)]
    merged_tmp = root / f"samples.csv.{os.getpid()}.tmp"
    with merged_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(merged_tmp, root / "samples.csv")

    accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
    std = float(accuracy.std(ddof=1)) if accuracy.size > 1 else 0.0
    center_path = root / "center" / "center_metrics.json"
    center = json.loads(center_path.read_text()) if center_path.exists() else None
    summary = {
        "status": "completed",
        "sampling": "independent_per_physical_cell_uniform_over_24state_members",
        "sample_count": int(accuracy.size),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": std,
        "accuracy_sem": std / math.sqrt(accuracy.size),
        "accuracy_mean_ci95_low": float(
            accuracy.mean() - 1.96 * std / math.sqrt(accuracy.size)
        ),
        "accuracy_mean_ci95_high": float(
            accuracy.mean() + 1.96 * std / math.sqrt(accuracy.size)
        ),
        "accuracy_min": float(accuracy.min()),
        "accuracy_p1": float(np.percentile(accuracy, 1)),
        "accuracy_p5": float(np.percentile(accuracy, 5)),
        "accuracy_median": float(np.percentile(accuracy, 50)),
        "accuracy_p95": float(np.percentile(accuracy, 95)),
        "accuracy_p99": float(np.percentile(accuracy, 99)),
        "accuracy_max": float(accuracy.max()),
        "center_curve_accuracy": None if center is None else center.get("accuracy"),
        "mean_minus_center_pp": (
            None
            if center is None
            else float(accuracy.mean() - float(center["accuracy"]))
        ),
        "seeds": sorted(by_seed),
    }
    atomic_text(
        root / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
