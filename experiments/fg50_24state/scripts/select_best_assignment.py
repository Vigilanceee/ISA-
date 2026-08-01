#!/usr/bin/env python3
"""Select an activation-aware assignment and enforce an accuracy gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=60.0)
    parser.add_argument("--min-improvement-pp", type=float, default=3.0)
    args = parser.parse_args()

    root = args.assignment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads(args.baseline_metrics.read_text())
    baseline_accuracy = float(baseline_payload["baseline_measured"]["accuracy"])

    rows: list[dict] = []
    for metrics_path in sorted(root.glob("a*/metrics.json")):
        assignment_path = metrics_path.parent / "activation_aware_assignment.pt"
        try:
            payload = json.loads(metrics_path.read_text())
            measured = payload["activation_aware_measured"]
            run_args = payload["args"]
            status = payload["status"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if status != "completed" or not assignment_path.is_file():
            continue
        accuracy = float(measured["accuracy"])
        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "accuracy": accuracy,
                "loss": float(measured["loss"]),
                "improvement_pp": accuracy - baseline_accuracy,
                "topk": int(run_args["topk"]),
                "max_tokens_per_layer": int(run_args["max_tokens_per_layer"]),
                "coord_block": int(run_args["coord_block"]),
                "seed": int(run_args["seed"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
                "assignment": str(assignment_path.resolve()),
                "metrics": str(metrics_path.resolve()),
            }
        )

    if not rows:
        raise SystemExit("No completed activation-aware assignment was found")
    rows.sort(key=lambda row: (-row["accuracy"], row["loss"]))

    summary_path = root / "assignment_matrix_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    passed = (
        best["accuracy"] >= args.min_accuracy
        and best["improvement_pp"] >= args.min_improvement_pp
    )
    gate = {
        "status": "PASS" if passed else "FAIL",
        "baseline_accuracy": baseline_accuracy,
        "min_accuracy": args.min_accuracy,
        "min_improvement_pp": args.min_improvement_pp,
        "best": best,
        "candidate_count": len(rows),
        "summary": str(summary_path.resolve()),
    }
    write_json_atomic(root / "best_assignment_metrics.json", best)
    write_json_atomic(root / "assignment_gate.json", gate)
    (root / "best_assignment.txt").write_text(best["assignment"] + "\n")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
