#!/usr/bin/env python3
"""Select fixed-assignment measured compensation and enforce its gate."""

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
    parser.add_argument("--compensation-root", type=Path, required=True)
    parser.add_argument("--assignment-selection", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=65.0)
    parser.add_argument("--min-improvement-pp", type=float, default=1.0)
    args = parser.parse_args()

    root = args.compensation_root.resolve()
    assignment = json.loads(args.assignment_selection.read_text())
    assignment_accuracy = float(assignment["accuracy"])
    rows: list[dict] = []
    for metrics_path in sorted(root.glob("b*/metrics.json")):
        checkpoint_path = metrics_path.parent / "best_checkpoint.pth"
        try:
            payload = json.loads(metrics_path.read_text())
            status = payload["status"]
            best_accuracy = float(payload["best_measured_accuracy"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if status != "completed" or not checkpoint_path.is_file():
            continue
        final_measured = payload.get("measured") or {}
        final_continuous = payload.get("continuous") or {}
        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "best_measured_accuracy": best_accuracy,
                "final_measured_accuracy": final_measured.get("accuracy"),
                "final_continuous_accuracy": final_continuous.get("accuracy"),
                "improvement_over_assignment_pp": best_accuracy - assignment_accuracy,
                "checkpoint": str(checkpoint_path.resolve()),
                "metrics": str(metrics_path.resolve()),
            }
        )

    if not rows:
        raise SystemExit("No completed fixed-assignment compensation was found")
    rows.sort(key=lambda row: -row["best_measured_accuracy"])
    summary_path = root / "compensation_matrix_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    passed = (
        best["best_measured_accuracy"] >= args.min_accuracy
        and best["improvement_over_assignment_pp"] >= args.min_improvement_pp
    )
    gate = {
        "status": "PASS" if passed else "FAIL",
        "assignment_accuracy": assignment_accuracy,
        "min_accuracy": args.min_accuracy,
        "min_improvement_pp": args.min_improvement_pp,
        "best": best,
        "candidate_count": len(rows),
        "summary": str(summary_path.resolve()),
    }
    write_json_atomic(root / "best_compensation_metrics.json", best)
    write_json_atomic(root / "compensation_gate.json", gate)
    (root / "best_checkpoint.txt").write_text(best["checkpoint"] + "\n")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
