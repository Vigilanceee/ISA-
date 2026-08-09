#!/usr/bin/env python3
"""Validate that all nine GPT-2 CIM checkpoints have complete formal results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODELS = (
    "tiny_standard",
    "mid_standard",
    "small_standard",
    "tiny_hybrid",
    "mid_hybrid",
    "small_hybrid",
    "tiny_physical",
    "mid_physical",
    "small_physical",
)
EXPECTED_TASKS = ("owt", "tinystories", "blimp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    errors: list[str] = []
    rows: list[dict] = []

    for model in EXPECTED_MODELS:
        path = result_dir / f"{model}.json"
        if not path.is_file():
            errors.append(f"{model}: missing result file {path}")
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{model}: invalid JSON ({exc})")
            continue

        metadata = payload.get("metadata", {})
        tasks = payload.get("tasks", {})
        missing = [task for task in EXPECTED_TASKS if task not in tasks]
        if missing:
            errors.append(f"{model}: missing tasks {', '.join(missing)}")
            continue

        required_keys = {
            "owt": "ppl",
            "tinystories": "ppl",
            "blimp": "acc",
        }
        for task, key in required_keys.items():
            if key not in tasks[task]:
                errors.append(f"{model}: {task} is missing metric '{key}'")

        rows.append(
            {
                "model": model,
                "checkpoint_step": metadata.get("checkpoint_step"),
                "checkpoint": metadata.get("checkpoint"),
                "owt_ppl": tasks["owt"].get("ppl"),
                "tinystories_ppl": tasks["tinystories"].get("ppl"),
                "blimp_acc_percent": (
                    None if tasks["blimp"].get("acc") is None
                    else 100.0 * float(tasks["blimp"]["acc"])
                ),
                "owt_tokens": tasks["owt"].get("tokens"),
                "tinystories_tokens": tasks["tinystories"].get("tokens"),
                "blimp_examples": tasks["blimp"].get("examples"),
            }
        )

    audit_path = result_dir / "easy_language_results_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "complete": not errors,
                "expected_models": list(EXPECTED_MODELS),
                "expected_tasks": list(EXPECTED_TASKS),
                "errors": errors,
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if errors:
        for error in errors:
            print(f"[error] {error}")
        print(f"[audit] {audit_path}")
        raise SystemExit(1)

    print("[ok] all nine models have OWT, TinyStories, and BLiMP results")
    for row in rows:
        print(
            f"{row['model']:16s} "
            f"OWT={row['owt_ppl']:.2f} "
            f"TinyStories={row['tinystories_ppl']:.2f} "
            f"BLiMP={row['blimp_acc_percent']:.2f}% "
            f"step={row['checkpoint_step']}"
        )
    print(f"[audit] {audit_path}")


if __name__ == "__main__":
    main()
