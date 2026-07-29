#!/usr/bin/env python3
"""Summarize the minimal language-model usability evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERRED_ORDER = [
    "tiny_standard", "mid_standard", "small_standard",
    "tiny_hybrid", "mid_hybrid", "small_hybrid",
    "tiny_physical", "mid_physical", "small_physical",
]


def metric(payload, task, key, scale=1.0):
    item = payload.get("tasks", {}).get(task, {})
    return "—" if key not in item else f"{item[key] * scale:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()

    root = Path(args.result_dir)
    available = {path.stem for path in root.glob("*.json")}
    if args.models:
        names = [name for name in args.models if name in available]
    else:
        names = [name for name in PREFERRED_ORDER if name in available]

    lines = [
        "| Model | Type | d | Context | OWT val PPL ↓ | TinyStories PPL ↓ | BLiMP Acc. ↑ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        row = [
            name,
            str(metadata.get("model_type", "—")),
            str(metadata.get("embed_dim", "—")),
            str(metadata.get("max_seq_len", "—")),
            metric(payload, "owt", "ppl"),
            metric(payload, "tinystories", "ppl"),
            metric(payload, "blimp", "acc", 100.0),
        ]
        lines.append("| " + " | ".join(row) + " |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {output}")


if __name__ == "__main__":
    main()
