#!/usr/bin/env python3
"""Validate the shape and numeric contents of the reference-result manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="results/reference_results.json")
    args = parser.parse_args()

    path = Path(args.reference)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if len(rows) != 5:
        raise SystemExit(f"Expected five benchmark rows, found {len(rows)}")

    for row in rows:
        for variant in ("digital", "hybrid", "physical"):
            values = row.get(variant)
            if not isinstance(values, list) or len(values) != 3:
                raise SystemExit(
                    f"{row.get('evaluation')}/{variant}: expected S/M/L values"
                )
            if not all(isinstance(value, (int, float)) for value in values):
                raise SystemExit(
                    f"{row.get('evaluation')}/{variant}: non-numeric metric"
                )
    print(f"[ok] {path}: five benchmarks × three variants × three sizes")


if __name__ == "__main__":
    main()
