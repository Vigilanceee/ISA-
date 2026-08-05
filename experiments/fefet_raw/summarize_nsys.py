#!/usr/bin/env python3
"""Extract raw FeFET kernel shares from an Nsight Systems CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


RAW_PATTERNS = {
    "forward": re.compile(r"fefet_fwd", re.IGNORECASE),
    "grad_voltage": re.compile(r"fefet_grad_v", re.IGNORECASE),
    "grad_weights": re.compile(r"fefet_grad_w", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_column(fieldnames: list[str], *needles: str) -> str:
    for field in fieldnames:
        candidate = normalize(field)
        if all(needle in candidate for needle in needles):
            return field
    raise KeyError(f"could not find column containing {needles}; columns={fieldnames}")


def numeric(value: str) -> float:
    return float(value.strip().replace(",", ""))


def main() -> None:
    args = parse_args()
    with args.csv_report.open(newline="", encoding="utf-8-sig") as handle:
        lines = [line for line in handle if line.strip()]
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.lstrip().startswith("Time (%)")
        )
    except StopIteration as error:
        raise ValueError(f"Nsight CSV header not found in {args.csv_report}") from error
    reader = csv.DictReader(lines[header_index:])
    fieldnames = list(reader.fieldnames or [])
    name_column = find_column(fieldnames, "name")
    total_column = find_column(fieldnames, "totaltime")
    rows = list(reader)

    all_duration = sum(numeric(row[total_column]) for row in rows)
    raw_rows: list[dict[str, object]] = []
    category_duration = {key: 0.0 for key in RAW_PATTERNS}
    for row in rows:
        name = row[name_column]
        matched = next((key for key, pattern in RAW_PATTERNS.items() if pattern.search(name)), None)
        if matched is None:
            continue
        duration = numeric(row[total_column])
        category_duration[matched] += duration
        raw_rows.append({"category": matched, "kernel": name, "total_time_ns": duration})

    raw_duration = sum(category_duration.values())
    summary = {
        "status": "PASS" if all(category_duration.values()) else "FAIL",
        "trace_contains_all_raw_kernels": bool(all(category_duration.values())),
        "all_cuda_kernel_time_ns": all_duration,
        "raw_fefet_kernel_time_ns": raw_duration,
        "raw_fefet_share_of_all_cuda_kernel_time": raw_duration / all_duration if all_duration else None,
        "raw_kernel_category_time_ns": category_duration,
        "raw_kernel_category_share": {
            key: value / raw_duration if raw_duration else None for key, value in category_duration.items()
        },
        "matching_rows": raw_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
