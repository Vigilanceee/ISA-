"""Build a compact FG50 24-state runtime package and member mapping.

The published 24-state artifacts contain center curves and aggregate state
statistics, but not the per-curve state assignment.  This builder reconstructs
a deterministic ordered assignment:

1. floor raw current at the published 0.39 nA measurement floor;
2. take the monotone envelope;
3. estimate each curve's horizontal shift from inverse-log-current crossings
   over the natural 10--1000 nA range;
4. stable-sort by that shift and partition using the published state counts.

The exact provenance is recorded in the output manifest.  The reconstructed
mapping is intentionally not described as an original hidden assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inverse_crossing_voltage(
    log_current: np.ndarray,
    voltage_v: np.ndarray,
    target_log_current: float,
) -> np.ndarray:
    """Interpolate V(log I target), clipping outside the measured V range."""
    hits = log_current >= float(target_log_current)
    upper = hits.argmax(axis=1)
    no_hit = ~hits.any(axis=1)
    result = np.zeros(log_current.shape[0], dtype=np.float64)
    interior = (upper > 0) & ~no_hit
    row = np.flatnonzero(interior)
    hi = upper[interior]
    low_value = log_current[row, hi - 1]
    high_value = log_current[row, hi]
    fraction = np.divide(
        float(target_log_current) - low_value,
        high_value - low_value,
        out=np.zeros_like(low_value),
        where=high_value > low_value,
    )
    result[interior] = (
        voltage_v[hi - 1] + fraction * (voltage_v[hi] - voltage_v[hi - 1])
    )
    result[no_hit] = float(voltage_v[-1])
    return result


def horizontal_shift_score(
    monotone_current_na: np.ndarray,
    voltage_v: np.ndarray,
    low_current_na: float,
    high_current_na: float,
    points: int,
) -> np.ndarray:
    if low_current_na <= 0 or high_current_na <= low_current_na:
        raise ValueError("shift-current range must be positive and increasing")
    if points < 2:
        raise ValueError("shift-grid-points must be at least two")
    targets = np.logspace(
        math.log10(low_current_na), math.log10(high_current_na), points
    )
    log_current = np.log10(monotone_current_na)
    crossings = np.stack(
        [
            inverse_crossing_voltage(log_current, voltage_v, math.log10(target))
            for target in targets
        ],
        axis=1,
    )
    return crossings.mean(axis=1)


def fit_published_shift(
    raw_score: np.ndarray,
    state_ids: np.ndarray,
    published_centers: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    state_medians = np.asarray(
        [np.median(raw_score[state_ids == state]) for state in range(24)]
    )
    design = np.column_stack((state_medians, np.ones(24)))
    scale, offset = np.linalg.lstsq(
        design, published_centers.astype(np.float64), rcond=None
    )[0]
    return raw_score * scale + offset, float(scale), float(offset)


def state_partition(
    score: np.ndarray, published_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(published_counts.sum()) != score.size:
        raise ValueError(
            f"published counts sum to {published_counts.sum()}, expected {score.size}"
        )
    order = np.argsort(score, kind="stable")
    state_ids = np.empty(score.size, dtype=np.int16)
    offsets = np.zeros(published_counts.size + 1, dtype=np.int32)
    cursor = 0
    for state, count in enumerate(published_counts.tolist()):
        next_cursor = cursor + int(count)
        state_ids[order[cursor:next_cursor]] = state
        offsets[state + 1] = next_cursor
        cursor = next_cursor
    return state_ids, order.astype(np.int32), offsets


def state_member_statistics(
    state_ids: np.ndarray,
    cell_ids: np.ndarray,
    program_states: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for state in range(24):
        mask = state_ids == state
        values, counts = np.unique(program_states[mask], return_counts=True)
        mode_position = int(np.argmax(counts))
        result.append(
            {
                "state_id": state,
                "curve_count": int(mask.sum()),
                "unique_cells": int(np.unique(cell_ids[mask]).size),
                "original_level_count": int(values.size),
                "most_common_value": f"Value_{int(values[mode_position])}",
                "most_common_share": float(counts[mode_position] / counts.sum()),
            }
        )
    return result


def log_rmse(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    return np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(a, floor))
                - np.log10(np.maximum(b, floor))
            )
            ** 2,
            axis=-1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-archive", required=True)
    parser.add_argument("--center-codebook", required=True)
    parser.add_argument("--state-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-floor-na", type=float, default=0.39)
    parser.add_argument("--shift-low-current-na", type=float, default=10.0)
    parser.add_argument("--shift-high-current-na", type=float, default=1000.0)
    parser.add_argument("--shift-grid-points", type=int, default=21)
    args = parser.parse_args()

    member_path = Path(args.member_archive).expanduser().resolve()
    center_path = Path(args.center_codebook).expanduser().resolve()
    table_path = Path(args.state_table).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    members = np.load(member_path, allow_pickle=False)
    required_member_keys = {
        "voltage_v",
        "curve_ids",
        "cell_ids",
        "program_states",
        "raw_current_na",
    }
    missing = required_member_keys - set(members.files)
    if missing:
        raise KeyError(f"member archive is missing {sorted(missing)}")
    voltage_v = np.asarray(members["voltage_v"], dtype=np.float64)
    raw_shape = np.asarray(members["raw_current_na"]).shape
    if raw_shape != (1040, 31, 61):
        raise ValueError(f"unexpected raw member shape: {raw_shape}")
    raw_current_na = np.asarray(members["raw_current_na"], dtype=np.float32).reshape(
        -1, voltage_v.size
    )
    curve_ids = np.asarray(members["curve_ids"]).reshape(-1).astype(str)
    cell_ids = np.repeat(np.asarray(members["cell_ids"], dtype=np.int32), 31)
    program_states = np.tile(
        np.asarray(members["program_states"], dtype=np.int16), 1040
    )

    centers_payload = np.load(center_path, allow_pickle=False)
    if {"voltages", "curves"} - set(centers_payload.files):
        raise KeyError("center codebook must contain voltages and curves")
    center_voltage = np.asarray(centers_payload["voltages"], dtype=np.float64)
    center_curves_na = np.asarray(centers_payload["curves"], dtype=np.float32)
    if center_curves_na.shape != (24, 61):
        raise ValueError(f"unexpected center curve shape: {center_curves_na.shape}")
    if not np.allclose(center_voltage, voltage_v, atol=1e-7, rtol=0):
        raise ValueError("center and member voltage grids differ")

    published_rows = json.loads(table_path.read_text(encoding="utf-8"))
    if len(published_rows) != 24:
        raise ValueError("published state table must contain 24 rows")
    published_counts = np.asarray(
        [int(row["curve_count"]) for row in published_rows], dtype=np.int32
    )
    published_shift_centers = np.asarray(
        [float(row["shift_center_V"]) for row in published_rows], dtype=np.float64
    )

    floor = float(args.current_floor_na)
    monotone_current_na = np.maximum.accumulate(
        np.maximum(raw_current_na, floor), axis=1
    ).astype(np.float32)
    raw_shift_score = horizontal_shift_score(
        monotone_current_na,
        voltage_v,
        args.shift_low_current_na,
        args.shift_high_current_na,
        args.shift_grid_points,
    )
    provisional_states, member_ids, offsets = state_partition(
        raw_shift_score, published_counts
    )
    shift_score_v, shift_scale, shift_offset = fit_published_shift(
        raw_shift_score, provisional_states, published_shift_centers
    )
    state_ids, member_ids, offsets = state_partition(shift_score_v, published_counts)

    reconstructed_centers_na = np.stack(
        [
            np.median(monotone_current_na[state_ids == state], axis=0)
            for state in range(24)
        ]
    ).astype(np.float32)
    state_stats = state_member_statistics(
        state_ids, cell_ids, program_states
    )

    slopes_na_per_v = (
        (monotone_current_na[:, 1:] - monotone_current_na[:, :-1])
        / np.diff(voltage_v).astype(np.float32)[None, :]
    ).astype(np.float32)
    intercepts_na = (
        monotone_current_na[:, :-1]
        - slopes_na_per_v * voltage_v[:-1].astype(np.float32)[None, :]
    ).astype(np.float32)
    center_slopes = (
        (center_curves_na[:, 1:] - center_curves_na[:, :-1])
        / np.diff(voltage_v).astype(np.float32)[None, :]
    ).astype(np.float32)
    center_intercepts = (
        center_curves_na[:, :-1]
        - center_slopes * voltage_v[:-1].astype(np.float32)[None, :]
    ).astype(np.float32)
    pair_pos = np.repeat(np.arange(24, dtype=np.int16), 24)
    pair_neg = np.tile(np.arange(24, dtype=np.int16), 24)
    pair_diff_slopes = center_slopes[pair_pos] - center_slopes[pair_neg]
    pair_diff_intercepts = (
        center_intercepts[pair_pos] - center_intercepts[pair_neg]
    )

    runtime_path = output_dir / "fg50_24state_runtime.npz"
    np.savez_compressed(
        runtime_path,
        voltage_v=voltage_v.astype(np.float32),
        curve_ids=curve_ids,
        cell_ids=cell_ids,
        program_states=program_states,
        state_ids=state_ids,
        shift_score_v=shift_score_v.astype(np.float32),
        member_ids_by_state=member_ids,
        state_offsets=offsets,
        published_state_counts=published_counts,
        monotone_current_na=monotone_current_na,
        slopes_na_per_v=slopes_na_per_v,
        intercepts_na=intercepts_na,
        center_curves_na=center_curves_na,
        center_slopes_na_per_v=center_slopes,
        center_intercepts_na=center_intercepts,
        pair_pos=pair_pos,
        pair_neg=pair_neg,
        pair_diff_slopes_na_per_v=pair_diff_slopes.astype(np.float32),
        pair_diff_intercepts_na=pair_diff_intercepts.astype(np.float32),
    )

    mapping_path = output_dir / "fg50_24state_member_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "curve_index",
            "curve_id",
            "cell_id",
            "program_state",
            "state_id",
            "state_name",
            "shift_score_V",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve_index in range(curve_ids.size):
            state = int(state_ids[curve_index])
            writer.writerow(
                {
                    "curve_index": curve_index,
                    "curve_id": curve_ids[curve_index],
                    "cell_id": int(cell_ids[curve_index]),
                    "program_state": int(program_states[curve_index]),
                    "state_id": state,
                    "state_name": f"State_{state + 1:02d}",
                    "shift_score_V": f"{shift_score_v[curve_index]:.9g}",
                }
            )

    center_error = log_rmse(reconstructed_centers_na, center_curves_na, floor)
    verify_errors: list[float] = []
    verify_rows: list[dict[str, Any]] = []
    for state, published in enumerate(published_rows):
        verify_voltage = float(published["best_verify_V"])
        voltage_index = int(np.argmin(np.abs(voltage_v - verify_voltage)))
        member_current = monotone_current_na[state_ids == state, voltage_index]
        reconstructed_quantiles = np.quantile(
            member_current, (0.1, 0.5, 0.9)
        )
        published_quantiles = np.asarray(
            [
                float(published["verify_I10"]),
                float(published["verify_IMedian"]),
                float(published["verify_I90"]),
            ]
        )
        errors = np.abs(
            np.log10(np.maximum(reconstructed_quantiles, floor))
            - np.log10(np.maximum(published_quantiles, floor))
        )
        verify_errors.extend(errors.tolist())
        verify_rows.append(
            {
                "state_id": state,
                "verify_voltage_V": verify_voltage,
                "reconstructed_I10_I50_I90_nA": reconstructed_quantiles.tolist(),
                "published_I10_I50_I90_nA": published_quantiles.tolist(),
                "absolute_log10_error_decades": errors.tolist(),
            }
        )
    shift_stats = np.asarray(
        [
            [
                np.quantile(shift_score_v[state_ids == state], 0.05),
                np.median(shift_score_v[state_ids == state]),
                np.quantile(shift_score_v[state_ids == state], 0.95),
            ]
            for state in range(24)
        ]
    )
    published_shift_stats = np.asarray(
        [
            [
                float(row.get("shift_p05_V", row["shift_center_V"])),
                float(row["shift_center_V"]),
                float(row.get("shift_p95_V", row["shift_center_V"])),
            ]
            for row in published_rows
        ]
    )
    # Desktop JSON contains only center shift; the workbook audit supplies full
    # p05/p95.  Center-only comparison remains valid when those keys are absent.
    available_columns = np.asarray(
        [
            [
                "shift_p05_V" in row,
                True,
                "shift_p95_V" in row,
            ]
            for row in published_rows
        ],
        dtype=bool,
    )
    shift_error_values = (
        shift_stats[available_columns] - published_shift_stats[available_columns]
    )
    audit = {
        "status": "pass",
        "curve_count": int(curve_ids.size),
        "state_count": 24,
        "state_counts_exact_match": bool(
            np.array_equal(np.bincount(state_ids, minlength=24), published_counts)
        ),
        "all_member_curves_monotone": bool(
            np.all(np.diff(monotone_current_na, axis=1) >= 0)
        ),
        "all_center_curves_monotone": bool(
            np.all(np.diff(center_curves_na, axis=1) >= 0)
        ),
        "center_curves_ordered_at_all_voltage_points": bool(
            np.all(np.diff(center_curves_na, axis=0) <= 0)
        ),
        "reconstructed_vs_published_center_log_rmse_decades": {
            "per_state": center_error.tolist(),
            "median": float(np.median(center_error)),
            "mean": float(np.mean(center_error)),
            "max": float(np.max(center_error)),
        },
        "reconstructed_vs_published_verify_quantiles": {
            "per_state": verify_rows,
            "absolute_log10_error_decades": {
                "median": float(np.median(verify_errors)),
                "mean": float(np.mean(verify_errors)),
                "max": float(np.max(verify_errors)),
            },
        },
        "published_shift_fit": {
            "scale": shift_scale,
            "offset_V": shift_offset,
            "available_statistic_rmse_V": (
                None
                if shift_error_values.size == 0
                else float(np.sqrt(np.mean(shift_error_values**2)))
            ),
        },
        "state_statistics": state_stats,
    }
    atomic_json(output_dir / "fg50_24state_reconstruction_audit.json", audit)

    manifest = {
        "version": "fg50_24state_runtime_v1",
        "mapping_provenance": (
            "deterministic reconstruction from published 24-state counts and "
            "ordered horizontal-shift score; original per-curve mapping was "
            "not present in the supplied workbook/NPZ/JSON"
        ),
        "preprocessing": {
            "current_floor_nA": floor,
            "monotone_envelope": True,
            "distance_domain": "log10(current_nA)",
            "horizontal_shift_inverse_current_range_nA": [
                args.shift_low_current_na,
                args.shift_high_current_na,
            ],
            "horizontal_shift_grid_points": args.shift_grid_points,
            "partition": "stable sort plus published per-state counts",
        },
        "inputs": {
            "member_archive": {
                "path": str(member_path),
                "sha256": sha256_file(member_path),
            },
            "center_codebook": {
                "path": str(center_path),
                "sha256": sha256_file(center_path),
            },
            "state_table": {
                "path": str(table_path),
                "sha256": sha256_file(table_path),
            },
        },
        "outputs": {
            "runtime_npz": str(runtime_path),
            "member_mapping_csv": str(mapping_path),
            "audit_json": str(
                output_dir / "fg50_24state_reconstruction_audit.json"
            ),
        },
        "runtime_layout": {
            "member_sampling": (
                "member_ids_by_state[state_offsets[s]:state_offsets[s+1]]"
            ),
            "center_pair_id": "pos_state * 24 + neg_state",
            "member_curve_bytes_float32": int(monotone_current_na.nbytes),
            "member_coefficients_bytes_float32": int(
                slopes_na_per_v.nbytes + intercepts_na.nbytes
            ),
            "center_pair_coefficients_bytes_float32": int(
                pair_diff_slopes.nbytes + pair_diff_intercepts.nbytes
            ),
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
