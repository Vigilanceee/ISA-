#!/usr/bin/env python3
"""Use the compact 24-state runtime in the existing cell-wise MC evaluator.

The center-state assignment stays fixed.  Each seed samples one raw member
curve per positive/negative physical cell on the GPU, then keeps that complete
curve realization fixed for the full validation set.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

runtime = os.environ.get("FG50_24STATE_RUNTIME", "")
if not runtime:
    raise SystemExit("FG50_24STATE_RUNTIME must point to fg50_24state_runtime.npz")
runtime_path = Path(runtime).expanduser().resolve()
if not runtime_path.is_file():
    raise SystemExit(f"FG50 runtime does not exist: {runtime_path}")

from isa.measured_deployment.fg50_loader import load_center_library
from isa.measured_deployment.fused_backend import install_fused_backend
from isa.measured_deployment.ragged import sample_member_ids

requested_module = os.environ.get(
    "FG50_MC_MODULE", "isa.measured_deployment.monte_carlo"
)
try:
    target = importlib.import_module(requested_module)
except ModuleNotFoundError:
    if requested_module != "isa.measured_deployment.monte_carlo":
        raise
    target = importlib.import_module("isa.measured_deployment.monte_carlo")
# Some remote revisions expose a thin FG50 adapter whose main function is
# defined in the generic module. Patch the defining module as well as the
# adapter so both layouts behave identically.
implementation = sys.modules[target.main.__module__]


with np.load(runtime_path, allow_pickle=False) as payload:
    voltage_v = torch.from_numpy(
        np.asarray(payload["voltage_v"], dtype=np.float32)
    ).contiguous()
    currents_a = torch.from_numpy(
        np.asarray(payload["monotone_current_na"], dtype=np.float32)
    ).contiguous()
    curve_names = np.asarray(payload["curve_ids"]).reshape(-1).astype(str).tolist()
    member_ids_by_state = torch.from_numpy(
        np.asarray(payload["member_ids_by_state"], dtype=np.int32)
    ).contiguous()
    state_offsets = torch.from_numpy(
        np.asarray(payload["state_offsets"], dtype=np.int32)
    ).contiguous()

state_count = int(state_offsets.numel() - 1)
state_to_curve_ids: dict[int, tuple[int, ...]] = {
    state_id: tuple(
        int(value)
        for value in member_ids_by_state[
            int(state_offsets[state_id]) : int(state_offsets[state_id + 1])
        ].tolist()
    )
    for state_id in range(state_count)
}
state_members: dict[int, list[str]] = {
    state_id: [curve_names[curve_id] for curve_id in curve_ids]
    for state_id, curve_ids in state_to_curve_ids.items()
}
device_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

original_iv_load = implementation.IVLibrary.load.__func__
original_parse_args = implementation.parse_args


def load_center_override(
    cls,
    path,
    sheet,
    current_scale,
    min_valid_current_raw=None,
):
    if Path(path).expanduser().resolve() == runtime_path:
        return load_center_library(runtime_path, current_scale)
    return original_iv_load(
        cls, path, sheet, current_scale, min_valid_current_raw
    )


def load_state_members_override(_path, _fold_id, _codebook_sheet):
    return state_members


def load_member_library_override(args):
    return implementation.MemberCurveLibrary(
        vg=voltage_v,
        currents_a=(currents_a * float(args.measured_current_scale)).contiguous(),
        curve_names=curve_names,
        state_to_curve_ids=state_to_curve_ids,
    )


def tables_for(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = str(device)
    cached = device_tables.get(key)
    if cached is None:
        ids = member_ids_by_state.to(device=device, dtype=torch.int32)
        offsets = state_offsets.to(device=device, dtype=torch.int32)
        counts = (offsets[1:] - offsets[:-1]).to(torch.long)
        cached = (ids, offsets, counts)
        device_tables[key] = cached
    return cached


@torch.no_grad()
def set_random_member_realization(
    model,
    state_assignment,
    member_library,
    seed: int,
) -> dict[str, int]:
    del member_library
    first_layer = next(iter(implementation.iter_deployment_fc1(model)))[1]
    device = first_layer.vth_pos.device
    flat_ids, offsets, member_counts = tables_for(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    seen = torch.zeros(flat_ids.numel(), dtype=torch.bool, device=device)
    total_cells = 0
    singleton_cells = torch.zeros((), dtype=torch.long, device=device)

    for block_id, layer in implementation.iter_deployment_fc1(model):
        pos_state_cpu, neg_state_cpu = state_assignment[block_id]
        pos_state = pos_state_cpu.to(device=device, dtype=torch.int32)
        neg_state = neg_state_cpu.to(device=device, dtype=torch.int32)
        pos_curve = sample_member_ids(
            pos_state, flat_ids, offsets, generator
        )
        neg_curve = sample_member_ids(
            neg_state, flat_ids, offsets, generator
        )
        layer.set_assignment(pos_curve, neg_curve)
        seen[pos_curve.long()] = True
        seen[neg_curve.long()] = True
        total_cells += pos_state.numel() + neg_state.numel()
        singleton_cells += (member_counts[pos_state.long()] == 1).sum()
        singleton_cells += (member_counts[neg_state.long()] == 1).sum()

    return {
        "total_physical_cells": int(total_cells),
        "singleton_state_cells": int(singleton_cells.item()),
        "unique_raw_curves_used": int(seen.sum().item()),
    }


def atomic_append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    if any(int(item["seed"]) == int(row["seed"]) for item in existing):
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def path_identity(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def parse_args_override():
    args = original_parse_args()
    identity = {
        "runtime": path_identity(str(runtime_path)),
        "checkpoint": path_identity(args.checkpoint),
        "fixed_assignment": path_identity(args.fixed_assignment),
        "model_scale": args.model_scale,
        "measured_current_scale": float(args.measured_current_scale),
        "min_valid_current_na": float(args.min_valid_current_na),
        "vin_lut_bins": int(args.vin_lut_bins),
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity_path = out_dir / "resume_identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if previous != identity:
            raise RuntimeError(
                f"Refusing to mix incompatible Monte Carlo runs in {out_dir}"
            )
    else:
        implementation.atomic_json(identity_path, identity)
    return args


for module in {target, implementation}:
    module.IVLibrary.load = classmethod(load_center_override)
    module.load_state_members = load_state_members_override
    module.load_member_curve_library = load_member_library_override
    module.set_random_member_realization = set_random_member_realization
    module.append_csv = atomic_append_csv
    module.parse_args = parse_args_override
install_fused_backend()
target.main()
