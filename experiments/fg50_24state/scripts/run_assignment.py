#!/usr/bin/env python3
"""Run the existing activation-aware assignment search on the 24-state LUT."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

runtime = os.environ.get("FG50_24STATE_RUNTIME", "")
if not runtime:
    raise SystemExit("FG50_24STATE_RUNTIME must point to fg50_24state_runtime.npz")
runtime_path = Path(runtime).expanduser().resolve()
if not runtime_path.is_file():
    raise SystemExit(f"FG50 runtime does not exist: {runtime_path}")

from isa.measured_deployment import assignment as target
from isa.measured_deployment.fg50_loader import load_center_library
from isa.measured_deployment.fused_backend import install_fused_backend

original_load = target.IVLibrary.load.__func__


def load_override(
    cls,
    path,
    sheet,
    current_scale,
    min_valid_current_raw=None,
):
    if Path(path).expanduser().resolve() == runtime_path:
        return load_center_library(runtime_path, current_scale)
    return original_load(
        cls, path, sheet, current_scale, min_valid_current_raw
    )


target.IVLibrary.load = classmethod(load_override)
install_fused_backend()
target.main()


def cli_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Missing required CLI value {name}") from error


out_dir = Path(cli_value("--output"))
assignment_name = (
    "qpoint_assignment.pt"
    if "--qpoint-only" in sys.argv
    else "activation_aware_assignment.pt"
)
assignment_path = out_dir / assignment_name
payload = torch.load(assignment_path, map_location="cpu")
block_ids = sorted(
    int(key.split(".")[1])
    for key in payload
    if key.startswith("blocks.") and key.endswith(".pos_idx")
)
if not block_ids or block_ids != list(range(max(block_ids) + 1)):
    raise RuntimeError("Assignment block ids are missing or non-contiguous")
for block_id in block_ids:
    pos = payload[f"blocks.{block_id}.pos_idx"]
    neg = payload[f"blocks.{block_id}.neg_idx"]
    if pos.ndim != 2 or neg.shape != pos.shape:
        raise RuntimeError(f"Invalid assignment shape in block {block_id}")
    if int(pos.min()) < 0 or int(neg.min()) < 0:
        raise RuntimeError(f"Negative state id in block {block_id}")
    if int(pos.max()) >= 24 or int(neg.max()) >= 24:
        raise RuntimeError(f"State id outside 24-state codebook in block {block_id}")

completion = {
    "status": "completed",
    "assignment": str(assignment_path.resolve()),
    "block_count": len(block_ids),
    "state_count": 24,
}
temporary = out_dir / f"assignment_complete.json.{os.getpid()}.tmp"
temporary.write_text(json.dumps(completion, indent=2) + "\n")
os.replace(temporary, out_dir / "assignment_complete.json")
