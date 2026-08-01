#!/usr/bin/env python3
"""Run the existing post-training program with the 24-state NPZ + fused LUT."""

from __future__ import annotations

import os
from pathlib import Path

runtime = os.environ.get("FG50_24STATE_RUNTIME", "")
if not runtime:
    raise SystemExit("FG50_24STATE_RUNTIME must point to fg50_24state_runtime.npz")
runtime_path = Path(runtime).expanduser().resolve()
if not runtime_path.is_file():
    raise SystemExit(f"FG50 runtime does not exist: {runtime_path}")

from isa.measured_deployment import posttrain as target
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
