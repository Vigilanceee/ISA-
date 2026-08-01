"""Load the compact FG50 24-state center codebook into the IVLibrary API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_center_library(runtime_path: str | Path, current_scale: float):
    """Create an :class:`IVLibrary` from a generated FG50 runtime archive."""
    from isa.measured_deployment.codebook import IVLibrary

    path = Path(runtime_path).expanduser().resolve()
    payload = np.load(path, allow_pickle=False)
    voltage = np.asarray(payload["voltage_v"], dtype=np.float32)
    curves_na = np.asarray(payload["center_curves_na"], dtype=np.float32)
    if curves_na.shape != (24, voltage.size):
        raise ValueError(
            f"expected 24 center curves on {voltage.size} points, got {curves_na.shape}"
        )
    return IVLibrary(
        torch.from_numpy(voltage),
        torch.from_numpy(curves_na * float(current_scale)),
        [f"State_{state + 1:02d}" for state in range(24)],
        float(current_scale),
        float(np.min(curves_na)),
        float(np.max(curves_na)),
        None,
        int(curves_na.size),
        0,
        0,
        0,
    )
