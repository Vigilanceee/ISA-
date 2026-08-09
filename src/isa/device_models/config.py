"""Shared physical configuration for the fitted Flash-transistor EKV model."""

from __future__ import annotations

import os
from typing import Any


FLASH_TRANSISTOR_PARAMETERS: dict[str, float] = {
    "n": 4.1360,
    "V_sat": 8.2624,
    "I_S": 7.5875e-7,
    "U_T": 0.0259,
    "V_D": 0.1,
    "R_TIA": 1e5,
    "V_TH_MIN": 0.0,
    "V_TH_MAX": 5.0,
    "V_min": 0.0,
    "V_max": 4.0,
    "V_signed_min": -4.0,
    "V_signed_max": 4.0,
}


def make_flash_transistor_ekv_config(default_lut_size: int) -> dict[str, Any]:
    """Return the shared device parameters plus runtime kernel settings."""
    return {
        **FLASH_TRANSISTOR_PARAMETERS,
        "FORWARD_CHUNK_SIZE": 64,
        "use_triton": True,
        "ekv_approx": "lut",
        "LUT_DELTA_SIZE": int(os.environ.get("EKV_LUT_SIZE", str(default_lut_size))),
        "TRITON_BLOCK_M": 16,
        "TRITON_BLOCK_O": 16,
        "TRITON_BLOCK_K": int(os.environ.get("EKV_TRITON_BLOCK_K", "8")),
        "TRITON_SPLIT_M": 8,
        "CUDA_SHARED_BACKWARD": os.environ.get("EKV_CUDA_SHARED_BACKWARD", "1") != "0",
        "R_TIA_LOG_SCALE_MIN": -2.0,
        "R_TIA_LOG_SCALE_MAX": 2.0,
    }


FLASH_TRANSISTOR_EKV_CONFIG = make_flash_transistor_ekv_config(default_lut_size=8192)
