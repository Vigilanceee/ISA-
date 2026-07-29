"""EKV lookup tables for fast approximate Triton kernels (BF16)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def _cpu_lut_cached(key: Tuple) -> tuple[torch.Tensor, torch.Tensor]:
    (
        n,
        v_sat,
        i_s,
        u_t,
        v_d,
        delta_min,
        delta_max,
        delta_size,
    ) = key
    del v_sat
    delta = torch.linspace(delta_min, delta_max, delta_size, dtype=torch.float32)
    inv_2nut = 1.0 / (2.0 * n * u_t)
    u1 = delta * inv_2nut
    u2 = (delta - n * v_d) * inv_2nut
    sp1 = F.softplus(u1)
    sp2 = F.softplus(u2)
    base = i_s * (sp1.square() - sp2.square())
    sig1 = torch.where(sp1 > 20.0, torch.ones_like(sp1), 1.0 - torch.exp(-sp1))
    sig2 = torch.where(sp2 > 20.0, torch.ones_like(sp2), 1.0 - torch.exp(-sp2))
    grad_delta = i_s * 2.0 * inv_2nut * (sp1 * sig1 - sp2 * sig2)
    return base.contiguous().to(torch.bfloat16), grad_delta.contiguous().to(torch.bfloat16)


_device_cache: dict[Tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def get_ekv_lut_tables(cfg: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_min = float(cfg.get("LUT_V_MIN", cfg.get("V_signed_min", -4.0)))
    v_max = float(cfg.get("LUT_V_MAX", cfg.get("V_signed_max", 4.0)))
    vth_min = float(cfg.get("LUT_VTH_MIN", cfg.get("V_TH_MIN", 0.0)))
    vth_max = float(cfg.get("LUT_VTH_MAX", cfg.get("V_TH_MAX", 5.0)))
    delta_min = float(cfg.get("LUT_DELTA_MIN", v_min - vth_max))
    delta_max = float(cfg.get("LUT_DELTA_MAX", v_max - vth_min))
    delta_size = int(cfg.get("LUT_DELTA_SIZE", 16384))
    cpu_key = (
        float(cfg["n"]),
        float(cfg["V_sat"]),
        float(cfg["I_S"]),
        float(cfg["U_T"]),
        float(cfg["V_D"]),
        delta_min,
        delta_max,
        delta_size,
    )
    cache_key = (str(device), cpu_key)
    if cache_key not in _device_cache:
        tables = _cpu_lut_cached(cpu_key)
        _device_cache[cache_key] = tuple(t.to(device=device, non_blocking=True) for t in tables)
    return _device_cache[cache_key]
