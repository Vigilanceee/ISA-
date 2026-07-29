"""CIM-based physical FFN layers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from isa.device_models.config import FLASH_TRANSISTOR_EKV_CONFIG as PHYSICAL_CONFIG
from isa.device_models.flash_transistor import ekv_diff_current_fn

try:
    from isa.kernels.transformer_ffn.ekv_triton import triton_ekv_matmul
except Exception:
    triton_ekv_matmul = None
try:
    from isa.kernels.transformer_ffn.ekv_lut_triton import triton_lut_ekv_matmul
except Exception:
    triton_lut_ekv_matmul = None


class VoltageMapping(nn.Module):
    """Map external digital activations to the first-layer gate voltage range."""

    def __init__(self, voltage_max: float = 4.0):
        super().__init__()
        self.voltage_max = voltage_max
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def set_affine(self, scale: float, shift: float) -> None:
        self.scale.fill_(float(scale))
        self.shift.fill_(float(shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = (x.float() * self.scale + self.shift).clamp(0.0, self.voltage_max)
        return v.to(dtype=x.dtype)


def clamp_signed(v: torch.Tensor, v_min: float = -4.0, v_max: float = 4.0) -> torch.Tensor:
    return v.clamp(v_min, v_max)


class CIMLinear(nn.Module):
    """Compute-and-memory linear layer via EKV differential current and TIA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        physical_config: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = dict(physical_config or PHYSICAL_CONFIG)

        self.vth_pos = nn.Parameter(torch.empty(out_features, in_features))
        self.vth_neg = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer(
            "r_tia_base",
            torch.full((1,), float(self.cfg.get("R_TIA", 1e5)), dtype=torch.float32),
        )
        self.r_tia_log_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.track_output_stats = False
        self.register_buffer("_stat_total", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("_stat_clipped", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("_stat_abs_max", torch.zeros((), dtype=torch.float32), persistent=False)
        self.reset_vth(mode="centered_random", eps_k=0.5, symmetric=False)

    @property
    def r_tia(self) -> torch.Tensor:
        lo = float(self.cfg.get("R_TIA_LOG_SCALE_MIN", -2.0))
        hi = float(self.cfg.get("R_TIA_LOG_SCALE_MAX", 2.0))
        return self.r_tia_base * self.r_tia_log_scale.clamp(lo, hi).exp()

    def set_r_tia(self, value: float) -> None:
        with torch.no_grad():
            self.r_tia_base.fill_(float(value))
            self.r_tia_log_scale.zero_()

    def reset_output_stats(self) -> None:
        self._stat_total.zero_()
        self._stat_clipped.zero_()
        self._stat_abs_max.zero_()

    def set_output_stats_tracking(self, enabled: bool, reset: bool = False) -> None:
        self.track_output_stats = bool(enabled)
        if reset:
            self.reset_output_stats()

    def output_stats(self) -> Dict[str, float]:
        total = float(self._stat_total.item())
        clipped = float(self._stat_clipped.item())
        return {
            "total": total,
            "clipped": clipped,
            "clip_ratio": clipped / max(total, 1.0),
            "preclamp_abs_max": float(self._stat_abs_max.item()),
        }

    def _clamp_output(self, out: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        if self.track_output_stats:
            with torch.no_grad():
                detached = out.detach()
                self._stat_total.add_(detached.numel())
                self._stat_clipped.add_(((detached <= v_min) | (detached >= v_max)).sum())
                self._stat_abs_max.copy_(torch.maximum(self._stat_abs_max, detached.abs().max().float()))
        return clamp_signed(out, v_min, v_max)

    def reset_vth(
        self,
        center: float | torch.Tensor | None = None,
        eps_k: float = 0.5,
        symmetric: bool = True,
        mode: str = "centered_symmetric",
    ) -> None:
        v_min = float(self.cfg["V_TH_MIN"])
        v_max = float(self.cfg["V_TH_MAX"])
        if center is None:
            center_t = torch.tensor(0.5 * (v_min + v_max), device=self.vth_pos.device, dtype=self.vth_pos.dtype)
        elif torch.is_tensor(center):
            center_t = center.to(device=self.vth_pos.device, dtype=self.vth_pos.dtype)
        else:
            center_t = torch.tensor(float(center), device=self.vth_pos.device, dtype=self.vth_pos.dtype)
        center_t = center_t.clamp(v_min, v_max)
        eps_scale = float(eps_k) * 2.0 * float(self.cfg["n"]) * float(self.cfg["U_T"])
        with torch.no_grad():
            if mode in {"random", "centered_random"} or not symmetric:
                self.vth_pos.uniform_(-eps_scale, eps_scale).add_(center_t)
                self.vth_neg.uniform_(-eps_scale, eps_scale).add_(center_t)
            else:
                eps = torch.empty_like(self.vth_pos).normal_(0.0, eps_scale)
                self.vth_pos.copy_(center_t + eps)
                self.vth_neg.copy_(center_t - eps)
            self.clamp_vth()

    @torch.no_grad()
    def estimate_idiff(self, x: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features).contiguous()
        if flat.shape[0] > max_rows:
            flat = flat[:max_rows]
        if bool(self.cfg.get("use_triton", False)) and flat.is_cuda:
            approx = str(self.cfg.get("ekv_approx", "lut")).lower()
            if approx == "lut" and triton_lut_ekv_matmul is not None:
                return triton_lut_ekv_matmul(flat, self.vth_pos, self.vth_neg, self.cfg)
            if triton_ekv_matmul is not None:
                return triton_ekv_matmul(flat, self.vth_pos, self.vth_neg, self.cfg)
        chunk = int(self.cfg.get("FORWARD_CHUNK_SIZE", 64))
        outputs = []
        x_exp = flat.unsqueeze(-1)
        for o_start in range(0, self.out_features, chunk):
            o_end = min(o_start + chunk, self.out_features)
            vth_p = self.vth_pos[o_start:o_end]
            vth_n = self.vth_neg[o_start:o_end]
            vgs = x_exp.expand(-1, -1, vth_p.shape[0])
            vth_p_b = vth_p.t().unsqueeze(0)
            vth_n_b = vth_n.t().unsqueeze(0)
            i_diff = ekv_diff_current_fn(vgs, vth_p_b, vth_n_b, self.cfg)
            outputs.append(i_diff.sum(dim=1))
        return torch.cat(outputs, dim=-1)

    def clamp_vth(self) -> None:
        v_min = self.cfg["V_TH_MIN"]
        v_max = self.cfg["V_TH_MAX"]
        with torch.no_grad():
            self.vth_pos.clamp_(v_min, v_max)
            self.vth_neg.clamp_(v_min, v_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, in_f = x.shape
        assert in_f == self.in_features

        r_tia = self.r_tia.to(device=x.device, dtype=torch.float32).view(1, 1, 1)
        v_signed_min = float(self.cfg.get("V_signed_min", -4.0))
        v_signed_max = float(self.cfg.get("V_signed_max", 4.0))

        if bool(self.cfg.get("use_triton", False)) and x.is_cuda:
            flat = x.reshape(b * n, in_f).contiguous()
            approx = str(self.cfg.get("ekv_approx", "lut")).lower()
            if approx == "lut" and triton_lut_ekv_matmul is not None:
                idiff = triton_lut_ekv_matmul(flat, self.vth_pos, self.vth_neg, self.cfg)
            elif triton_ekv_matmul is not None:
                idiff = triton_ekv_matmul(flat, self.vth_pos, self.vth_neg, self.cfg)
            else:
                idiff = None
            if idiff is not None:
                # kernel returns I_diff, already clamped to [v_signed_min, v_signed_max]
                # multiply by learnable R_TIA here for gradient tracking
                out = idiff.reshape(b, n, self.out_features) * r_tia
                out = out.clamp(v_signed_min, v_signed_max)
                if self.track_output_stats:
                    with torch.no_grad():
                        detached = out.detach()
                        self._stat_total.add_(detached.numel())
                        self._stat_abs_max.copy_(torch.maximum(self._stat_abs_max, detached.abs().max().float()))
                return out

        chunk = int(self.cfg.get("FORWARD_CHUNK_SIZE", 64))
        outputs = []
        x_exp = x.unsqueeze(-1)
        for o_start in range(0, self.out_features, chunk):
            o_end = min(o_start + chunk, self.out_features)
            vth_p = self.vth_pos[o_start:o_end]
            vth_n = self.vth_neg[o_start:o_end]
            vgs = x_exp.expand(-1, -1, -1, vth_p.shape[0])
            vth_p_b = vth_p.t().unsqueeze(0).unsqueeze(0)
            vth_n_b = vth_n.t().unsqueeze(0).unsqueeze(0)
            i_diff = ekv_diff_current_fn(vgs, vth_p_b, vth_n_b, self.cfg)
            outputs.append(i_diff.sum(dim=2) * r_tia)
        out = torch.cat(outputs, dim=-1)
        return self._clamp_output(out, v_signed_min, v_signed_max)


class PhysicalFFN(nn.Module):
    """EKV physical FFN: external voltage mapping -> CIMLinear -> CIMLinear.

    There is no explicit GELU/ReLU and no remapping between the two CIM layers.
    The first CIM layer's signed TIA output directly drives the second layer;
    negative voltages are naturally filtered by the next layer's non-negative Vth.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        physical_config: Optional[Dict[str, float]] = None,
        voltage_max: float = 4.0,
    ):
        super().__init__()
        self.input_mapping = VoltageMapping(voltage_max=voltage_max)
        self.fc1 = CIMLinear(dim, hidden_dim, physical_config)
        self.fc2 = CIMLinear(hidden_dim, dim, physical_config)
        self.adc_affine_scale = nn.Parameter(torch.ones(dim))
        self.adc_affine_shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_mapping(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x * self.adc_affine_scale + self.adc_affine_shift

    def clamp_vth(self) -> None:
        self.fc1.clamp_vth()
        self.fc2.clamp_vth()


class HybridFFN(nn.Module):
    """Hybrid FFN: CIM first layer + standard Linear second layer.

    fc1 = CIMLinear with TIA (same as PhysicalFFN)
    fc2 = nn.Linear (standard weight matrix, no EKV, no GELU)

    The first CIM layer's signed TIA output directly drives the second
    standard linear layer. TIA gain is preserved on fc1.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        physical_config: Optional[Dict[str, float]] = None,
        voltage_max: float = 4.0,
    ):
        super().__init__()
        self.input_mapping = VoltageMapping(voltage_max=voltage_max)
        self.fc1 = CIMLinear(dim, hidden_dim, physical_config)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_mapping(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def clamp_vth(self) -> None:
        self.fc1.clamp_vth()
        # fc2 is nn.Linear, no Vth to clamp
