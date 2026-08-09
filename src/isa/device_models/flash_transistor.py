"""EKV differential current autograd function.

This module follows the signed-voltage EKV formulation used by the
EKV-Vth-TIA design note:

  I_basic = I_S * [softplus((V - Vth)/(2 n U_T))^2
                  - softplus((V - Vth - V_D)/(2 n U_T))^2]
  I       = I_basic / (1 + V / V_sat)
  I_diff  = I(V, Vth_pos) - I(V, Vth_neg)

The TIA gain and signed output clamp are applied by CIMLinear.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _softplus_sigmoid_from_sp(sp_x: torch.Tensor) -> torch.Tensor:
    """Recover sigmoid(x) from softplus(x), stable for large positive x."""
    return torch.where(sp_x > 20.0, torch.ones_like(sp_x), 1.0 - torch.exp(-sp_x))


def ekv_current(vgs: torch.Tensor, vth: torch.Tensor, cfg: Dict[str, float]) -> torch.Tensor:
    """Compute one EKV branch current I(V_gs, V_th)."""
    n = cfg["n"]
    ut = cfg["U_T"]
    inv_2nut = 1.0 / (2.0 * n * ut)
    vd = cfg["V_D"]
    i_s = cfg["I_S"]
    v_sat = cfg["V_sat"]

    u1 = (vgs - vth) * inv_2nut
    u2 = (vgs - vth - vd) * inv_2nut
    sp1 = F.softplus(u1)
    sp2 = F.softplus(u2)
    i_basic = sp1 * sp1 - sp2 * sp2
    return i_s * i_basic / (1.0 + vgs / v_sat)


def ekv_current_grad_v(vgs: torch.Tensor, vth: torch.Tensor, cfg: Dict[str, float]) -> torch.Tensor:
    """Analytic dI/dV for one EKV branch."""
    n = cfg["n"]
    ut = cfg["U_T"]
    inv_2nut = 1.0 / (2.0 * n * ut)
    vd = cfg["V_D"]
    i_s = cfg["I_S"]
    v_sat = cfg["V_sat"]

    u1 = (vgs - vth) * inv_2nut
    u2 = (vgs - vth - vd) * inv_2nut
    sp1 = F.softplus(u1)
    sp2 = F.softplus(u2)
    sig1 = _softplus_sigmoid_from_sp(sp1)
    sig2 = _softplus_sigmoid_from_sp(sp2)

    di_dv_num = 2.0 * sp1 * sig1 * inv_2nut - 2.0 * sp2 * sig2 * inv_2nut
    i_basic = sp1 * sp1 - sp2 * sp2
    denom = 1.0 + vgs / v_sat
    return i_s * (di_dv_num / denom - i_basic / (v_sat * denom * denom))


def ekv_current_grad_vth(vgs: torch.Tensor, vth: torch.Tensor, cfg: Dict[str, float]) -> torch.Tensor:
    """Analytic dI/dVth for one EKV branch."""
    n = cfg["n"]
    ut = cfg["U_T"]
    inv_2nut = 1.0 / (2.0 * n * ut)
    vd = cfg["V_D"]
    i_s = cfg["I_S"]

    u1 = (vgs - vth) * inv_2nut
    u2 = (vgs - vth - vd) * inv_2nut
    sp1 = F.softplus(u1)
    sp2 = F.softplus(u2)
    sig1 = _softplus_sigmoid_from_sp(sp1)
    sig2 = _softplus_sigmoid_from_sp(sp2)

    di = -2.0 * sp1 * sig1 * inv_2nut + 2.0 * sp2 * sig2 * inv_2nut
    return i_s * di / (1.0 + vgs / cfg["V_sat"])


def ekv_diff_current(
    vgs: torch.Tensor,
    vth_pos: torch.Tensor,
    vth_neg: torch.Tensor,
    cfg: Dict[str, float],
) -> torch.Tensor:
    """Differential current I(V_gs,Vth+) - I(V_gs,Vth-), before TIA."""
    return ekv_current(vgs, vth_pos, cfg) - ekv_current(vgs, vth_neg, cfg)


class EKVDiffCurrentFn(torch.autograd.Function):
    """Custom autograd for EKV differential current with analytic gradients."""

    @staticmethod
    def forward(
        ctx,
        vgs: torch.Tensor,
        vth_pos: torch.Tensor,
        vth_neg: torch.Tensor,
        config: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        cfg = dict(config or {})
        out = ekv_diff_current(vgs.float(), vth_pos.float(), vth_neg.float(), cfg)
        ctx.save_for_backward(vgs, vth_pos, vth_neg)
        ctx.config = cfg
        return out.to(dtype=vgs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        vgs, vth_pos, vth_neg = ctx.saved_tensors
        cfg = ctx.config
        grad = grad_output.float()
        v = vgs.float()
        vp = vth_pos.float()
        vn = vth_neg.float()

        grad_vgs = grad_vth_pos = grad_vth_neg = None
        if ctx.needs_input_grad[0]:
            dpos = ekv_current_grad_v(v, vp, cfg)
            dneg = ekv_current_grad_v(v, vn, cfg)
            grad_vgs = (grad * (dpos - dneg)).to(dtype=vgs.dtype)
        if ctx.needs_input_grad[1]:
            grad_vth_pos = (grad * ekv_current_grad_vth(v, vp, cfg)).to(dtype=vth_pos.dtype)
        if ctx.needs_input_grad[2]:
            grad_vth_neg = (grad * (-ekv_current_grad_vth(v, vn, cfg))).to(dtype=vth_neg.dtype)

        return grad_vgs, grad_vth_pos, grad_vth_neg, None


def ekv_diff_current_fn(
    vgs: torch.Tensor,
    vth_pos: torch.Tensor,
    vth_neg: torch.Tensor,
    config: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    return EKVDiffCurrentFn.apply(vgs, vth_pos, vth_neg, config)
