"""Polynomial evaluation of the Flash EKV response used by device checkpoints.

The scalar curve is fitted over the recorded voltage interval. Expanding its
powers evaluates the differential array as matrix products. The fit degree,
interval and matrix precision are part of the checkpoint configuration.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_COEFFICIENTS = {}


def flash_curve(x, params):
    inv = 1.0 / (2.0 * float(params["n"]) * float(params["U_T"]))
    u = x * inv
    return F.softplus(u).square() - F.softplus(u - float(params["V_D"]) * inv).square()


def polynomial_coefficients(params, device):
    keys = ("n", "U_T", "V_D", "poly_degree", "poly_fit_points", "poly_x_min", "poly_x_max")
    key = (str(device),) + tuple((name, float(params[name])) for name in keys if name in params)
    if key in _COEFFICIENTS:
        return _COEFFICIENTS[key]
    degree = int(params.get("poly_degree", 7))
    if degree < 1:
        raise ValueError("poly_degree must be positive")
    points = int(params.get("poly_fit_points", max(4096, 512 * (degree + 1))))
    lo = float(params.get("poly_x_min", params.get("lut_x_min", -5.0)))
    hi = float(params.get("poly_x_max", params.get("lut_x_max", 4.0)))
    if lo >= hi or points <= degree:
        raise ValueError("Invalid polynomial fitting interval or sample count")
    x = torch.linspace(lo, hi, points, device=device, dtype=torch.float64)
    y = flash_curve(x, params)
    center, scale = 0.5 * (lo + hi), 0.5 * (hi - lo)
    z = (x - center) / scale
    columns = [torch.ones_like(z)]
    for _ in range(degree):
        columns.append(columns[-1] * z)
    matrix = torch.stack(columns, dim=1)
    coefficients = torch.linalg.lstsq(matrix, y[:, None]).solution[:, 0].to(torch.float32)
    packed = torch.cat([torch.tensor([center, scale], device=device), coefficients])
    _COEFFICIENTS[key] = packed
    return packed


class FlashPolynomialFunction:
    @staticmethod
    def apply(v_in, w_pos, w_neg, params):
        v, wp, wn = v_in.contiguous(), w_pos.contiguous(), w_neg.contiguous()
        packed = polynomial_coefficients(params, v.device)
        center, scale, coefficients = packed[0], packed[1].clamp_min(1e-8), packed[2:]
        degree = coefficients.numel() - 1
        use_bf16 = bool(params.get("poly_bf16", True))
        va, wpa, wna = v / scale, (-wp - center) / scale, (-wn - center) / scale
        vp, wpp, wnp = [torch.ones_like(va), va], [torch.ones_like(wpa), wpa], [torch.ones_like(wna), wna]
        for _ in range(2, degree + 1):
            vp.append(vp[-1] * va)
            wpp.append(wpp[-1] * wpa)
            wnp.append(wnp[-1] * wna)
        out = torch.zeros(v.shape[0], wp.shape[0], device=v.device, dtype=torch.float32)
        denominator = (1.0 + v / float(params["V_sat"])).reciprocal()
        for power in range(1, degree + 1):
            coefficient = coefficients[power]
            if float(coefficient.abs()) == 0.0:
                continue
            for index in range(power + 1):
                a = vp[index] * denominator
                b = wpp[power - index] - wnp[power - index]
                product = a.to(torch.bfloat16).matmul(b.to(torch.bfloat16).t()).float() if use_bf16 else a.matmul(b.t())
                out = out + (coefficient * math.comb(power, index)) * product
        return (out * float(params["I_S"])).to(v_in.dtype)
