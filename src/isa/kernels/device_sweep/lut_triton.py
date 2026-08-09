"""
LUT/planar accelerated NVM device kernels.

The exact device kernels are still available in the per-device modules.  This
file provides a faster approximation path for training:

* Flash/FeFET: 1D LUT over x = V - Vth with a derivative table.
* PCM: 2D bilinear LUT over (V, w), also caching dI/dV and dI/dw.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 32
_BLOCK_K = 16


@triton.jit
def _interp1(table, x, x_min: tl.constexpr, inv_dx: tl.constexpr, size: tl.constexpr):
    xf = (x - x_min) * inv_dx
    xf = tl.minimum(tl.maximum(xf, 0.0), size - 1.001)
    i0 = xf.to(tl.int32)
    t = xf - i0.to(tl.float32)
    y0 = tl.load(table + i0)
    y1 = tl.load(table + i0 + 1)
    return y0 + t * (y1 - y0)


@triton.jit
def _lut1_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr, Y_ptr,
    M, N, K,
    stride_vm, stride_vk, stride_wn, stride_wk, stride_om, stride_on,
    x_min: tl.constexpr, inv_dx: tl.constexpr, lut_size: tl.constexpr,
    I_S, V_sat,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(BLOCK_K):
        k = k_start + kk
        m_mask = offs_m < M
        n_mask = offs_n < N
        k_mask = k < K
        v = tl.load(V_ptr + offs_m * stride_vm + k * stride_vk, mask=m_mask & k_mask, other=0.0).to(tl.float32)
        wp = tl.load(Wp_ptr + offs_n * stride_wn + k * stride_wk, mask=n_mask & k_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + offs_n * stride_wn + k * stride_wk, mask=n_mask & k_mask, other=0.0).to(tl.float32)
        x_p = v[:, None] - wp[None, :]
        x_n = v[:, None] - wn[None, :]
        y = _interp1(Y_ptr, x_p, x_min, inv_dx, lut_size) - _interp1(Y_ptr, x_n, x_min, inv_dx, lut_size)
        if MODE == 1 or MODE == 2:
            scale = I_S / (1.0 + v / V_sat)
            y = y * scale[:, None]
        acc += y

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=mask)


@triton.jit
def _lut1_grad_v_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr, Y_ptr, DY_ptr,
    M, N, K,
    stride_gom, stride_gon, stride_vm, stride_vk, stride_wn, stride_wk, stride_dvm, stride_dvk,
    x_min: tl.constexpr, inv_dx: tl.constexpr, lut_size: tl.constexpr,
    I_S, V_sat,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K
    mk_mask = m_mask & k_mask
    v = tl.load(V_ptr + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk, mask=mk_mask, other=0.0).to(tl.float32)
    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    n_start = pid_n * BLOCK_N
    for nn in range(BLOCK_N):
        n = n_start + nn
        n_valid = n < N
        go = tl.load(GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon, mask=m_mask & n_valid, other=0.0).to(tl.float32)
        wp = tl.load(Wp_ptr + n * stride_wn + offs_k[None, :] * stride_wk, mask=k_mask & n_valid, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + n * stride_wn + offs_k[None, :] * stride_wk, mask=k_mask & n_valid, other=0.0).to(tl.float32)
        x_p = v - wp
        x_n = v - wn
        dyp = _interp1(DY_ptr, x_p, x_min, inv_dx, lut_size)
        dyn = _interp1(DY_ptr, x_n, x_min, inv_dx, lut_size)
        local = dyp - dyn
        if MODE == 1:
            yp = _interp1(Y_ptr, x_p, x_min, inv_dx, lut_size)
            yn = _interp1(Y_ptr, x_n, x_min, inv_dx, lut_size)
            denom = 1.0 + v / V_sat
            inv_den = 1.0 / denom
            local = I_S * ((dyp - dyn) * inv_den - (yp - yn) * inv_den * inv_den / V_sat)
        dv += go * local

    tl.atomic_add(dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk, dv, mask=mk_mask)


@triton.jit
def _lut1_grad_w_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr, DY_ptr,
    M, N, K,
    stride_gom, stride_gon, stride_vm, stride_vk, stride_wn, stride_wk, stride_dwn, stride_dwk,
    x_min: tl.constexpr, inv_dx: tl.constexpr, lut_size: tl.constexpr,
    I_S, V_sat,
    MODE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < K
    nk_mask = n_mask & k_mask
    wp = tl.load(Wp_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=nk_mask, other=0.0).to(tl.float32)
    wn = tl.load(Wn_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=nk_mask, other=0.0).to(tl.float32)
    dwp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwn = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    m_start = pid_m * BLOCK_M
    for mm in range(BLOCK_M):
        m = m_start + mm
        m_valid = m < M
        go = tl.load(GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon, mask=n_mask & m_valid, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + m * stride_vm + offs_k[None, :] * stride_vk, mask=k_mask & m_valid, other=0.0).to(tl.float32)
        scale = 1.0
        if MODE == 1 or MODE == 2:
            scale = I_S / (1.0 + v / V_sat)
        dyp = _interp1(DY_ptr, v - wp, x_min, inv_dx, lut_size) * scale
        dyn = _interp1(DY_ptr, v - wn, x_min, inv_dx, lut_size) * scale
        dwp += go * (-dyp)
        dwn += go * dyn

    tl.atomic_add(dWp_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk, dwp, mask=nk_mask)
    tl.atomic_add(dWn_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk, dwn, mask=nk_mask)


@triton.jit
def _interp2(table, v, w, v_min: tl.constexpr, inv_dv: tl.constexpr, nv: tl.constexpr,
             w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr):
    vf = (v - v_min) * inv_dv
    wf = (w - w_min) * inv_dw
    vf = tl.minimum(tl.maximum(vf, 0.0), nv - 1.001)
    wf = tl.minimum(tl.maximum(wf, 0.0), nw - 1.001)
    i = vf.to(tl.int32)
    j = wf.to(tl.int32)
    tv = vf - i.to(tl.float32)
    tw = wf - j.to(tl.float32)
    base = i * nw + j
    y00 = tl.load(table + base)
    y01 = tl.load(table + base + 1)
    y10 = tl.load(table + base + nw)
    y11 = tl.load(table + base + nw + 1)
    return (1.0 - tv) * ((1.0 - tw) * y00 + tw * y01) + tv * ((1.0 - tw) * y10 + tw * y11)


@triton.jit
def _lut2_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr, Y_ptr,
    M, N, K,
    stride_vm, stride_vk, stride_wn, stride_wk, stride_om, stride_on,
    v_min: tl.constexpr, inv_dv: tl.constexpr, nv: tl.constexpr,
    w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        m_mask = offs_m < M
        n_mask = offs_n < N
        v = tl.load(V_ptr + offs_m * stride_vm + k * stride_vk, mask=m_mask, other=0.0).to(tl.float32)
        wp = tl.load(Wp_ptr + offs_n * stride_wn + k * stride_wk, mask=n_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + offs_n * stride_wn + k * stride_wk, mask=n_mask, other=0.0).to(tl.float32)
        acc += _interp2(Y_ptr, v[:, None], wp[None, :], v_min, inv_dv, nv, w_min, inv_dw, nw)
        acc -= _interp2(Y_ptr, v[:, None], wn[None, :], v_min, inv_dv, nv, w_min, inv_dw, nw)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=mask)


@triton.jit
def _lut2_grad_v_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr, DV_ptr,
    M, N, K,
    stride_gom, stride_gon, stride_vm, stride_vk, stride_wn, stride_wk, stride_dvm, stride_dvk,
    v_min: tl.constexpr, inv_dv: tl.constexpr, nv: tl.constexpr,
    w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K
    mk_mask = m_mask & k_mask
    v = tl.load(V_ptr + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk, mask=mk_mask, other=0.0).to(tl.float32)
    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(N):
        go = tl.load(GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon, mask=m_mask, other=0.0).to(tl.float32)
        wp = tl.load(Wp_ptr + n * stride_wn + offs_k[None, :] * stride_wk, mask=k_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + n * stride_wn + offs_k[None, :] * stride_wk, mask=k_mask, other=0.0).to(tl.float32)
        dv += go * (
            _interp2(DV_ptr, v, wp, v_min, inv_dv, nv, w_min, inv_dw, nw)
            - _interp2(DV_ptr, v, wn, v_min, inv_dv, nv, w_min, inv_dw, nw)
        )
    tl.store(dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk, dv, mask=mk_mask)


@triton.jit
def _lut2_grad_w_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr, DW_ptr,
    M, N, K,
    stride_gom, stride_gon, stride_vm, stride_vk, stride_wn, stride_wk, stride_dwn, stride_dwk,
    v_min: tl.constexpr, inv_dv: tl.constexpr, nv: tl.constexpr,
    w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < K
    nk_mask = n_mask & k_mask
    wp = tl.load(Wp_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=nk_mask, other=0.0).to(tl.float32)
    wn = tl.load(Wn_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=nk_mask, other=0.0).to(tl.float32)
    dwp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwn = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m in range(M):
        go = tl.load(GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon, mask=n_mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + m * stride_vm + offs_k[None, :] * stride_vk, mask=k_mask, other=0.0).to(tl.float32)
        dwp += go * _interp2(DW_ptr, v, wp, v_min, inv_dv, nv, w_min, inv_dw, nw)
        dwn += go * (-_interp2(DW_ptr, v, wn, v_min, inv_dv, nv, w_min, inv_dw, nw))
    tl.store(dWp_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk, dwp, mask=nk_mask)
    tl.store(dWn_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk, dwn, mask=nk_mask)


def _cache_key(params, device, dtype, names):
    return tuple(float(params[n]) for n in names) + (str(device), str(dtype))


_flash_cache = {}
_fefet_cache = {}
_pcm_cache = {}


def _lut_size(params, default):
    return int(params.get("lut_size", default))


def _make_flash_tables(params, device, dtype):
    size = _lut_size(params, 4096)
    x_min = float(params.get("lut_x_min", -5.0))
    x_max = float(params.get("lut_x_max", 4.0))
    key = _cache_key(params, device, torch.float32, ("I_S", "n", "U_T", "V_D", "V_sat")) + (size, x_min, x_max)
    cached = _flash_cache.get(key)
    if cached is not None:
        return cached
    x = torch.linspace(x_min, x_max, size, device=device, dtype=torch.float32)
    inv_2nUT = 1.0 / (2.0 * float(params["n"]) * float(params["U_T"]))
    vd = float(params["V_D"]) * inv_2nUT
    u = x * inv_2nUT
    sp1 = torch.nn.functional.softplus(u)
    sp2 = torch.nn.functional.softplus(u - vd)
    y = sp1.square() - sp2.square()
    dy = 2.0 * (sp1 * torch.sigmoid(u) - sp2 * torch.sigmoid(u - vd)) * inv_2nUT
    meta = (x_min, (size - 1) / (x_max - x_min), size, float(params["I_S"]), float(params["V_sat"]))
    _flash_cache[key] = (y.contiguous(), dy.contiguous(), meta)
    return _flash_cache[key]


def _fefet_lk_table(y_ext, params, a_lk, b_lk):
    A = float(a_lk)
    B = float(b_lk)
    x = y_ext / max(B, 1e-8)
    for _ in range(8):
        fp = 3.0 * A * x.square() + B
        x = x - (A * x.pow(3) + B * x - y_ext) / fp.clamp_min(1e-8)
    inv_fp = (3.0 * A * x.square() + B).clamp_min(1e-8).reciprocal()
    inv_2nUT = 1.0 / (2.0 * float(params["n"]) * float(params["U_T"]))
    vd = float(params["V_D"])
    u1 = x * inv_2nUT
    u2 = (x - vd) * inv_2nUT
    sp1 = torch.nn.functional.softplus(u1)
    sp2 = torch.nn.functional.softplus(u2)
    y = sp1.square() - sp2.square()
    dy = 2.0 * (sp1 * torch.sigmoid(u1) - sp2 * torch.sigmoid(u2)) * inv_2nUT * inv_fp
    return y, dy


def _make_fefet_tables(params, device, dtype):
    size = _lut_size(params, 4096)
    x_min = float(params.get("lut_x_min", -5.0))
    x_max = float(params.get("lut_x_max", 4.0))
    surrogate_enabled = bool(params.get("surrogate_backward_enabled", False))
    surrogate_alpha = float(params.get("surrogate_backward_alpha", 1.0 if surrogate_enabled else 0.0))
    surrogate_alpha = min(max(surrogate_alpha, 0.0), 1.0)
    surrogate_A = float(params.get("surrogate_A_lk", params["A_lk"]))
    surrogate_B = float(params.get("surrogate_B_lk", params["B_lk"]))
    key = (
        _cache_key(params, device, torch.float32, ("I_S", "n", "U_T", "V_D", "A_lk", "B_lk", "V_sat"))
        + (size, x_min, x_max, surrogate_enabled, surrogate_alpha, surrogate_A, surrogate_B)
    )
    cached = _fefet_cache.get(key)
    if cached is not None:
        return cached
    y_ext = torch.linspace(x_min, x_max, size, device=device, dtype=torch.float32)
    y, dy_exact = _fefet_lk_table(y_ext, params, params["A_lk"], params["B_lk"])
    dy = dy_exact
    if surrogate_enabled and surrogate_alpha > 0.0:
        _, dy_proxy = _fefet_lk_table(y_ext, params, surrogate_A, surrogate_B)
        dy = torch.lerp(dy_exact, dy_proxy, surrogate_alpha)
    meta = (x_min, (size - 1) / (x_max - x_min), size, float(params["I_S"]), float(params["V_sat"]))
    _fefet_cache[key] = (y.contiguous(), dy.contiguous(), meta)
    return _fefet_cache[key]

def _make_pcm_tables(params, device, dtype):
    nv = int(params.get("lut_v_size", params.get("lut_size", 512)))
    nw = int(params.get("lut_w_size", params.get("lut_size", 512)))
    v_min = float(params.get("lut_v_min", -1.0))
    v_max = float(params.get("lut_v_max", 1.0))
    w_min = float(params.get("w_min", 0.0))
    w_max = float(params.get("w_max", 1.0))
    key = _cache_key(params, device, torch.float32, ("I0_pcm", "I0_pcm_decay", "V_T_pcm")) + (nv, nw, v_min, v_max, w_min, w_max)
    cached = _pcm_cache.get(key)
    if cached is not None:
        return cached
    v = torch.linspace(v_min, v_max, nv, device=device, dtype=torch.float32)[:, None]
    w = torch.linspace(w_min, w_max, nw, device=device, dtype=torch.float32)[None, :]
    i0 = float(params["I0_pcm"])
    i0_decay = float(params.get("I0_pcm_decay", 0.0))
    vt = float(params["V_T_pcm"])
    chi = w
    i0_chi = i0 * torch.exp(-i0_decay * chi)
    arg = (chi * v / vt).clamp(-20.0, 20.0)
    y = i0_chi * chi * torch.sinh(arg)
    dv = i0_chi * chi.square() / vt * torch.cosh(arg)
    dw = i0_chi * ((1.0 - i0_decay * chi) * torch.sinh(arg) + chi * (v / vt) * torch.cosh(arg))
    meta = (v_min, (nv - 1) / (v_max - v_min), nv, w_min, (nw - 1) / (w_max - w_min), nw)
    _pcm_cache[key] = (y.contiguous(), dv.contiguous(), dw.contiguous(), meta)
    return _pcm_cache[key]


class _LUT1Function(torch.autograd.Function):
    mode = 0
    table_fn = None

    @classmethod
    def forward(cls, ctx, v_in, w_pos, w_neg, params):
        v_in = v_in.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        y, dy, meta = cls.table_fn(params, v_in.device, v_in.dtype)
        M, K = v_in.shape
        N = w_pos.shape[0]
        # K-split partials use atomic_add, which is unsupported for BF16 on
        # this Triton/CUDA stack.  FP32 also gives deterministic accumulation
        # precision before TIA scaling.
        out = torch.zeros(M, N, device=v_in.device, dtype=torch.float32)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
        _lut1_fwd_kernel[grid](
            v_in, w_pos, w_neg, out, y, M, N, K,
            v_in.stride(0), v_in.stride(1), w_pos.stride(0), w_pos.stride(1), out.stride(0), out.stride(1),
            meta[0], meta[1], meta[2], meta[3], meta[4],
            MODE=cls.mode, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )
        ctx.save_for_backward(v_in, w_pos, w_neg, y, dy)
        ctx.meta = meta
        ctx.mode = cls.mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn, y, dy = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N = wp.shape[0]
        meta = ctx.meta
        dV = None
        if ctx.needs_input_grad[0]:
            dV_acc = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            grid_v = (triton.cdiv(M, _BLOCK_M), triton.cdiv(K, _BLOCK_K), triton.cdiv(N, _BLOCK_N))
            _lut1_grad_v_kernel[grid_v](
                grad_out, v, wp, wn, dV_acc, y, dy, M, N, K,
                grad_out.stride(0), grad_out.stride(1), v.stride(0), v.stride(1), wp.stride(0), wp.stride(1), dV_acc.stride(0), dV_acc.stride(1),
                meta[0], meta[1], meta[2], meta[3], meta[4],
                MODE=ctx.mode, BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K, BLOCK_N=_BLOCK_N,
            )
            dV = dV_acc.to(v.dtype)
        dWp = torch.zeros_like(wp)
        dWn = torch.zeros_like(wn)
        grid_w = (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K), triton.cdiv(M, _BLOCK_M))
        _lut1_grad_w_kernel[grid_w](
            grad_out, v, wp, wn, dWp, dWn, dy, M, N, K,
            grad_out.stride(0), grad_out.stride(1), v.stride(0), v.stride(1), wp.stride(0), wp.stride(1), dWp.stride(0), dWp.stride(1),
            meta[0], meta[1], meta[2], meta[3], meta[4],
            MODE=ctx.mode, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, BLOCK_M=_BLOCK_M,
        )
        return dV, dWp, dWn, None


class FlashLUTFunction(_LUT1Function):
    mode = 1
    table_fn = staticmethod(_make_flash_tables)


class FeFETLUTFunction(_LUT1Function):
    mode = 1
    table_fn = staticmethod(_make_fefet_tables)


class PCMLUTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        v_in = v_in.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        y, dv, dw, meta = _make_pcm_tables(params, v_in.device, v_in.dtype)
        M, K = v_in.shape
        N = w_pos.shape[0]
        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _lut2_fwd_kernel[grid](
            v_in, w_pos, w_neg, out, y, M, N, K,
            v_in.stride(0), v_in.stride(1), w_pos.stride(0), w_pos.stride(1), out.stride(0), out.stride(1),
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )
        ctx.save_for_backward(v_in, w_pos, w_neg, dv, dw)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn, dv_table, dw_table = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N = wp.shape[0]
        meta = ctx.meta
        dV = torch.empty_like(v)
        grid_v = (triton.cdiv(M, _BLOCK_M), triton.cdiv(K, _BLOCK_K))
        _lut2_grad_v_kernel[grid_v](
            grad_out, v, wp, wn, dV, dv_table, M, N, K,
            grad_out.stride(0), grad_out.stride(1), v.stride(0), v.stride(1), wp.stride(0), wp.stride(1), dV.stride(0), dV.stride(1),
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
            BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K,
        )
        dWp = torch.empty_like(wp)
        dWn = torch.empty_like(wn)
        grid_w = (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
        _lut2_grad_w_kernel[grid_w](
            grad_out, v, wp, wn, dWp, dWn, dw_table, M, N, K,
            grad_out.stride(0), grad_out.stride(1), v.stride(0), v.stride(1), wp.stride(0), wp.stride(1), dWp.stride(0), dWp.stride(1),
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
            BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )
        return dV, dWp, dWn, None


class PCMPlanarFunction(torch.autograd.Function):
    """Fast PCM path using a separable odd-polynomial expansion.

    I = I0(w) * w * sinh(w * V / V_T)
      ~= I0_prefactor * exp(-I0_decay*w)
          * sum_r w^(2r+2) * V^(2r+1)
              / (V_T^(2r+1) * (2r+1)!)

    The default r=0..3 truncation is very accurate over the configured
    training range (|V| <= 1, 0 <= w <= 1) and turns the layer into a small
    sum of dense GEMMs.
    """

    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        v = v_in.contiguous()
        wp = w_pos.contiguous()
        wn = w_neg.contiguous()
        i0 = float(params["I0_pcm"])
        i0_decay = float(params.get("I0_pcm_decay", 0.0))
        vt = float(params["V_T_pcm"])
        order = int(params.get("planar_order", 7))
        max_r = max(0, (order - 1) // 2)

        chi_p = wp
        chi_n = wn
        i0_p = torch.exp(-i0_decay * chi_p)
        i0_n = torch.exp(-i0_decay * chi_n)
        out = torch.zeros(v.shape[0], wp.shape[0], device=v.device, dtype=v.dtype)
        v_terms = []
        coeff_diffs = []
        coeff_pos = []
        coeff_neg = []
        coeff_scales = []

        fact = 1.0
        for r in range(max_r + 1):
            power = 2 * r + 1
            if r == 0:
                fact = 1.0
            else:
                fact *= (power - 1) * power
            scale = i0 / ((vt ** power) * fact)
            v_term = v ** power
            cp = scale * i0_p * (chi_p ** (power + 1))
            cn = scale * i0_n * (chi_n ** (power + 1))
            cd = cp - cn
            out = out + v_term.matmul(cd.t())
            v_terms.append(v_term)
            coeff_diffs.append(cd)
            coeff_pos.append(cp)
            coeff_neg.append(cn)
            coeff_scales.append((power, scale))

        ctx.save_for_backward(v, wp, wn, chi_p, chi_n, *v_terms, *coeff_diffs, *coeff_pos, *coeff_neg)
        ctx.meta = (len(v_terms), coeff_scales, i0_decay)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        v, wp, wn, chi_p, chi_n = saved[:5]
        n_terms, coeff_scales, i0_decay = ctx.meta
        off = 5
        v_terms = saved[off:off + n_terms]
        off += n_terms
        coeff_diffs = saved[off:off + n_terms]
        off += n_terms
        coeff_pos = saved[off:off + n_terms]
        off += n_terms
        coeff_neg = saved[off:off + n_terms]

        go = grad_out.contiguous()
        dV = torch.zeros_like(v)
        dWp = torch.zeros_like(wp)
        dWn = torch.zeros_like(wn)
        for idx, (power, scale) in enumerate(coeff_scales):
            cd = coeff_diffs[idx]
            vt = v_terms[idx]
            if power == 1:
                d_vterm = torch.ones_like(v)
            else:
                d_vterm = power * (v ** (power - 1))
            dV = dV + go.matmul(cd) * d_vterm

            common = go.t().matmul(vt)
            # d/dw [scale * exp(-decay*w) * w^(power+1)], with chi=w.
            dcp_dw = scale * torch.exp(-i0_decay * chi_p) * (
                (power + 1) * (chi_p ** power) - i0_decay * (chi_p ** (power + 1))
            )
            dcn_dw = scale * torch.exp(-i0_decay * chi_n) * (
                (power + 1) * (chi_n ** power) - i0_decay * (chi_n ** (power + 1))
            )
            dWp = dWp + common * dcp_dw
            dWn = dWn - common * dcn_dw

        return dV, dWp, dWn, None
