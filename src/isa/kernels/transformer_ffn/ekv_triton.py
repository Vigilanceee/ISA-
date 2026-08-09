"""Exact fused Triton implementation for EKV differential current matmul."""

from __future__ import annotations

from typing import Any, Dict

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import-time fallback
    triton = None
    tl = None


def _ceil_div(a: int, b: int) -> int:
    return triton.cdiv(a, b)


if triton is not None:

    @triton.jit
    def _softplus(x):
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


    @triton.jit
    def _sigmoid(x):
        return 1.0 / (1.0 + tl.exp(-x))


    @triton.jit
    def _ekv_current(v, vth, inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr):
        u1 = (v - vth) * inv_2nut
        u2 = (v - vth - v_d) * inv_2nut
        sp1 = _softplus(u1)
        sp2 = _softplus(u2)
        i_basic = sp1 * sp1 - sp2 * sp2
        return i_s * i_basic / (1.0 + v / v_sat)


    @triton.jit
    def _ekv_grad_v(v, vth, inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr):
        u1 = (v - vth) * inv_2nut
        u2 = (v - vth - v_d) * inv_2nut
        sp1 = _softplus(u1)
        sp2 = _softplus(u2)
        sig1 = _sigmoid(u1)
        sig2 = _sigmoid(u2)
        di_dv_num = 2.0 * sp1 * sig1 * inv_2nut - 2.0 * sp2 * sig2 * inv_2nut
        denom = 1.0 + v / v_sat
        i_basic = sp1 * sp1 - sp2 * sp2
        return i_s * (di_dv_num / denom - i_basic / (v_sat * denom * denom))


    @triton.jit
    def _ekv_grad_vth(v, vth, inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr):
        u1 = (v - vth) * inv_2nut
        u2 = (v - vth - v_d) * inv_2nut
        sp1 = _softplus(u1)
        sp2 = _softplus(u2)
        sig1 = _sigmoid(u1)
        sig2 = _sigmoid(u2)
        di = -2.0 * sp1 * sig1 * inv_2nut + 2.0 * sp2 * sig2 * inv_2nut
        return i_s * di / (1.0 + v / v_sat)


    @triton.jit
    def _forward_kernel(
        x, wpos, wneg, out,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_o = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_O), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            xv = tl.load(x + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
            wp = tl.load(wpos + offs_o[:, None] * K + k[None, :], mask=(offs_o[:, None] < O) & (k[None, :] < K), other=0.0).to(tl.float32)
            wn = tl.load(wneg + offs_o[:, None] * K + k[None, :], mask=(offs_o[:, None] < O) & (k[None, :] < K), other=0.0).to(tl.float32)
            ip = _ekv_current(xv[:, None, :], wp[None, :, :], inv_2nut, v_d, i_s, v_sat)
            inn = _ekv_current(xv[:, None, :], wn[None, :, :], inv_2nut, v_d, i_s, v_sat)
            acc += tl.sum(ip - inn, axis=2)

        tl.store(out + offs_m[:, None] * O + offs_o[None, :], acc, mask=(offs_m[:, None] < M) & (offs_o[None, :] < O))


    @triton.jit
    def _grad_x_kernel(
        x, wpos, wneg, grad_out, grad_x,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_O: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_o = tl.arange(0, BLOCK_O)
        xv = tl.load(x + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)

        for o0 in range(0, O, BLOCK_O):
            o = o0 + offs_o
            g = tl.load(grad_out + offs_m[:, None] * O + o[None, :], mask=(offs_m[:, None] < M) & (o[None, :] < O), other=0.0).to(tl.float32)
            wp = tl.load(wpos + o[:, None] * K + offs_k[None, :], mask=(o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
            wn = tl.load(wneg + o[:, None] * K + offs_k[None, :], mask=(o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
            dpos = _ekv_grad_v(xv[:, None, :], wp[None, :, :], inv_2nut, v_d, i_s, v_sat)
            dneg = _ekv_grad_v(xv[:, None, :], wn[None, :, :], inv_2nut, v_d, i_s, v_sat)
            acc += tl.sum(g[:, :, None] * (dpos - dneg), axis=1)

        tl.store(grad_x + offs_m[:, None] * K + offs_k[None, :], acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


    @triton.jit
    def _grad_w_kernel(
        x, wpos, wneg, grad_out, grad_wpos, grad_wneg,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        inv_2nut: tl.constexpr, v_d: tl.constexpr, i_s: tl.constexpr, v_sat: tl.constexpr,
        BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_o = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        wp = tl.load(wpos + offs_o[:, None] * K + offs_k[None, :], mask=(offs_o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        wn = tl.load(wneg + offs_o[:, None] * K + offs_k[None, :], mask=(offs_o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        acc_wp = tl.zeros((BLOCK_O, BLOCK_K), tl.float32)
        acc_wn = tl.zeros((BLOCK_O, BLOCK_K), tl.float32)

        for m0 in range(0, M, BLOCK_M):
            m = m0 + offs_m
            xv = tl.load(x + m[:, None] * K + offs_k[None, :], mask=(m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
            g = tl.load(grad_out + m[None, :] * O + offs_o[:, None], mask=(offs_o[:, None] < O) & (m[None, :] < M), other=0.0).to(tl.float32)
            dwp = _ekv_grad_vth(xv[None, :, :], wp[:, None, :], inv_2nut, v_d, i_s, v_sat)
            dwn = _ekv_grad_vth(xv[None, :, :], wn[:, None, :], inv_2nut, v_d, i_s, v_sat)
            acc_wp += tl.sum(g[:, :, None] * dwp, axis=1)
            acc_wn += tl.sum(g[:, :, None] * (-dwn), axis=1)

        tl.store(grad_wpos + offs_o[:, None] * K + offs_k[None, :], acc_wp, mask=(offs_o[:, None] < O) & (offs_k[None, :] < K))
        tl.store(grad_wneg + offs_o[:, None] * K + offs_k[None, :], acc_wn, mask=(offs_o[:, None] < O) & (offs_k[None, :] < K))


class TritonEKVMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, wpos: torch.Tensor, wneg: torch.Tensor, cfg: Dict[str, Any]):
        m, k = x.shape
        o = wpos.shape[0]
        out = torch.empty((m, o), device=x.device, dtype=x.dtype)
        inv_2nut = 1.0 / (2.0 * float(cfg['n']) * float(cfg['U_T']))
        v_d = float(cfg['V_D'])
        i_s = float(cfg['I_S'])
        v_sat = float(cfg['V_sat'])
        bm = int(cfg.get('TRITON_BLOCK_M', 16))
        bo = int(cfg.get('TRITON_BLOCK_O', 16))
        bk = int(cfg.get('TRITON_BLOCK_K', 32))
        _forward_kernel[(_ceil_div(m, bm), _ceil_div(o, bo))](
            x.contiguous(), wpos.contiguous(), wneg.contiguous(), out,
            m, k, o, inv_2nut, v_d, i_s, v_sat,
            BLOCK_M=bm, BLOCK_O=bo, BLOCK_K=bk,
            num_warps=4,
        )
        ctx.save_for_backward(x, wpos, wneg)
        ctx.cfg = dict(cfg)
        ctx.blocks = (bm, bo, bk)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, wpos, wneg = ctx.saved_tensors
        cfg = ctx.cfg
        bm, bo, bk = ctx.blocks
        m, k = x.shape
        o = wpos.shape[0]
        grad_out = grad_out.contiguous()
        inv_2nut = 1.0 / (2.0 * float(cfg['n']) * float(cfg['U_T']))
        v_d = float(cfg['V_D'])
        i_s = float(cfg['I_S'])
        v_sat = float(cfg['V_sat'])

        grad_x = torch.empty_like(x, dtype=torch.float32) if x.requires_grad else None
        grad_wpos = torch.empty_like(wpos, dtype=torch.float32)
        grad_wneg = torch.empty_like(wneg, dtype=torch.float32)

        if grad_x is not None:
            _grad_x_kernel[(_ceil_div(m, bm), _ceil_div(k, bk))](
                x.contiguous(), wpos.contiguous(), wneg.contiguous(), grad_out, grad_x,
                m, k, o, inv_2nut, v_d, i_s, v_sat,
                BLOCK_M=bm, BLOCK_K=bk, BLOCK_O=bo,
                num_warps=4,
            )
        _grad_w_kernel[(_ceil_div(o, bo), _ceil_div(k, bk))](
            x.contiguous(), wpos.contiguous(), wneg.contiguous(), grad_out, grad_wpos, grad_wneg,
            m, k, o, inv_2nut, v_d, i_s, v_sat,
            BLOCK_O=bo, BLOCK_K=bk, BLOCK_M=bm,
            num_warps=4,
        )
        if grad_x is not None:
            grad_x = grad_x.to(dtype=x.dtype)
        return grad_x, grad_wpos.to(dtype=wpos.dtype), grad_wneg.to(dtype=wneg.dtype), None


def triton_ekv_matmul(x: torch.Tensor, wpos: torch.Tensor, wneg: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('triton is not available')
    return TritonEKVMatmulFn.apply(x, wpos, wneg, cfg)
