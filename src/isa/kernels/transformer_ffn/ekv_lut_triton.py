"""LUT-accelerated fused Triton EKV matmul."""

from __future__ import annotations

from typing import Any, Dict

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

from isa.approximations.delta_lut import get_ekv_lut_tables


def _ceil_div(a: int, b: int) -> int:
    return triton.cdiv(a, b)


if triton is not None:

    @triton.jit
    def _lut_lookup(table, delta, DELTA_MIN: tl.constexpr, DELTA_MAX: tl.constexpr, DELTA_SIZE: tl.constexpr):
        f = (delta - DELTA_MIN) * ((DELTA_SIZE - 1) / (DELTA_MAX - DELTA_MIN))
        f = tl.minimum(tl.maximum(f, 0.0), (DELTA_SIZE - 1) * 1.0)
        i0 = f.to(tl.int32)
        i1 = tl.minimum(i0 + 1, DELTA_SIZE - 1)
        a = f - i0
        q0 = tl.load(table + i0)
        q1 = tl.load(table + i1)
        return q0 * (1.0 - a) + q1 * a


    @triton.jit
    def _forward_kernel(
        x, wpos, wneg, table_i, out,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        DELTA_MIN: tl.constexpr, DELTA_MAX: tl.constexpr, DELTA_SIZE: tl.constexpr,
        V_SAT: tl.constexpr,
        V_OUT_MIN: tl.constexpr, V_OUT_MAX: tl.constexpr,
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
            base_p = _lut_lookup(table_i, xv[:, None, :] - wp[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            base_n = _lut_lookup(table_i, xv[:, None, :] - wn[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            denom = 1.0 + xv[:, None, :] / V_SAT
            acc += tl.sum((base_p - base_n) / denom, axis=2)
        acc = tl.minimum(tl.maximum(acc, V_OUT_MIN), V_OUT_MAX)
        tl.store(out + offs_m[:, None] * O + offs_o[None, :], acc, mask=(offs_m[:, None] < M) & (offs_o[None, :] < O))


    @triton.jit
    def _grad_x_kernel(
        x, wpos, wneg, grad_out, table_i, table_ddelta, grad_x,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        DELTA_MIN: tl.constexpr, DELTA_MAX: tl.constexpr, DELTA_SIZE: tl.constexpr,
        V_SAT: tl.constexpr,
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
            base_p = _lut_lookup(table_i, xv[:, None, :] - wp[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            base_n = _lut_lookup(table_i, xv[:, None, :] - wn[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            grad_p = _lut_lookup(table_ddelta, xv[:, None, :] - wp[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            grad_n = _lut_lookup(table_ddelta, xv[:, None, :] - wn[None, :, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            denom = 1.0 + xv[:, None, :] / V_SAT
            deriv = (grad_p - grad_n) / denom - (base_p - base_n) / (V_SAT * denom * denom)
            acc += tl.sum(g[:, :, None] * deriv, axis=1)
        tl.store(grad_x + offs_m[:, None] * K + offs_k[None, :], acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


    @triton.jit
    def _grad_w_kernel(
        x, wpos, wneg, grad_out, table_ddelta, grad_wpos, grad_wneg,
        M: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
        DELTA_MIN: tl.constexpr, DELTA_MAX: tl.constexpr, DELTA_SIZE: tl.constexpr,
        V_SAT: tl.constexpr,
        BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, SPLIT_M: tl.constexpr,
    ):
        pid_o = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_split = tl.program_id(2)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        wp = tl.load(wpos + offs_o[:, None] * K + offs_k[None, :], mask=(offs_o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        wn = tl.load(wneg + offs_o[:, None] * K + offs_k[None, :], mask=(offs_o[:, None] < O) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        acc_wp = tl.zeros((BLOCK_O, BLOCK_K), tl.float32)
        acc_wn = tl.zeros((BLOCK_O, BLOCK_K), tl.float32)
        rows_per_split: tl.constexpr = tl.cdiv(M, SPLIT_M)
        split_start = pid_split * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, M)
        for m0 in range(0, rows_per_split, BLOCK_M):
            m = split_start + m0 + offs_m
            m_mask = m < split_end
            xv = tl.load(x + m[:, None] * K + offs_k[None, :], mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0).to(tl.float32)
            g = tl.load(grad_out + m[None, :] * O + offs_o[:, None], mask=(offs_o[:, None] < O) & m_mask[None, :], other=0.0).to(tl.float32)
            denom = 1.0 + xv[None, :, :] / V_SAT
            grad_p = _lut_lookup(table_ddelta, xv[None, :, :] - wp[:, None, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            grad_n = _lut_lookup(table_ddelta, xv[None, :, :] - wn[:, None, :], DELTA_MIN, DELTA_MAX, DELTA_SIZE)
            acc_wp += tl.sum(g[:, :, None] * (-grad_p / denom), axis=1)
            acc_wn += tl.sum(g[:, :, None] * (grad_n / denom), axis=1)
        mask = (offs_o[:, None] < O) & (offs_k[None, :] < K)
        tl.atomic_add(grad_wpos + offs_o[:, None] * K + offs_k[None, :], acc_wp, mask=mask)
        tl.atomic_add(grad_wneg + offs_o[:, None] * K + offs_k[None, :], acc_wn, mask=mask)


class LUTTritonEKVMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, wpos: torch.Tensor, wneg: torch.Tensor, cfg: Dict[str, Any]):
        m, k = x.shape
        o = wpos.shape[0]
        table_i, table_ddelta = get_ekv_lut_tables(cfg, x.device)
        out = torch.empty((m, o), device=x.device, dtype=torch.float32)
        v_min = float(cfg.get('LUT_V_MIN', cfg.get('V_signed_min', -4.0)))
        v_max = float(cfg.get('LUT_V_MAX', cfg.get('V_signed_max', 4.0)))
        vth_min = float(cfg.get('LUT_VTH_MIN', cfg.get('V_TH_MIN', 0.0)))
        vth_max = float(cfg.get('LUT_VTH_MAX', cfg.get('V_TH_MAX', 5.0)))
        delta_min = float(cfg.get('LUT_DELTA_MIN', v_min - vth_max))
        delta_max = float(cfg.get('LUT_DELTA_MAX', v_max - vth_min))
        delta_size = int(cfg.get('LUT_DELTA_SIZE', 16384))
        v_sat = float(cfg['V_sat'])
        bm = int(cfg.get('TRITON_BLOCK_M', 16))
        bo = int(cfg.get('TRITON_BLOCK_O', 16))
        bk = int(cfg.get('TRITON_BLOCK_K', 32))
        v_out_min = float(cfg.get('V_signed_min', -4.0))
        v_out_max = float(cfg.get('V_signed_max', 4.0))
        _forward_kernel[(_ceil_div(m, bm), _ceil_div(o, bo))](
            x.contiguous(), wpos.contiguous(), wneg.contiguous(), table_i, out,
            m, k, o, delta_min, delta_max, delta_size, v_sat,
            V_OUT_MIN=v_out_min, V_OUT_MAX=v_out_max,
            BLOCK_M=bm, BLOCK_O=bo, BLOCK_K=bk,
            num_warps=4,
        )
        ctx.save_for_backward(x, wpos, wneg, table_i, table_ddelta)
        ctx.cfg = dict(cfg)
        ctx.meta = (delta_min, delta_max, delta_size, v_sat, bm, bo, bk)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, wpos, wneg, table_i, table_ddelta = ctx.saved_tensors
        delta_min, delta_max, delta_size, v_sat, bm, bo, bk = ctx.meta
        m, k = x.shape
        o = wpos.shape[0]
        grad_out = grad_out.contiguous()

        if bool(ctx.cfg.get("CUDA_SHARED_BACKWARD", False)):
            if delta_size > 8192:
                raise RuntimeError(
                    "CUDA shared EKV backward requires LUT_DELTA_SIZE <= 8192"
                )
            from isa.kernels.transformer_ffn.cuda_backend import shared_backward

            grad_x, grad_wpos, grad_wneg = shared_backward(
                x,
                wpos,
                wneg,
                grad_out,
                table_i,
                table_ddelta,
                delta_min,
                delta_max,
                v_sat,
                need_grad_x=x.requires_grad,
                need_grad_w=wpos.requires_grad or wneg.requires_grad,
            )
            return grad_x, grad_wpos, grad_wneg, None

        grad_x = torch.empty_like(x, dtype=torch.float32) if x.requires_grad else None
        grad_wpos = torch.zeros_like(wpos, dtype=torch.float32) if wpos.requires_grad else None
        grad_wneg = torch.zeros_like(wneg, dtype=torch.float32) if wneg.requires_grad else None
        if grad_x is not None:
            _grad_x_kernel[(_ceil_div(m, bm), _ceil_div(k, bk))](
                x.contiguous(), wpos.contiguous(), wneg.contiguous(), grad_out, table_i, table_ddelta, grad_x,
                m, k, o, delta_min, delta_max, delta_size, v_sat,
                BLOCK_M=bm, BLOCK_K=bk, BLOCK_O=bo,
                num_warps=4,
            )
        if grad_wpos is not None or grad_wneg is not None:
            split_m = min(int(ctx.cfg.get('TRITON_SPLIT_M', 8)), max(1, _ceil_div(m, bm)))
            _grad_w_kernel[(_ceil_div(o, bo), _ceil_div(k, bk), split_m)](
                x.contiguous(), wpos.contiguous(), wneg.contiguous(), grad_out, table_ddelta, grad_wpos, grad_wneg,
                m, k, o, delta_min, delta_max, delta_size, v_sat,
                BLOCK_O=bo, BLOCK_K=bk, BLOCK_M=bm, SPLIT_M=split_m,
                num_warps=4,
            )
        if grad_x is not None:
            grad_x = grad_x.to(dtype=x.dtype)
        return (
            grad_x,
            grad_wpos.to(dtype=wpos.dtype) if grad_wpos is not None else None,
            grad_wneg.to(dtype=wneg.dtype) if grad_wneg is not None else None,
            None,
        )


def triton_lut_ekv_matmul(x: torch.Tensor, wpos: torch.Tensor, wneg: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('triton is not available')
    return LUTTritonEKVMatmulFn.apply(x, wpos, wneg, cfg)
