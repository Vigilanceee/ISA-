"""
STT-RAM (自旋转移矩磁随机存储器) Triton Kernels & Autograd Function

Physics:
  theta = w                           (magnetisation angle, constrained in [0, pi])
  I     = I0_stt * V * (1 + alpha*V^2) * (1 + TMR*cos(theta))
  I_net = I_pos - I_neg
        = I0_stt * V * (1+alpha*V^2) * TMR * [cos(theta_pos) - cos(theta_neg)]

Factorised form  =>  output = S @ C^T
  S[m,k] = I0_stt * V[m,k] * (1 + alpha * V[m,k]^2)
  C[n,k] = TMR * [cos(theta_pos) - cos(theta_neg)]
"""

import math as pymath
import torch
import triton
import triton.language as tl

_BLOCK_EW = 1024
_BLOCK_M  = 64
_BLOCK_N  = 64
_BLOCK_K  = 32
_PI       = pymath.pi

# ═══════════════════════ FORWARD: fused matmul ═══════════════════════

@triton.jit
def _stt_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr,
    M, N, K,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    I0_stt, alpha, TMR, PI,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        v_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        v = tl.load(V_ptr + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk,
                    mask=v_mask, other=0.0).to(tl.float32)

        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        wp = tl.load(Wp_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                     mask=w_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                     mask=w_mask, other=0.0).to(tl.float32)

        s = I0_stt * v * (1.0 + alpha * v * v)

        c = TMR * (tl.cos(wp) - tl.cos(wn))

        acc += tl.dot(s, c)

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=o_mask)

# ═══════════════════ BACKWARD: element-wise kernels ══════════════════

@triton.jit
def _stt_s_kernel(V_ptr, S_ptr, n, I0_stt, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(V_ptr + offs, mask=mask).to(tl.float32)
    tl.store(S_ptr + offs, I0_stt * v * (1.0 + alpha * v * v), mask=mask)


@triton.jit
def _stt_c_kernel(Wp_ptr, Wn_ptr, C_ptr, n, TMR, PI, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    wp = tl.load(Wp_ptr + offs, mask=mask).to(tl.float32)
    wn = tl.load(Wn_ptr + offs, mask=mask).to(tl.float32)
    tl.store(C_ptr + offs, TMR * (tl.cos(wp) - tl.cos(wn)), mask=mask)


@triton.jit
def _stt_grad_v_kernel(dLdS_ptr, V_ptr, dLdV_ptr, n, I0_stt, alpha,
                       BLOCK: tl.constexpr):
    """dS/dV = I0_stt * (1 + 3*alpha*V^2)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dl_ds = tl.load(dLdS_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(V_ptr + offs, mask=mask).to(tl.float32)
    ds_dv = I0_stt * (1.0 + 3.0 * alpha * v * v)
    tl.store(dLdV_ptr + offs, dl_ds * ds_dv, mask=mask)


@triton.jit
def _stt_grad_w_kernel(
    dLdC_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr, n,
    TMR, PI, BLOCK: tl.constexpr,
):
    """
    dC/dw_pos = TMR * (-sin(theta_pos))
    dC/dw_neg = TMR *   sin(theta_neg)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dl = tl.load(dLdC_ptr + offs, mask=mask).to(tl.float32)
    wp = tl.load(Wp_ptr + offs, mask=mask).to(tl.float32)
    wn = tl.load(Wn_ptr + offs, mask=mask).to(tl.float32)

    dc_dwp = TMR * (-tl.sin(wp))
    dc_dwn = TMR * tl.sin(wn)

    tl.store(dWp_ptr + offs, dl * dc_dwp, mask=mask)
    tl.store(dWn_ptr + offs, dl * dc_dwn, mask=mask)

# ═══════════════════════ Autograd Function ═══════════════════════════

def _grid_ew(n):
    return lambda meta: (triton.cdiv(n, meta['BLOCK']),)


class STTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        v_in  = v_in.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        M, K = v_in.shape
        N    = w_pos.shape[0]

        I0_stt = float(params['I0_stt'])
        alpha  = float(params['alpha'])
        TMR    = float(params['TMR'])

        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _stt_fwd_kernel[grid](
            v_in, w_pos, w_neg, out,
            M, N, K,
            v_in.stride(0),  v_in.stride(1),
            w_pos.stride(0), w_pos.stride(1),
            out.stride(0),   out.stride(1),
            I0_stt, alpha, TMR, _PI,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        ctx.save_for_backward(v_in, w_pos, w_neg)
        ctx.p = (I0_stt, alpha, TMR)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn = ctx.saved_tensors
        I0_stt, alpha, TMR = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wp.shape[0]
        nv   = M * K
        nw   = N * K

        S = torch.empty_like(v)
        C = torch.empty_like(wp)
        _stt_s_kernel[_grid_ew(nv)](v, S, nv, I0_stt, alpha, BLOCK=_BLOCK_EW)
        _stt_c_kernel[_grid_ew(nw)](wp, wn, C, nw, TMR, _PI, BLOCK=_BLOCK_EW)

        dL_dS = torch.mm(grad_out, C)
        dL_dC = torch.mm(grad_out.t(), S)
        del S, C

        dL_dV = torch.empty_like(v)
        _stt_grad_v_kernel[_grid_ew(nv)](dL_dS, v, dL_dV, nv,
                                         I0_stt, alpha, BLOCK=_BLOCK_EW)

        dWp = torch.empty_like(wp)
        dWn = torch.empty_like(wn)
        _stt_grad_w_kernel[_grid_ew(nw)](dL_dC, wp, wn, dWp, dWn, nw,
                                         TMR, _PI, BLOCK=_BLOCK_EW)
        return dL_dV, dWp, dWn, None
