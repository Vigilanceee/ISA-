"""
ReRAM (阻变存储器) Triton Kernels & Autograd Function

Physics:
  g     = g_min + tanh(w) * (g_max - g_min)
  I     = I0 * exp(-g / g0) * sinh(V_in / V0)
  I_net = I_pos - I_neg

Factorised form  =>  output = S @ C^T
  S[m,k] = sinh(V[m,k] / V0)
  C[n,k] = I0 * (exp(-g_pos[n,k]/g0) - exp(-g_neg[n,k]/g0))
"""

import torch
import triton
import triton.language as tl

# ───────────────────────── helper constants ─────────────────────────
_BLOCK_EW = 1024          # element-wise kernel block size
_BLOCK_M  = 64
_BLOCK_N  = 64
_BLOCK_K  = 32

# ═══════════════════════ FORWARD: fused matmul ═══════════════════════

@triton.jit
def _reram_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr,
    M, N, K,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    I0, g0, V0, g_min, g_max,
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

        v_s = tl.maximum(tl.minimum(v / V0, 10.0), -10.0)
        s = (tl.exp(v_s) - tl.exp(-v_s)) * 0.5

        # w directly represents physical gap g.
        g_p = wp
        g_n = wn
        c = I0 * (tl.exp(-g_p / g0) - tl.exp(-g_n / g0))

        acc += tl.dot(s, c)

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=o_mask)

# ═══════════════════ BACKWARD: element-wise kernels ══════════════════

@triton.jit
def _reram_s_kernel(V_ptr, S_ptr, n, V0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(V_ptr + offs, mask=mask).to(tl.float32)
    v_s = tl.maximum(tl.minimum(v / V0, 10.0), -10.0)
    s = (tl.exp(v_s) - tl.exp(-v_s)) * 0.5
    tl.store(S_ptr + offs, s, mask=mask)


@triton.jit
def _reram_c_kernel(Wp_ptr, Wn_ptr, C_ptr, n, I0, g0, g_min, g_max, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    wp = tl.load(Wp_ptr + offs, mask=mask).to(tl.float32)
    wn = tl.load(Wn_ptr + offs, mask=mask).to(tl.float32)
    g_p = wp
    g_n = wn
    c = I0 * (tl.exp(-g_p / g0) - tl.exp(-g_n / g0))
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _reram_grad_v_kernel(dLdS_ptr, V_ptr, dLdV_ptr, n, V0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dl_ds = tl.load(dLdS_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(V_ptr + offs, mask=mask).to(tl.float32)
    v_s = tl.maximum(tl.minimum(v / V0, 10.0), -10.0)
    cosh_v = (tl.exp(v_s) + tl.exp(-v_s)) * 0.5
    tl.store(dLdV_ptr + offs, dl_ds * cosh_v / V0, mask=mask)


@triton.jit
def _reram_grad_w_kernel(
    dLdC_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr, n,
    I0, g0, g_min, g_max, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dl = tl.load(dLdC_ptr + offs, mask=mask).to(tl.float32)
    wp = tl.load(Wp_ptr + offs, mask=mask).to(tl.float32)
    wn = tl.load(Wn_ptr + offs, mask=mask).to(tl.float32)
    g_p = wp
    dc_dwp = I0 * tl.exp(-g_p / g0) * (-1.0 / g0)

    g_n = wn
    dc_dwn = I0 * tl.exp(-g_n / g0) * (1.0 / g0)

    tl.store(dWp_ptr + offs, dl * dc_dwp, mask=mask)
    tl.store(dWn_ptr + offs, dl * dc_dwn, mask=mask)

# ═══════════════════════ Autograd Function ═══════════════════════════

def _grid_ew(n):
    return lambda meta: (triton.cdiv(n, meta['BLOCK']),)


class ReRAMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        v_in  = v_in.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        M, K = v_in.shape
        N    = w_pos.shape[0]

        I0    = float(params['I0'])
        g0    = float(params['g0'])
        V0    = float(params['V0'])
        g_min = float(params['g_min'])
        g_max = float(params['g_max'])

        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _reram_fwd_kernel[grid](
            v_in, w_pos, w_neg, out,
            M, N, K,
            v_in.stride(0),  v_in.stride(1),
            w_pos.stride(0), w_pos.stride(1),
            out.stride(0),   out.stride(1),
            I0, g0, V0, g_min, g_max,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        ctx.save_for_backward(v_in, w_pos, w_neg)
        ctx.p = (I0, g0, V0, g_min, g_max)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn = ctx.saved_tensors
        I0, g0, V0, g_min, g_max = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wp.shape[0]
        nv   = M * K
        nw   = N * K

        S = torch.empty_like(v)
        C = torch.empty_like(wp)
        _reram_s_kernel[_grid_ew(nv)](v, S, nv, V0, BLOCK=_BLOCK_EW)
        _reram_c_kernel[_grid_ew(nw)](wp, wn, C, nw, I0, g0, g_min, g_max, BLOCK=_BLOCK_EW)

        dL_dS = torch.mm(grad_out, C)        # (M,N)@(N,K)
        dL_dC = torch.mm(grad_out.t(), S)     # (N,M)@(M,K)
        del S, C

        dL_dV = torch.empty_like(v)
        _reram_grad_v_kernel[_grid_ew(nv)](dL_dS, v, dL_dV, nv, V0, BLOCK=_BLOCK_EW)

        dWp = torch.empty_like(wp)
        dWn = torch.empty_like(wn)
        _reram_grad_w_kernel[_grid_ew(nw)](dL_dC, wp, wn, dWp, dWn, nw,
                                           I0, g0, g_min, g_max, BLOCK=_BLOCK_EW)
        return dL_dV, dWp, dWn, None
