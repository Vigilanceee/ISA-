"""
PCM (Phase Change Memory) Triton Kernels & Autograd Function

Physics:
  chi = w
  I0(chi) = I0_prefactor * exp(-I0_decay * chi)
  I(V, w) = I0(w) * w * sinh(w * V / V_T)
  I_net   = I_pos - I_neg

Non-factorisable (chi appears both outside and inside sinh), so the
forward uses a fused K-accumulation kernel instead of the S@C^T split.
"""

import torch
import triton
import triton.language as tl

_BLOCK_M  = 64
_BLOCK_N  = 64
_BLOCK_K  = 64
_BLOCK_EW = 1024

# ═══════════════════════ FORWARD: fused K-accumulation ═════════════════

@triton.jit
def _pcm_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr,
    M, N, K,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    I0, I0_DECAY, V_T,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K):
        # Load V[m,k]  (BLOCK_M,)
        m_mask = offs_m < M
        v = tl.load(V_ptr + offs_m * stride_vm + k * stride_vk,
                    mask=m_mask, other=0.0).to(tl.float32)

        # Load Wp[n,k], Wn[n,k]  (BLOCK_N,)
        n_mask = offs_n < N
        wp = tl.load(Wp_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)

        # w directly represents the physical PCM state chi.
        chi_p = wp
        chi_n = wn

        # arg = chi * v / V_T  →  outer product (BLOCK_M, BLOCK_N)
        v_scaled = v / V_T
        arg_p = chi_p[None, :] * v_scaled[:, None]
        arg_n = chi_n[None, :] * v_scaled[:, None]

        # sinh(x) = (exp(x) - exp(-x)) / 2
        arg_p_safe = tl.maximum(tl.minimum(arg_p, 20.0), -20.0)
        ep = tl.exp(arg_p_safe)
        en = tl.exp(-arg_p_safe)
        sinh_p = (ep - en) * 0.5
        i0_p = I0 * tl.exp(-I0_DECAY * chi_p)
        ip = i0_p[None, :] * chi_p[None, :] * sinh_p

        arg_n_safe = tl.maximum(tl.minimum(arg_n, 20.0), -20.0)
        epn = tl.exp(arg_n_safe)
        enn = tl.exp(-arg_n_safe)
        sinh_n = (epn - enn) * 0.5
        i0_n = I0 * tl.exp(-I0_DECAY * chi_n)
        ineg = i0_n[None, :] * chi_n[None, :] * sinh_n

        acc += ip - ineg

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=o_mask)


# ═══════════════════ BACKWARD: element-wise reduction kernels ══════════

@triton.jit
def _pcm_grad_v_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dvm, stride_dvk,
    I0, I0_DECAY, V_T,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dV[m,k] = sum_n GO[m,n] * (dIpos_dV - dIneg_dV)

    dIpos_dV = I0(chi_p) * chi_p[n,k]^2 / V_T * cosh(chi_p[n,k] * V[m,k] / V_T)
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K
    mk_mask = m_mask & k_mask

    v = tl.load(V_ptr + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk,
                mask=mk_mask, other=0.0).to(tl.float32)
    v_scaled = v / V_T

    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(N):
        go = tl.load(GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon,
                     mask=m_mask, other=0.0).to(tl.float32)  # (BLOCK_M, 1)

        wp = tl.load(Wp_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)
        wn = tl.load(Wn_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)

        chi_p = wp
        chi_n = wn

        # cosh via exp
        arg_p = chi_p * v_scaled
        arg_p_safe = tl.maximum(tl.minimum(arg_p, 20.0), -20.0)
        cosh_p = (tl.exp(arg_p_safe) + tl.exp(-arg_p_safe)) * 0.5
        i0_p = I0 * tl.exp(-I0_DECAY * chi_p)
        dIpos_dV = i0_p * (chi_p * chi_p / V_T) * cosh_p

        arg_n = chi_n * v_scaled
        arg_n_safe = tl.maximum(tl.minimum(arg_n, 20.0), -20.0)
        cosh_n = (tl.exp(arg_n_safe) + tl.exp(-arg_n_safe)) * 0.5
        i0_n = I0 * tl.exp(-I0_DECAY * chi_n)
        dIneg_dV = i0_n * (chi_n * chi_n / V_T) * cosh_n

        dv += go * (dIpos_dV - dIneg_dV)

    tl.store(dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk,
             dv, mask=mk_mask)


@triton.jit
def _pcm_grad_w_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dwn, stride_dwk,
    I0, I0_DECAY, V_T,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dWp[n,k] = sum_m GO[m,n] * dIpos_dW[m,k]

    dIpos_dW = I0(chi) * [(1 - I0_decay*chi) * sinh(arg) + chi*(V/V_T)*cosh(arg)]
      where arg = chi * V / V_T, chi = w
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < K
    nk_mask = n_mask & k_mask

    wp = tl.load(Wp_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                 mask=nk_mask, other=0.0).to(tl.float32)
    wn = tl.load(Wn_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                 mask=nk_mask, other=0.0).to(tl.float32)

    chi_p = wp
    chi_n = wn

    dwp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwn = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m in range(M):
        go = tl.load(GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon,
                     mask=n_mask, other=0.0).to(tl.float32)  # (BLOCK_N, 1)

        v = tl.load(V_ptr + m * stride_vm + offs_k[None, :] * stride_vk,
                    mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)
        v_scaled = v / V_T

        arg_p = chi_p * v_scaled
        arg_n = chi_n * v_scaled

        # sinh(arg), cosh(arg)
        argp_safe = tl.maximum(tl.minimum(arg_p, 20.0), -20.0)
        ep = tl.exp(argp_safe); en = tl.exp(-argp_safe)
        sinh_p = (ep - en) * 0.5; cosh_p = (ep + en) * 0.5

        argn_safe = tl.maximum(tl.minimum(arg_n, 20.0), -20.0)
        epn = tl.exp(argn_safe); enn = tl.exp(-argn_safe)
        sinh_n = (epn - enn) * 0.5; cosh_n = (epn + enn) * 0.5

        i0_p = I0 * tl.exp(-I0_DECAY * chi_p)
        i0_n = I0 * tl.exp(-I0_DECAY * chi_n)

        # d/dw of I0(chi)*chi*sinh(arg), arg=chi*(V/V_T)
        dIpos_dw = i0_p * ((1.0 - I0_DECAY * chi_p) * sinh_p + chi_p * v_scaled * cosh_p)
        dIneg_dw = i0_n * ((1.0 - I0_DECAY * chi_n) * sinh_n + chi_n * v_scaled * cosh_n)

        dwp += go * dIpos_dw
        dwn += go * (-dIneg_dw)

    tl.store(dWp_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwp, mask=nk_mask)
    tl.store(dWn_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwn, mask=nk_mask)


# ═══════════════════════ Autograd Function ═══════════════════════════

def _grid_2d(n, m):
    return lambda meta: (triton.cdiv(n, meta['BLOCK_M']),
                         triton.cdiv(m, meta['BLOCK_N']))


class PCMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        v_in  = v_in.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        M, K = v_in.shape
        N    = w_pos.shape[0]

        I0 = float(params['I0_pcm'])
        I0_DECAY = float(params.get('I0_pcm_decay', 0.0))
        V_T = float(params['V_T_pcm'])

        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _pcm_fwd_kernel[grid](
            v_in, w_pos, w_neg, out,
            M, N, K,
            v_in.stride(0),  v_in.stride(1),
            w_pos.stride(0), w_pos.stride(1),
            out.stride(0),   out.stride(1),
            I0, I0_DECAY, V_T,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )

        ctx.save_for_backward(v_in, w_pos, w_neg)
        ctx.p = (I0, I0_DECAY, V_T)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn = ctx.saved_tensors
        I0, I0_DECAY, V_T = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wp.shape[0]

        dL_dV = torch.empty_like(v)
        grid_v = (triton.cdiv(M, _BLOCK_M), triton.cdiv(K, _BLOCK_K))
        _pcm_grad_v_kernel[grid_v](
            grad_out, v, wp, wn, dL_dV,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wp.stride(0), wp.stride(1),
            dL_dV.stride(0), dL_dV.stride(1),
            I0, I0_DECAY, V_T,
            BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K,
        )

        dWp = torch.empty_like(wp)
        dWn = torch.empty_like(wn)
        grid_w = (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
        _pcm_grad_w_kernel[grid_w](
            grad_out, v, wp, wn, dWp, dWn,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wp.stride(0), wp.stride(1),
            dWp.stride(0), dWp.stride(1),
            I0, I0_DECAY, V_T,
            BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        return dL_dV, dWp, dWn, None
