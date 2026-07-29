"""
Flash Transistor I-V Model — Triton Kernels & Autograd Function

Physics (EKV model with velocity saturation via denominator):
  Vth = theta (directly trainable)
  u1 = (V - Vth) * inv_2nUT
  u2 = (V - Vth - V_D) * inv_2nUT
  I_basic = softplus(u1)^2 - softplus(u2)^2
  I = I_S * I_basic / (1 + V / V_sat)

The forward uses a fused K-accumulation Triton kernel.
The backward uses element-wise Triton reduction kernels (no Python loops).
"""

import torch
import triton
import triton.language as tl

_BLOCK_M  = 64
_BLOCK_N  = 64
_BLOCK_K  = 64


# ═══════════════ Triton helpers ═══════════════

@triton.jit
def _sp(x):
    """Numerically stable softplus."""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _sigm(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + tl.exp(-x))


# ═══════════════════ FORWARD: fused K-accumulation ═════════════════

@triton.jit
def _flash_fwd_kernel(
    V_ptr, Wp_ptr, Wn_ptr, O_ptr,
    M, N, K,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K):
        m_mask = offs_m < M
        v = tl.load(V_ptr + offs_m * stride_vm + k * stride_vk,
                    mask=m_mask, other=0.0).to(tl.float32)

        n_mask = offs_n < N
        wp = tl.load(Wp_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)
        wn = tl.load(Wn_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)

        vth_p = wp
        vth_n = wn

        v_2d = v[:, None]  # (BLOCK_M, 1)
        denom = 1.0 + v_2d / V_sat

        # positive branch
        u_p = (v_2d - vth_p[None, :]) * inv_2nUT
        sp_a = _sp(u_p)
        sp_b = _sp(u_p - VD_over_2nUT)
        I_pos = sp_a * sp_a - sp_b * sp_b

        # negative branch
        u_n = (v_2d - vth_n[None, :]) * inv_2nUT
        sp_an = _sp(u_n)
        sp_bn = _sp(u_n - VD_over_2nUT)
        I_neg = sp_an * sp_an - sp_bn * sp_bn

        acc += (I_pos - I_neg) / denom

    acc = acc * I_S
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=o_mask)


# ═══════════════════ BACKWARD: element-wise reduction kernels ══════════

@triton.jit
def _flash_grad_v_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dvm, stride_dvk,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dV[m,k] = sum_n GO[m,n] * (dI_pos/dV - dI_neg/dV)

    dI/dV = I_S * [didv_basic / denom - i_basic / (denom^2 * V_sat)]
      where denom = 1 + V/V_sat
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
    denom = 1.0 + v / V_sat
    inv_denom = 1.0 / denom

    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(N):
        go = tl.load(GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon,
                     mask=m_mask, other=0.0).to(tl.float32)  # (BLOCK_M, 1)

        wp = tl.load(Wp_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)
        wn = tl.load(Wn_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)

        vth_p = wp; vth_n = wn

        # positive branch: didv and i_basic
        u_p = (v - vth_p) * inv_2nUT
        sp_p1 = _sp(u_p)
        sp_p2 = _sp(u_p - VD_over_2nUT)
        sig_p1 = _sigm(u_p)
        sig_p2 = _sigm(u_p - VD_over_2nUT)
        i_basic_p = sp_p1 * sp_p1 - sp_p2 * sp_p2
        didv_p = 2.0 * (sp_p1 * sig_p1 - sp_p2 * sig_p2) * inv_2nUT
        dI_dV_p = didv_p * inv_denom - i_basic_p * inv_denom * inv_denom / V_sat

        # negative branch
        u_n = (v - vth_n) * inv_2nUT
        sp_n1 = _sp(u_n)
        sp_n2 = _sp(u_n - VD_over_2nUT)
        sig_n1 = _sigm(u_n)
        sig_n2 = _sigm(u_n - VD_over_2nUT)
        i_basic_n = sp_n1 * sp_n1 - sp_n2 * sp_n2
        didv_n = 2.0 * (sp_n1 * sig_n1 - sp_n2 * sig_n2) * inv_2nUT
        dI_dV_n = didv_n * inv_denom - i_basic_n * inv_denom * inv_denom / V_sat

        dv += go * I_S * (dI_dV_p - dI_dV_n)

    tl.store(dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk,
             dv, mask=mk_mask)


@triton.jit
def _flash_grad_w_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dwn, stride_dwk,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dWp[n,k] = sum_m GO[m,n] * dI_pos/d(vth_pos)
    dL/dWn[n,k] = sum_m GO[m,n] * (-dI_neg/d(vth_neg))  # -GO from I_net=I_pos-I_neg

    dI/dvth = -I_S * didv_basic * inv_denom
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

    dwp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwn = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m in range(M):
        go = tl.load(GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon,
                     mask=n_mask, other=0.0).to(tl.float32)  # (BLOCK_N, 1)

        v = tl.load(V_ptr + m * stride_vm + offs_k[None, :] * stride_vk,
                    mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)

        denom = 1.0 + v / V_sat
        inv_denom = 1.0 / denom
        vth_p = wp; vth_n = wn

        # positive branch
        u_p = (v - vth_p) * inv_2nUT
        sp_p1 = _sp(u_p); sp_p2 = _sp(u_p - VD_over_2nUT)
        sig_p1 = _sigm(u_p); sig_p2 = _sigm(u_p - VD_over_2nUT)
        didv_p = 2.0 * (sp_p1 * sig_p1 - sp_p2 * sig_p2) * inv_2nUT
        dI_dvth_p = -I_S * didv_p * inv_denom

        # negative branch
        u_n = (v - vth_n) * inv_2nUT
        sp_n1 = _sp(u_n); sp_n2 = _sp(u_n - VD_over_2nUT)
        sig_n1 = _sigm(u_n); sig_n2 = _sigm(u_n - VD_over_2nUT)
        didv_n = 2.0 * (sp_n1 * sig_n1 - sp_n2 * sig_n2) * inv_2nUT
        dI_dvth_n = -I_S * didv_n * inv_denom

        # Match original code: dL_dwcom = GO * dIneg_dvth, dL_dwdiff = GO * dIneg_dvth
        dwp += go * dI_dvth_p
        dwn += -go * dI_dvth_n

    tl.store(dWp_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwp, mask=nk_mask)
    tl.store(dWn_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwn, mask=nk_mask)


# ═══════════════════════ Autograd Function ═══════════════════════════

class FlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, w_com, w_diff, params):
        v_in   = v_in.contiguous()
        w_com  = w_com.contiguous()
        w_diff = w_diff.contiguous()
        M, K = v_in.shape
        N    = w_com.shape[0]

        I_S       = float(params['I_S'])
        n_fac     = float(params['n'])
        U_T       = float(params['U_T'])
        V_D       = float(params['V_D'])
        V_sat     = float(params['V_sat'])
        inv_2nUT  = 1.0 / (2.0 * n_fac * U_T)
        VD_over_2nUT = V_D * inv_2nUT

        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _flash_fwd_kernel[grid](
            v_in, w_com, w_diff, out,
            M, N, K,
            v_in.stride(0),   v_in.stride(1),
            w_com.stride(0),  w_com.stride(1),
            out.stride(0),    out.stride(1),
            I_S, inv_2nUT, VD_over_2nUT, V_sat,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )

        ctx.save_for_backward(v_in, w_com, w_diff)
        ctx.p = (I_S, inv_2nUT, VD_over_2nUT, V_sat)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wc, wd = ctx.saved_tensors
        I_S, inv_2nUT, VD_over_2nUT, V_sat = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wc.shape[0]

        dL_dV = torch.empty_like(v)
        grid_v = (triton.cdiv(M, _BLOCK_M), triton.cdiv(K, _BLOCK_K))
        _flash_grad_v_kernel[grid_v](
            grad_out, v, wc, wd, dL_dV,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wc.stride(0), wc.stride(1),
            dL_dV.stride(0), dL_dV.stride(1),
            I_S, inv_2nUT, VD_over_2nUT, V_sat,
            BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K,
        )

        dWc = torch.empty_like(wc)
        dWd = torch.empty_like(wd)
        grid_w = (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
        _flash_grad_w_kernel[grid_w](
            grad_out, v, wc, wd, dWc, dWd,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wc.stride(0), wc.stride(1),
            dWc.stride(0), dWc.stride(1),
            I_S, inv_2nUT, VD_over_2nUT, V_sat,
            BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        return dL_dV, dWc, dWd, None
