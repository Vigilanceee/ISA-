"""
FeFET (Ferroelectric FET) Triton Kernels & Autograd Function

Physics:
  1) L-K mapping (gate -> internal node):
     Vext - Vth = B_lk * (Vint - Vth) + A_lk * (Vint - Vth)^3
     Solved via Newton iteration for Vint.
     Vth = theta (directly trainable, no vcom offset).
  2) EKV channel current:
     I_basic = I_S * [softplus((Vint-Vth)/(2nU_T))^2
                    - softplus((Vint-Vth-V_D)/(2nU_T))^2]
  3) Velocity saturation:
     I = I_basic / (1 + Vext / V_sat)

Analytical backward uses implicit differentiation through L-K solve
to avoid torch.autograd.grad (the previous OOM root cause).
"""

import torch
import triton
import triton.language as tl

_BLOCK_M  = 64
_BLOCK_N  = 64
_BLOCK_K  = 64


# ═══════════════ Triton helpers ═══════════════

@triton.jit
def _softplus(x):
    """Numerically stable softplus: log(1+exp(x))."""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _solve_lk(v_ext: tl.tensor, vth: tl.tensor,
              A_lk: float, B_lk: float, iters: tl.constexpr):
    """
    Solve A*x^3 + B*x - y = 0 for x = Vint - vth, where y = v_ext - vth.
    Returns (Vint, x, inv_fp) where inv_fp = 1/(3*A*x^2 + B) for backward.
    """
    y = v_ext - vth
    x = y / tl.maximum(B_lk, 1e-8)

    for _ in range(iters):
        f = A_lk * x * x * x + B_lk * x - y
        fp = 3.0 * A_lk * x * x + B_lk
        x = x - f / tl.maximum(fp, 1e-8)

    inv_fp = 1.0 / tl.maximum(3.0 * A_lk * x * x + B_lk, 1e-8)
    return vth + x, x, inv_fp


# ═══════════════════ FORWARD: fused K-accumulation ═════════════════

@triton.jit
def _fefet_fwd_kernel(
    V_ptr, Wc_ptr, Wd_ptr, O_ptr,
    M, N, K,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
    NEWTON_ITERS: tl.constexpr,
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
        wc = tl.load(Wc_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)
        wd = tl.load(Wd_ptr + offs_n * stride_wn + k * stride_wk,
                     mask=n_mask, other=0.0).to(tl.float32)

        vth_p = wc  # (BLOCK_N,)  Vth directly trainable
        vth_n = wd

        v_ext = v[:, None]  # (BLOCK_M, 1)

        # --- positive branch ---
        y_p = v_ext - vth_p[None, :]
        x_p = y_p / tl.maximum(B_lk, 1e-8)

        for _ in range(NEWTON_ITERS):
            f = A_lk * x_p * x_p * x_p + B_lk * x_p - y_p
            fp = 3.0 * A_lk * x_p * x_p + B_lk
            x_p = x_p - f / tl.maximum(fp, 1e-8)

        u1_p = x_p * inv_2nUT
        u2_p = (x_p - V_D) * inv_2nUT
        sp1_p = _softplus(u1_p)
        sp2_p = _softplus(u2_p)
        i_basic_p = I_S * (sp1_p * sp1_p - sp2_p * sp2_p)
        denom = 1.0 + v_ext / V_sat
        i_pos = i_basic_p / denom

        # --- negative branch ---
        y_n = v_ext - vth_n[None, :]
        x_n = y_n / tl.maximum(B_lk, 1e-8)

        for _ in range(NEWTON_ITERS):
            f = A_lk * x_n * x_n * x_n + B_lk * x_n - y_n
            fp = 3.0 * A_lk * x_n * x_n + B_lk
            x_n = x_n - f / tl.maximum(fp, 1e-8)

        u1_n = x_n * inv_2nUT
        u2_n = (x_n - V_D) * inv_2nUT
        sp1_n = _softplus(u1_n)
        sp2_n = _softplus(u2_n)
        i_basic_n = I_S * (sp1_n * sp1_n - sp2_n * sp2_n)
        denom = 1.0 + v_ext / V_sat
        i_neg = i_basic_n / denom

        acc += i_pos - i_neg

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=o_mask)


# ═══════════════════ BACKWARD: analytical gradients ═════════════════

@triton.jit
def _fefet_grad_v_kernel(
    GO_ptr, V_ptr, Wc_ptr, Wd_ptr, dV_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dvm, stride_dvk,
    I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
    NEWTON_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dV[m,k] = sum_n GO[m,n] * (dI_pos/dv_ext - dI_neg/dv_ext)

    By implicit differentiation through L-K:
      dVint/dv_ext = 1 / (3*A*x^2 + B) = inv_fp
    Then chain rule through EKV + velocity saturation to get dI/dv_ext.
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
    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(N):
        go = tl.load(GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon,
                     mask=m_mask, other=0.0).to(tl.float32)  # (BLOCK_M, 1)

        wc = tl.load(Wc_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)
        wd = tl.load(Wd_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
                     mask=k_mask, other=0.0).to(tl.float32)

        vth_p = wc; vth_n = wd
        v_ext = v

        # --- positive branch gradient dI_pos/dv_ext ---
        y_p = v_ext - vth_p
        x_p = y_p / tl.maximum(B_lk, 1e-8)
        for _ in range(NEWTON_ITERS):
            f = A_lk * x_p * x_p * x_p + B_lk * x_p - y_p
            fp = 3.0 * A_lk * x_p * x_p + B_lk
            x_p = x_p - f / tl.maximum(fp, 1e-8)

        # dVint/dv_ext = 1 / (3*A*x^2 + B)
        inv_fp_p = 1.0 / tl.maximum(3.0 * A_lk * x_p * x_p + B_lk, 1e-8)

        i_basic_p, dib_dx_p = _Ibasic_and_dIdx(x_p, I_S, inv_2nUT, V_D)

        # --- negative branch gradient dI_neg/dv_ext ---
        y_n = v_ext - vth_n
        x_n = y_n / tl.maximum(B_lk, 1e-8)
        for _ in range(NEWTON_ITERS):
            f = A_lk * x_n * x_n * x_n + B_lk * x_n - y_n
            fp = 3.0 * A_lk * x_n * x_n + B_lk
            x_n = x_n - f / tl.maximum(fp, 1e-8)

        inv_fp_n = 1.0 / tl.maximum(3.0 * A_lk * x_n * x_n + B_lk, 1e-8)
        i_basic_n, dib_dx_n = _Ibasic_and_dIdx(x_n, I_S, inv_2nUT, V_D)
        denom = 1.0 + v_ext / V_sat
        inv_den = 1.0 / denom
        local = ((dib_dx_p * inv_fp_p - dib_dx_n * inv_fp_n) * inv_den
                 - (i_basic_p - i_basic_n) * inv_den * inv_den / V_sat)

        dv += go * local

    tl.store(dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk,
             dv, mask=mk_mask)


@triton.jit
def _Ibasic_and_dIdx(x, I_S, inv_2nUT, V_D):
    """Return EKV basic current and dI_basic/dx for x = Vint - Vth."""
    u1 = x * inv_2nUT
    u2 = (x - V_D) * inv_2nUT
    sp1 = _softplus(u1)
    sp2 = _softplus(u2)
    sig1 = _sigmoid(u1)
    sig2 = _sigmoid(u2)
    i_basic = I_S * (sp1 * sp1 - sp2 * sp2)
    dib_dx = I_S * 2.0 * (sp1 * sig1 - sp2 * sig2) * inv_2nUT
    return i_basic, dib_dx


@triton.jit
def _fefet_grad_w_kernel(
    GO_ptr, V_ptr, Wc_ptr, Wd_ptr, dWc_ptr, dWd_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dwn, stride_dwk,
    I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
    NEWTON_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    dL/dWc[n,k] = sum_m GO[m,n] * dI_pos/d(vth_pos)
    dL/dWd[n,k] = sum_m GO[m,n] * (-dI_neg/d(vth_neg))

    With x = Vint - Vth and y = Vext - Vth:
      dx/dvth = -inv_fp
      dI/dvth = dI_basic/dx * dx/dvth / (1 + Vext / V_sat)
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < K
    nk_mask = n_mask & k_mask

    wc = tl.load(Wc_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                 mask=nk_mask, other=0.0).to(tl.float32)
    wd = tl.load(Wd_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                 mask=nk_mask, other=0.0).to(tl.float32)

    dwc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwd = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m in range(M):
        go = tl.load(GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon,
                     mask=n_mask, other=0.0).to(tl.float32)  # (BLOCK_N, 1)

        v = tl.load(V_ptr + m * stride_vm + offs_k[None, :] * stride_vk,
                    mask=k_mask, other=0.0).to(tl.float32)  # (1, BLOCK_K)

        vth_p = wc; vth_n = wd
        v_ext = v

        # --- positive branch ---
        y_p = v_ext - vth_p
        x_p = y_p / tl.maximum(B_lk, 1e-8)
        for _ in range(NEWTON_ITERS):
            f = A_lk * x_p * x_p * x_p + B_lk * x_p - y_p
            fp = 3.0 * A_lk * x_p * x_p + B_lk
            x_p = x_p - f / tl.maximum(fp, 1e-8)

        inv_fp_p = 1.0 / tl.maximum(3.0 * A_lk * x_p * x_p + B_lk, 1e-8)
        _i_basic_p, dib_dx_p = _Ibasic_and_dIdx(x_p, I_S, inv_2nUT, V_D)
        denom = 1.0 + v_ext / V_sat
        dI_dvth_p = -dib_dx_p * inv_fp_p / denom

        # --- negative branch ---
        y_n = v_ext - vth_n
        x_n = y_n / tl.maximum(B_lk, 1e-8)
        for _ in range(NEWTON_ITERS):
            f = A_lk * x_n * x_n * x_n + B_lk * x_n - y_n
            fp = 3.0 * A_lk * x_n * x_n + B_lk
            x_n = x_n - f / tl.maximum(fp, 1e-8)

        inv_fp_n = 1.0 / tl.maximum(3.0 * A_lk * x_n * x_n + B_lk, 1e-8)
        _i_basic_n, dib_dx_n = _Ibasic_and_dIdx(x_n, I_S, inv_2nUT, V_D)
        denom = 1.0 + v_ext / V_sat
        dI_dvth_n = -dib_dx_n * inv_fp_n / denom

        # dL/d(theta_pos) = dL/dI_pos * dI_pos/d(vth_pos) = GO * dI_dvth_p
        # dL/d(theta_neg) = dL/dI_neg * dI_neg/d(vth_neg) = -GO * dI_dvth_n
        dwc += go * dI_dvth_p
        dwd += go * (-dI_dvth_n)

    tl.store(dWc_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwc, mask=nk_mask)
    tl.store(dWd_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
             dwd, mask=nk_mask)


# ═══════════════════════ Autograd Function ═══════════════════════════

class FeFETFunction(torch.autograd.Function):
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
        A_lk      = float(params['A_lk'])
        B_lk      = float(params['B_lk'])
        V_sat = float(params['V_sat'])
        inv_2nUT  = 1.0 / (2.0 * n_fac * U_T)

        out = torch.empty(M, N, device=v_in.device, dtype=v_in.dtype)
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        _fefet_fwd_kernel[grid](
            v_in, w_com, w_diff, out,
            M, N, K,
            v_in.stride(0),   v_in.stride(1),
            w_com.stride(0),  w_com.stride(1),
            out.stride(0),    out.stride(1),
            I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
            NEWTON_ITERS=6,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        )

        ctx.save_for_backward(v_in, w_com, w_diff)
        ctx.p = (I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wc, wd = ctx.saved_tensors
        I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wc.shape[0]

        dL_dV = torch.empty_like(v)
        grid_v = (triton.cdiv(M, _BLOCK_M), triton.cdiv(K, _BLOCK_K))
        _fefet_grad_v_kernel[grid_v](
            grad_out, v, wc, wd, dL_dV,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wc.stride(0), wc.stride(1),
            dL_dV.stride(0), dL_dV.stride(1),
            I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
            NEWTON_ITERS=6,
            BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K,
        )

        dWc = torch.empty_like(wc)
        dWd = torch.empty_like(wd)
        grid_w = (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
        _fefet_grad_w_kernel[grid_w](
            grad_out, v, wc, wd, dWc, dWd,
            M, N, K,
            grad_out.stride(0), grad_out.stride(1),
            v.stride(0), v.stride(1),
            wc.stride(0), wc.stride(1),
            dWc.stride(0), dWc.stride(1),
            I_S, inv_2nUT, V_D, A_lk, B_lk, V_sat,
            NEWTON_ITERS=6,
            BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        return dL_dV, dWc, dWd, None
