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
_SPLIT_BLOCK_M = 32
_SPLIT_BLOCK_N = 16
_SPLIT_BLOCK_K = 16
_FWD_SPLIT_BLOCK_M = 16
_FWD_SPLIT_BLOCK_N = 16
_FWD_SPLIT_TARGET_K = 64
_FWD_SPLIT_MAX_PARTS = 16
_FWD_SPLIT_PARTIAL_BUDGET_BYTES = 256 * 1024 * 1024


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


@triton.jit
def _flash_fwd_split_k_kernel(
    V_ptr, Wp_ptr, Wn_ptr, Partial_ptr,
    M, N, K, SPLIT_COUNT,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_ps, stride_pm, stride_pn,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Evaluate one exact contiguous K shard into an FP32 partial buffer."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    chunk = (K + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_start = pid_s * chunk
    k_end = tl.minimum(k_start + chunk, K)

    for k in range(k_start, k_end):
        v = tl.load(
            V_ptr + offs_m * stride_vm + k * stride_vk,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        wp = tl.load(
            Wp_ptr + offs_n * stride_wn + k * stride_wk,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        wn = tl.load(
            Wn_ptr + offs_n * stride_wn + k * stride_wk,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        voltage = v[:, None]
        denom = 1.0 + voltage / V_sat
        u_p = (voltage - wp[None, :]) * inv_2nUT
        u_n = (voltage - wn[None, :]) * inv_2nUT
        sp_p1 = _sp(u_p)
        sp_p2 = _sp(u_p - VD_over_2nUT)
        sp_n1 = _sp(u_n)
        sp_n2 = _sp(u_n - VD_over_2nUT)
        acc += I_S * (
            sp_p1 * sp_p1
            - sp_p2 * sp_p2
            - sp_n1 * sp_n1
            + sp_n2 * sp_n2
        ) / denom

    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Partial_ptr
        + pid_s * stride_ps
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn,
        acc,
        mask=valid,
    )


@triton.jit
def _flash_fwd_split_k_reduce_kernel(
    Partial_ptr, O_ptr,
    M, N, SPLIT_COUNT,
    stride_ps, stride_pm, stride_pn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in range(SPLIT_COUNT):
        acc += tl.load(
            Partial_ptr
            + split * stride_ps
            + offs_m[:, None] * stride_pm
            + offs_n[None, :] * stride_pn,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=valid,
    )


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


@triton.jit
def _flash_grad_v_split_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dvm, stride_dvk,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Exact dV with N split across programs and FP32 atomic reduction."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K
    mk_mask = m_mask & k_mask
    v = tl.load(
        V_ptr + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk,
        mask=mk_mask,
        other=0.0,
    ).to(tl.float32)
    inv_denom = 1.0 / (1.0 + v / V_sat)
    dv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    n_start = pid_n * BLOCK_N

    for nn in range(BLOCK_N):
        n = n_start + nn
        n_valid = n < N
        go = tl.load(
            GO_ptr + offs_m[:, None] * stride_gom + n * stride_gon,
            mask=m_mask & n_valid,
            other=0.0,
        ).to(tl.float32)
        wp = tl.load(
            Wp_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
            mask=k_mask & n_valid,
            other=0.0,
        ).to(tl.float32)
        wn = tl.load(
            Wn_ptr + n * stride_wn + offs_k[None, :] * stride_wk,
            mask=k_mask & n_valid,
            other=0.0,
        ).to(tl.float32)
        u_p = (v - wp) * inv_2nUT
        u_n = (v - wn) * inv_2nUT
        sp_p1 = _sp(u_p)
        sp_p2 = _sp(u_p - VD_over_2nUT)
        sp_n1 = _sp(u_n)
        sp_n2 = _sp(u_n - VD_over_2nUT)
        sig_p1 = _sigm(u_p)
        sig_p2 = _sigm(u_p - VD_over_2nUT)
        sig_n1 = _sigm(u_n)
        sig_n2 = _sigm(u_n - VD_over_2nUT)
        ib_p = sp_p1 * sp_p1 - sp_p2 * sp_p2
        ib_n = sp_n1 * sp_n1 - sp_n2 * sp_n2
        dib_p = 2.0 * (sp_p1 * sig_p1 - sp_p2 * sig_p2) * inv_2nUT
        dib_n = 2.0 * (sp_n1 * sig_n1 - sp_n2 * sig_n2) * inv_2nUT
        local = I_S * (
            (dib_p - dib_n) * inv_denom
            - (ib_p - ib_n) * inv_denom * inv_denom / V_sat
        )
        dv += go * local

    tl.atomic_add(
        dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk,
        dv,
        mask=mk_mask,
    )


@triton.jit
def _flash_grad_w_split_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dwn, stride_dwk,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Exact dW with M split across programs and FP32 atomic reduction."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < K
    nk_mask = n_mask & k_mask
    wp = tl.load(
        Wp_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
        mask=nk_mask,
        other=0.0,
    ).to(tl.float32)
    wn = tl.load(
        Wn_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
        mask=nk_mask,
        other=0.0,
    ).to(tl.float32)
    dwp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dwn = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    m_start = pid_m * BLOCK_M

    for mm in range(BLOCK_M):
        m = m_start + mm
        m_valid = m < M
        go = tl.load(
            GO_ptr + m * stride_gom + offs_n[:, None] * stride_gon,
            mask=n_mask & m_valid,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_ptr + m * stride_vm + offs_k[None, :] * stride_vk,
            mask=k_mask & m_valid,
            other=0.0,
        ).to(tl.float32)
        inv_denom = 1.0 / (1.0 + v / V_sat)
        u_p = (v - wp) * inv_2nUT
        u_n = (v - wn) * inv_2nUT
        sp_p1 = _sp(u_p)
        sp_p2 = _sp(u_p - VD_over_2nUT)
        sp_n1 = _sp(u_n)
        sp_n2 = _sp(u_n - VD_over_2nUT)
        sig_p1 = _sigm(u_p)
        sig_p2 = _sigm(u_p - VD_over_2nUT)
        sig_n1 = _sigm(u_n)
        sig_n2 = _sigm(u_n - VD_over_2nUT)
        dib_p = 2.0 * (sp_p1 * sig_p1 - sp_p2 * sig_p2) * inv_2nUT
        dib_n = 2.0 * (sp_n1 * sig_n1 - sp_n2 * sig_n2) * inv_2nUT
        dwp += go * (-I_S * dib_p * inv_denom)
        dwn += go * (I_S * dib_n * inv_denom)

    tl.atomic_add(
        dWp_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
        dwp,
        mask=nk_mask,
    )
    tl.atomic_add(
        dWn_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
        dwn,
        mask=nk_mask,
    )


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
        forward_backend = str(params.get("raw_forward_backend", "split_k"))
        if forward_backend not in {"legacy", "split_k"}:
            raise ValueError(
                f"Unknown Flash raw_forward_backend={forward_backend!r}; "
                "expected 'legacy' or 'split_k'"
            )
        output_elements = max(M * N, 1)
        maximum_parts_by_memory = max(
            1,
            _FWD_SPLIT_PARTIAL_BUDGET_BYTES // (output_elements * 4),
        )
        split_count = min(
            _FWD_SPLIT_MAX_PARTS,
            triton.cdiv(K, _FWD_SPLIT_TARGET_K),
            maximum_parts_by_memory,
        )
        if forward_backend == "legacy" or split_count <= 1:
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
        else:
            partial = torch.empty(
                (split_count, M, N),
                device=v_in.device,
                dtype=torch.float32,
            )
            split_grid = (
                triton.cdiv(M, _FWD_SPLIT_BLOCK_M),
                triton.cdiv(N, _FWD_SPLIT_BLOCK_N),
                split_count,
            )
            _flash_fwd_split_k_kernel[split_grid](
                v_in, w_com, w_diff, partial,
                M, N, K, split_count,
                v_in.stride(0), v_in.stride(1),
                w_com.stride(0), w_com.stride(1),
                partial.stride(0), partial.stride(1), partial.stride(2),
                I_S, inv_2nUT, VD_over_2nUT, V_sat,
                BLOCK_M=_FWD_SPLIT_BLOCK_M,
                BLOCK_N=_FWD_SPLIT_BLOCK_N,
            )
            reduce_grid = (
                triton.cdiv(M, _FWD_SPLIT_BLOCK_M),
                triton.cdiv(N, _FWD_SPLIT_BLOCK_N),
            )
            _flash_fwd_split_k_reduce_kernel[reduce_grid](
                partial, out,
                M, N, split_count,
                partial.stride(0), partial.stride(1), partial.stride(2),
                out.stride(0), out.stride(1),
                BLOCK_M=_FWD_SPLIT_BLOCK_M,
                BLOCK_N=_FWD_SPLIT_BLOCK_N,
            )

        ctx.save_for_backward(v_in, w_com, w_diff)
        ctx.p = (I_S, inv_2nUT, VD_over_2nUT, V_sat)
        ctx.raw_kernel_backend = str(params.get("raw_kernel_backend", "split"))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wc, wd = ctx.saved_tensors
        I_S, inv_2nUT, VD_over_2nUT, V_sat = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wc.shape[0]

        backend = ctx.raw_kernel_backend
        if backend not in {"legacy", "split"}:
            raise ValueError(
                f"Unknown Flash raw_kernel_backend={backend!r}; "
                "expected 'legacy' or 'split'"
            )
        if backend == "legacy":
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
        else:
            dV_acc = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            grid_v = (
                triton.cdiv(M, _SPLIT_BLOCK_M),
                triton.cdiv(K, _SPLIT_BLOCK_K),
                triton.cdiv(N, _SPLIT_BLOCK_N),
            )
            _flash_grad_v_split_kernel[grid_v](
                grad_out, v, wc, wd, dV_acc,
                M, N, K,
                grad_out.stride(0), grad_out.stride(1),
                v.stride(0), v.stride(1),
                wc.stride(0), wc.stride(1),
                dV_acc.stride(0), dV_acc.stride(1),
                I_S, inv_2nUT, VD_over_2nUT, V_sat,
                BLOCK_M=_SPLIT_BLOCK_M,
                BLOCK_K=_SPLIT_BLOCK_K,
                BLOCK_N=_SPLIT_BLOCK_N,
            )
            dL_dV = dV_acc.to(v.dtype)
            dWc = torch.zeros(wc.shape, device=wc.device, dtype=torch.float32)
            dWd = torch.zeros(wd.shape, device=wd.device, dtype=torch.float32)
            grid_w = (
                triton.cdiv(N, _SPLIT_BLOCK_N),
                triton.cdiv(K, _SPLIT_BLOCK_K),
                triton.cdiv(M, _SPLIT_BLOCK_M),
            )
            _flash_grad_w_split_kernel[grid_w](
                grad_out, v, wc, wd, dWc, dWd,
                M, N, K,
                grad_out.stride(0), grad_out.stride(1),
                v.stride(0), v.stride(1),
                wc.stride(0), wc.stride(1),
                dWc.stride(0), dWc.stride(1),
                I_S, inv_2nUT, VD_over_2nUT, V_sat,
                BLOCK_N=_SPLIT_BLOCK_N,
                BLOCK_K=_SPLIT_BLOCK_K,
                BLOCK_M=_SPLIT_BLOCK_M,
            )
            dWc = dWc.to(wc.dtype)
            dWd = dWd.to(wd.dtype)

        return dL_dV, dWc, dWd, None


class FlashSplitFunction(torch.autograd.Function):
    """Exact Flash EKV operator using the profiled split-reduction kernels.

    The wrapper selects the specialized Flash split-K forward and
    split-reduction backward without changing the EKV response.
    """

    @staticmethod
    def forward(ctx, v_in, w_pos, w_neg, params):
        exact_params = dict(params)
        exact_params.update(
            {"raw_kernel_backend": "split", "raw_forward_backend": "split_k"}
        )
        return FlashFunction.forward(ctx, v_in, w_pos, w_neg, exact_params)

    @staticmethod
    def backward(ctx, grad_out):
        return FlashFunction.backward(ctx, grad_out)
