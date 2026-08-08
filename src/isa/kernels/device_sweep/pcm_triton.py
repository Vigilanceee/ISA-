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
_SPLIT_BLOCK_M = 32
_SPLIT_BLOCK_N = 16
_SPLIT_BLOCK_K = 16
_FWD_SPLIT_BLOCK_M = 16
_FWD_SPLIT_BLOCK_N = 16
_FWD_SPLIT_TARGET_K = 64
_FWD_SPLIT_MAX_PARTS = 16
_FWD_SPLIT_PARTIAL_BUDGET_BYTES = 256 * 1024 * 1024

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


@triton.jit
def _pcm_fwd_split_k_kernel(
    V_ptr, Wp_ptr, Wn_ptr, Partial_ptr,
    M, N, K, SPLIT_COUNT,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_ps, stride_pm, stride_pn,
    I0, I0_DECAY, V_T,
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
        chi_p = wp[None, :]
        chi_n = wn[None, :]
        arg_p = tl.maximum(tl.minimum(chi_p * voltage / V_T, 20.0), -20.0)
        arg_n = tl.maximum(tl.minimum(chi_n * voltage / V_T, 20.0), -20.0)
        sinh_p = 0.5 * (tl.exp(arg_p) - tl.exp(-arg_p))
        sinh_n = 0.5 * (tl.exp(arg_n) - tl.exp(-arg_n))
        i0_p = I0 * tl.exp(-I0_DECAY * chi_p)
        i0_n = I0 * tl.exp(-I0_DECAY * chi_n)
        acc += i0_p * chi_p * sinh_p - i0_n * chi_n * sinh_n

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
def _pcm_fwd_split_k_reduce_kernel(
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


@triton.jit
def _pcm_grad_v_split_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dV_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dvm, stride_dvk,
    I0, I0_DECAY, V_T,
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
    v_scaled = v / V_T
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
        arg_p = tl.maximum(tl.minimum(wp * v_scaled, 20.0), -20.0)
        arg_n = tl.maximum(tl.minimum(wn * v_scaled, 20.0), -20.0)
        cosh_p = 0.5 * (tl.exp(arg_p) + tl.exp(-arg_p))
        cosh_n = 0.5 * (tl.exp(arg_n) + tl.exp(-arg_n))
        i0_p = I0 * tl.exp(-I0_DECAY * wp)
        i0_n = I0 * tl.exp(-I0_DECAY * wn)
        dpos = i0_p * (wp * wp / V_T) * cosh_p
        dneg = i0_n * (wn * wn / V_T) * cosh_n
        dv += go * (dpos - dneg)

    tl.atomic_add(
        dV_ptr + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvk,
        dv,
        mask=mk_mask,
    )


@triton.jit
def _pcm_grad_w_split_kernel(
    GO_ptr, V_ptr, Wp_ptr, Wn_ptr, dWp_ptr, dWn_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_vm, stride_vk,
    stride_wn, stride_wk,
    stride_dwn, stride_dwk,
    I0, I0_DECAY, V_T,
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
        v_scaled = v / V_T
        arg_p = tl.maximum(tl.minimum(wp * v_scaled, 20.0), -20.0)
        arg_n = tl.maximum(tl.minimum(wn * v_scaled, 20.0), -20.0)
        exp_p = tl.exp(arg_p)
        exp_mp = tl.exp(-arg_p)
        exp_n = tl.exp(arg_n)
        exp_mn = tl.exp(-arg_n)
        sinh_p = 0.5 * (exp_p - exp_mp)
        cosh_p = 0.5 * (exp_p + exp_mp)
        sinh_n = 0.5 * (exp_n - exp_mn)
        cosh_n = 0.5 * (exp_n + exp_mn)
        i0_p = I0 * tl.exp(-I0_DECAY * wp)
        i0_n = I0 * tl.exp(-I0_DECAY * wn)
        dpos = i0_p * (
            (1.0 - I0_DECAY * wp) * sinh_p
            + wp * v_scaled * cosh_p
        )
        dneg = i0_n * (
            (1.0 - I0_DECAY * wn) * sinh_n
            + wn * v_scaled * cosh_n
        )
        dwp += go * dpos
        dwn += go * (-dneg)

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
        forward_backend = str(params.get("raw_forward_backend", "split_k"))
        if forward_backend not in {"legacy", "split_k"}:
            raise ValueError(
                f"Unknown PCM raw_forward_backend={forward_backend!r}; "
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
            _pcm_fwd_kernel[grid](
                v_in, w_pos, w_neg, out,
                M, N, K,
                v_in.stride(0),  v_in.stride(1),
                w_pos.stride(0), w_pos.stride(1),
                out.stride(0),   out.stride(1),
                I0, I0_DECAY, V_T,
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
            _pcm_fwd_split_k_kernel[split_grid](
                v_in, w_pos, w_neg, partial,
                M, N, K, split_count,
                v_in.stride(0), v_in.stride(1),
                w_pos.stride(0), w_pos.stride(1),
                partial.stride(0), partial.stride(1), partial.stride(2),
                I0, I0_DECAY, V_T,
                BLOCK_M=_FWD_SPLIT_BLOCK_M,
                BLOCK_N=_FWD_SPLIT_BLOCK_N,
            )
            reduce_grid = (
                triton.cdiv(M, _FWD_SPLIT_BLOCK_M),
                triton.cdiv(N, _FWD_SPLIT_BLOCK_N),
            )
            _pcm_fwd_split_k_reduce_kernel[reduce_grid](
                partial, out,
                M, N, split_count,
                partial.stride(0), partial.stride(1), partial.stride(2),
                out.stride(0), out.stride(1),
                BLOCK_M=_FWD_SPLIT_BLOCK_M,
                BLOCK_N=_FWD_SPLIT_BLOCK_N,
            )

        ctx.save_for_backward(v_in, w_pos, w_neg)
        ctx.p = (I0, I0_DECAY, V_T)
        ctx.raw_kernel_backend = str(params.get("raw_kernel_backend", "split"))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, wp, wn = ctx.saved_tensors
        I0, I0_DECAY, V_T = ctx.p
        grad_out = grad_out.contiguous()
        M, K = v.shape
        N    = wp.shape[0]

        backend = ctx.raw_kernel_backend
        if backend not in {"legacy", "split"}:
            raise ValueError(
                f"Unknown PCM raw_kernel_backend={backend!r}; "
                "expected 'legacy' or 'split'"
            )
        if backend == "legacy":
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
        else:
            dV_acc = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            grid_v = (
                triton.cdiv(M, _SPLIT_BLOCK_M),
                triton.cdiv(K, _SPLIT_BLOCK_K),
                triton.cdiv(N, _SPLIT_BLOCK_N),
            )
            _pcm_grad_v_split_kernel[grid_v](
                grad_out, v, wp, wn, dV_acc,
                M, N, K,
                grad_out.stride(0), grad_out.stride(1),
                v.stride(0), v.stride(1),
                wp.stride(0), wp.stride(1),
                dV_acc.stride(0), dV_acc.stride(1),
                I0, I0_DECAY, V_T,
                BLOCK_M=_SPLIT_BLOCK_M,
                BLOCK_K=_SPLIT_BLOCK_K,
                BLOCK_N=_SPLIT_BLOCK_N,
            )
            dL_dV = dV_acc.to(v.dtype)
            dWp = torch.zeros(wp.shape, device=wp.device, dtype=torch.float32)
            dWn = torch.zeros(wn.shape, device=wn.device, dtype=torch.float32)
            grid_w = (
                triton.cdiv(N, _SPLIT_BLOCK_N),
                triton.cdiv(K, _SPLIT_BLOCK_K),
                triton.cdiv(M, _SPLIT_BLOCK_M),
            )
            _pcm_grad_w_split_kernel[grid_w](
                grad_out, v, wp, wn, dWp, dWn,
                M, N, K,
                grad_out.stride(0), grad_out.stride(1),
                v.stride(0), v.stride(1),
                wp.stride(0), wp.stride(1),
                dWp.stride(0), dWp.stride(1),
                I0, I0_DECAY, V_T,
                BLOCK_N=_SPLIT_BLOCK_N,
                BLOCK_K=_SPLIT_BLOCK_K,
                BLOCK_M=_SPLIT_BLOCK_M,
            )
            dWp = dWp.to(wp.dtype)
            dWn = dWn.to(wn.dtype)

        return dL_dV, dWp, dWn, None
