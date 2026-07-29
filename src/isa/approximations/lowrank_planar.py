"""Tensor-core compression of the physical node-planar backend.

The existing node-planar backend evaluates the physical current at fixed
state nodes and expands every input channel by ``N_NODES`` before calling
cuDNN.  A differential state weight uses only two adjacent nodes per branch,
so the dense convolution spends most of its work multiplying structural
zeros.

This module keeps the same state-axis piecewise-linear representation, but
projects the physical node curves onto a small, device-specific orthogonal
basis:

    T(V, node) - mean_node(T) ~= F(V, rank) @ Q(node, rank).T

The node-independent mean cancels exactly between the positive and negative
branches.  The projected activation curves and their input derivatives are
stored as one-dimensional LUTs.  State coefficients are obtained by linearly
interpolating rows of Q, so state gradients remain analytical.  cuDNN/cuBLAS
then sees ``rank`` expanded channels instead of ``N_NODES`` channels.

The basis is generated from the configured physical LUT at first use.  No
polynomial model or learned surrogate is introduced.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from isa.kernels.device_sweep.lut_triton import _make_fefet_tables, _make_flash_tables


_BLOCK = int(os.environ.get("NVM_LOWRANK_BLOCK_SIZE", "512"))
if _BLOCK not in {64, 128, 256, 512, 1024}:
    raise ValueError(
        "NVM_LOWRANK_BLOCK_SIZE must be one of 64, 128, 256, 512, 1024"
    )
_table_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _autocast_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.is_cuda and torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    return tensor.dtype


@triton.jit
def _interp_rank_table(
    table,
    rank,
    x,
    x_min: tl.constexpr,
    inv_dx: tl.constexpr,
    table_size: tl.constexpr,
):
    coord = (x - x_min) * inv_dx
    coord = tl.maximum(0.0, tl.minimum(coord, table_size - 1.001))
    left = coord.to(tl.int32)
    frac = coord - left
    base = rank * table_size + left
    y0 = tl.load(table + base)
    y1 = tl.load(table + base + 1)
    return y0 + frac * (y1 - y0)


@triton.jit
def _transform_fwd_kernel(
    X,
    TABLE,
    OUT,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sx_b,
    sx_c,
    sx_h,
    sx_w,
    so_b,
    so_c,
    so_h,
    so_w,
    x_min: tl.constexpr,
    inv_dx: tl.constexpr,
    table_size: tl.constexpr,
    RANK: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = H * W
    inner = C * spatial
    total = B * RANK * inner
    mask = offsets < total
    if CHANNELS_LAST:
        out_channels = RANK * C
        out_c = offsets % out_channels
        spatial_offset = (offsets // out_channels) % spatial
        b = offsets // (out_channels * spatial)
        rank = out_c // C
        c = out_c % C
        h = spatial_offset // W
        w = spatial_offset % W
    else:
        b = offsets // (RANK * inner)
        rem = offsets % (RANK * inner)
        rank = rem // inner
        q = rem % inner
        c = q // spatial
        hw = q % spatial
        h = hw // W
        w = hw % W

    x = tl.load(
        X + b * sx_b + c * sx_c + h * sx_h + w * sx_w,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = _interp_rank_table(
        TABLE, rank, x, x_min, inv_dx, table_size
    )
    out_c = rank * C + c
    tl.store(
        OUT + b * so_b + out_c * so_c + h * so_h + w * so_w,
        value,
        mask=mask,
    )


@triton.jit
def _transform_bwd_kernel(
    GO,
    X,
    D_TABLE,
    GX,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sgo_b,
    sgo_c,
    sgo_h,
    sgo_w,
    sx_b,
    sx_c,
    sx_h,
    sx_w,
    sgx_b,
    sgx_c,
    sgx_h,
    sgx_w,
    x_min: tl.constexpr,
    inv_dx: tl.constexpr,
    table_size: tl.constexpr,
    RANK: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = H * W
    inner = C * spatial
    total = B * inner
    mask = offsets < total
    if CHANNELS_LAST:
        c = offsets % C
        spatial_offset = (offsets // C) % spatial
        b = offsets // (C * spatial)
        h = spatial_offset // W
        w = spatial_offset % W
    else:
        b = offsets // inner
        q = offsets % inner
        c = q // spatial
        hw = q % spatial
        h = hw // W
        w = hw % W
    x = tl.load(
        X + b * sx_b + c * sx_c + h * sx_h + w * sx_w,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad_x = tl.zeros((BLOCK,), dtype=tl.float32)

    for rank in range(RANK):
        grad_out = tl.load(
            GO
            + b * sgo_b
            + (rank * C + c) * sgo_c
            + h * sgo_h
            + w * sgo_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        local = _interp_rank_table(
            D_TABLE, rank, x, x_min, inv_dx, table_size
        )
        grad_x += grad_out * local

    tl.store(
        GX + b * sgx_b + c * sgx_c + h * sgx_h + w * sgx_w,
        grad_x,
        mask=mask,
    )


@triton.jit
def _basis_fwd_kernel(
    THETA_POS,
    THETA_NEG,
    Q,
    OUT,
    E,
    K: tl.constexpr,
    so_n,
    so_c,
    so_h,
    so_w,
    lo,
    inv_step,
    N_NODES: tl.constexpr,
    RANK: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CONV_LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = E * RANK
    mask = offsets < total
    k = offsets % K
    rank = (offsets // K) % RANK
    n = offsets // (RANK * K)
    element = n * K + k

    pos = tl.load(THETA_POS + element, mask=mask, other=lo).to(tl.float32)
    neg = tl.load(THETA_NEG + element, mask=mask, other=lo).to(tl.float32)
    pos_coord = (pos - lo) * inv_step
    neg_coord = (neg - lo) * inv_step
    pos_left = tl.maximum(
        0, tl.minimum(pos_coord.to(tl.int32), N_NODES - 2)
    )
    neg_left = tl.maximum(
        0, tl.minimum(neg_coord.to(tl.int32), N_NODES - 2)
    )
    pos_frac = tl.maximum(0.0, tl.minimum(pos_coord - pos_left, 1.0))
    neg_frac = tl.maximum(0.0, tl.minimum(neg_coord - neg_left, 1.0))

    q_pos_left = tl.load(
        Q + pos_left * RANK + rank, mask=mask, other=0.0
    ).to(tl.float32)
    q_pos_right = tl.load(
        Q + (pos_left + 1) * RANK + rank, mask=mask, other=0.0
    ).to(tl.float32)
    q_neg_left = tl.load(
        Q + neg_left * RANK + rank, mask=mask, other=0.0
    ).to(tl.float32)
    q_neg_right = tl.load(
        Q + (neg_left + 1) * RANK + rank, mask=mask, other=0.0
    ).to(tl.float32)
    value = (
        q_pos_left + pos_frac * (q_pos_right - q_pos_left)
        - q_neg_left - neg_frac * (q_neg_right - q_neg_left)
    )
    if CONV_LAYOUT:
        kernel_plane = KH * KW
        in_channel = k // kernel_plane
        kernel_offset = k % kernel_plane
        kernel_h = kernel_offset // KW
        kernel_w = kernel_offset % KW
        out_channel = rank * IN_CHANNELS + in_channel
        out_offset = (
            n * so_n
            + out_channel * so_c
            + kernel_h * so_h
            + kernel_w * so_w
        )
    else:
        out_offset = offsets
    tl.store(
        OUT + out_offset,
        value,
        mask=mask,
    )


@triton.jit
def _basis_bwd_kernel(
    GO,
    THETA_POS,
    THETA_NEG,
    Q,
    GPOS,
    GNEG,
    E,
    K: tl.constexpr,
    sgo_n,
    sgo_c,
    sgo_h,
    sgo_w,
    lo,
    inv_step,
    N_NODES: tl.constexpr,
    RANK: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CONV_LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E
    n = offsets // K
    k = offsets % K
    pos = tl.load(THETA_POS + offsets, mask=mask, other=lo).to(tl.float32)
    neg = tl.load(THETA_NEG + offsets, mask=mask, other=lo).to(tl.float32)
    pos_coord = (pos - lo) * inv_step
    neg_coord = (neg - lo) * inv_step
    pos_left = tl.maximum(
        0, tl.minimum(pos_coord.to(tl.int32), N_NODES - 2)
    )
    neg_left = tl.maximum(
        0, tl.minimum(neg_coord.to(tl.int32), N_NODES - 2)
    )
    grad_pos = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_neg = tl.zeros((BLOCK,), dtype=tl.float32)

    for rank in range(RANK):
        if CONV_LAYOUT:
            kernel_plane = KH * KW
            in_channel = k // kernel_plane
            kernel_offset = k % kernel_plane
            kernel_h = kernel_offset // KW
            kernel_w = kernel_offset % KW
            out_channel = rank * IN_CHANNELS + in_channel
            grad_out_offset = (
                n * sgo_n
                + out_channel * sgo_c
                + kernel_h * sgo_h
                + kernel_w * sgo_w
            )
        else:
            grad_out_offset = n * (RANK * K) + rank * K + k
        grad_out = tl.load(
            GO + grad_out_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        pos_left_q = tl.load(
            Q + pos_left * RANK + rank, mask=mask, other=0.0
        ).to(tl.float32)
        pos_right_q = tl.load(
            Q + (pos_left + 1) * RANK + rank, mask=mask, other=0.0
        ).to(tl.float32)
        neg_left_q = tl.load(
            Q + neg_left * RANK + rank, mask=mask, other=0.0
        ).to(tl.float32)
        neg_right_q = tl.load(
            Q + (neg_left + 1) * RANK + rank, mask=mask, other=0.0
        ).to(tl.float32)
        grad_pos += grad_out * (pos_right_q - pos_left_q) * inv_step
        grad_neg -= grad_out * (neg_right_q - neg_left_q) * inv_step

    tl.store(GPOS + offsets, grad_pos, mask=mask)
    tl.store(GNEG + offsets, grad_neg, mask=mask)


def _torch_interp1(
    table: torch.Tensor,
    values: torch.Tensor,
    x_min: float,
    inv_dx: float,
) -> torch.Tensor:
    coord = ((values - x_min) * inv_dx).clamp(0.0, table.numel() - 1.001)
    left = coord.floor().long()
    frac = coord - left
    return table[left] + frac * (table[left + 1] - table[left])


def _sample_physical_nodes(
    device_type: str,
    params: dict,
    nodes: torch.Tensor,
    voltage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = voltage[:, None].float()
    w = nodes[None, :].float()
    if device_type == "pcm":
        v_t = float(params["V_T_pcm"])
        coeff = (
            float(params["I0_pcm"])
            * torch.exp(-float(params["I0_pcm_decay"]) * w)
            * w
        )
        argument = (w * v / v_t).clamp(-20.0, 20.0)
        current = coeff * torch.sinh(argument)
        derivative = coeff * (w / v_t) * torch.cosh(argument)
        return current, derivative

    if device_type == "flash":
        table, dtable, meta = _make_flash_tables(
            params, voltage.device, torch.float32
        )
    elif device_type == "fefet":
        table, dtable, meta = _make_fefet_tables(
            params, voltage.device, torch.float32
        )
    else:
        raise ValueError(
            f"low-rank planar backend is not defined for {device_type!r}"
        )

    raw = _torch_interp1(table, v - w, meta[0], meta[1])
    draw = _torch_interp1(dtable, v - w, meta[0], meta[1])
    i_s = float(meta[3])
    v_sat = float(meta[4])
    denominator = 1.0 + v / v_sat
    inv_denominator = denominator.reciprocal()
    current = i_s * raw * inv_denominator
    derivative = i_s * (
        draw * inv_denominator
        - raw * inv_denominator.square() / v_sat
    )
    return current, derivative


def _table_domain(device_type: str, params: dict) -> tuple[float, float]:
    if device_type == "pcm":
        return (
            float(params.get("lut_v_min", -0.5)),
            float(params.get("lut_v_max", 0.5)),
        )
    return (
        float(params.get("lowrank_v_min", -4.0)),
        float(params.get("lowrank_v_max", 4.0)),
    )


def _cache_key(
    device_type: str,
    params: dict,
    nodes: torch.Tensor,
    rank: int,
    table_size: int,
    v_min: float,
    v_max: float,
) -> tuple:
    physical_names = {
        "pcm": ("I0_pcm", "I0_pcm_decay", "V_T_pcm"),
        "flash": ("I_S", "n", "U_T", "V_D", "V_sat"),
        "fefet": (
            "I_S",
            "n",
            "U_T",
            "V_D",
            "A_lk",
            "B_lk",
            "V_sat",
        ),
    }[device_type]
    physical_params = tuple(
        (name, float(params[name])) for name in physical_names
    )
    lut_params = (
        int(params.get("lut_size", 4096)),
        float(params.get("lut_x_min", -5.0)),
        float(params.get("lut_x_max", 4.0)),
        bool(params.get("surrogate_backward_enabled", False)),
        float(params.get("surrogate_backward_alpha", 0.0)),
        float(params.get("surrogate_A_lk", params.get("A_lk", 0.0))),
        float(params.get("surrogate_B_lk", params.get("B_lk", 0.0))),
    )
    return (
        device_type,
        str(nodes.device),
        rank,
        table_size,
        v_min,
        v_max,
        nodes.numel(),
        float(params["w_min"]),
        float(params["w_max"]),
        physical_params,
        lut_params,
    )


@torch.no_grad()
def _make_lowrank_tables(
    device_type: str,
    params: dict,
    nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, int]]:
    rank = int(params.get("lowrank_rank", 8))
    if rank < 1 or rank >= nodes.numel():
        raise ValueError(
            f"lowrank_rank must be in [1, {nodes.numel() - 1}], got {rank}"
        )
    table_size = int(params.get("lowrank_lut_size", 8192))
    if table_size < 256:
        raise ValueError("lowrank_lut_size must be at least 256")
    v_min, v_max = _table_domain(device_type, params)
    key = _cache_key(
        device_type, params, nodes, rank, table_size, v_min, v_max
    )
    cached = _table_cache.get(key)
    if cached is not None:
        return cached

    voltage = torch.linspace(
        v_min,
        v_max,
        table_size,
        device=nodes.device,
        dtype=torch.float32,
    )
    curves, derivatives = _sample_physical_nodes(
        device_type, params, nodes, voltage
    )
    centered_curves = curves - curves.mean(dim=1, keepdim=True)
    centered_derivatives = (
        derivatives - derivatives.mean(dim=1, keepdim=True)
    )
    _u, _s, vh = torch.linalg.svd(centered_curves, full_matrices=False)
    q = vh[:rank].transpose(0, 1).contiguous()
    features = (centered_curves @ q).transpose(0, 1).contiguous()
    derivative_features = (
        centered_derivatives @ q
    ).transpose(0, 1).contiguous()
    meta = (v_min, (table_size - 1) / (v_max - v_min), table_size)
    _table_cache[key] = (
        features,
        derivative_features,
        q,
        meta,
    )
    return _table_cache[key]


class _LowRankTransform(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        nodes,
        device_type: str,
        params: dict,
        channels_last: bool,
    ):
        if x.ndim != 4:
            raise ValueError(
                f"low-rank transform expects NCHW input, got {tuple(x.shape)}"
            )
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        nodes = nodes.contiguous()
        features, derivative_features, q, meta = _make_lowrank_tables(
            device_type, params, nodes
        )
        rank = q.shape[1]
        b, c, h, w = x.shape
        shape = (b, rank * c, h, w)
        output_dtype = _autocast_dtype(x)
        if channels_last:
            out = torch.empty(
                shape,
                device=x.device,
                dtype=output_dtype,
                memory_format=torch.channels_last,
            )
        else:
            out = torch.empty(shape, device=x.device, dtype=output_dtype)
        total = b * rank * c * h * w
        _transform_fwd_kernel[(triton.cdiv(total, _BLOCK),)](
            x,
            features,
            out,
            b,
            c,
            h,
            w,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            meta[0],
            meta[1],
            meta[2],
            RANK=rank,
            CHANNELS_LAST=channels_last,
            BLOCK=_BLOCK,
        )
        ctx.save_for_backward(x, derivative_features)
        ctx.meta = (meta, rank, channels_last)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, derivative_features = ctx.saved_tensors
        meta, rank, channels_last = ctx.meta
        if channels_last:
            grad_out = grad_out.contiguous(memory_format=torch.channels_last)
        else:
            grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        b, c, h, w = x.shape
        total = b * c * h * w
        _transform_bwd_kernel[(triton.cdiv(total, _BLOCK),)](
            grad_out,
            x,
            derivative_features,
            grad_x,
            b,
            c,
            h,
            w,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            grad_x.stride(3),
            meta[0],
            meta[1],
            meta[2],
            RANK=rank,
            CHANNELS_LAST=channels_last,
            BLOCK=_BLOCK,
        )
        return grad_x, None, None, None, None


class _LowRankDifferentialBasis(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        theta_pos,
        theta_neg,
        q,
        lo: float,
        inv_step: float,
        node_count: int,
        conv_shape: tuple[int, int, int] | None,
    ):
        theta_pos = theta_pos.contiguous()
        theta_neg = theta_neg.contiguous()
        q = q.contiguous()
        n, k = theta_pos.shape
        rank = q.shape[1]
        if conv_shape is None:
            out = torch.empty(
                n,
                rank,
                k,
                device=theta_pos.device,
                dtype=_autocast_dtype(theta_pos),
            )
            in_channels, kh, kw = 1, 1, 1
            conv_layout = False
        else:
            in_channels, kh, kw = conv_shape
            if k != in_channels * kh * kw:
                raise ValueError("state tensor and convolution geometry disagree")
            out = torch.empty(
                (n, rank * in_channels, kh, kw),
                device=theta_pos.device,
                dtype=_autocast_dtype(theta_pos),
                memory_format=torch.channels_last,
            )
            conv_layout = True
        elements = theta_pos.numel()
        total = elements * rank
        _basis_fwd_kernel[(triton.cdiv(total, _BLOCK),)](
            theta_pos,
            theta_neg,
            q,
            out,
            elements,
            k,
            out.stride(0),
            out.stride(1),
            out.stride(2) if out.ndim == 4 else 0,
            out.stride(3) if out.ndim == 4 else 0,
            lo,
            inv_step,
            N_NODES=node_count,
            RANK=rank,
            IN_CHANNELS=in_channels,
            KH=kh,
            KW=kw,
            CONV_LAYOUT=conv_layout,
            BLOCK=_BLOCK,
        )
        ctx.save_for_backward(theta_pos, theta_neg, q)
        ctx.meta = (
            lo,
            inv_step,
            node_count,
            rank,
            k,
            conv_shape,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        theta_pos, theta_neg, q = ctx.saved_tensors
        lo, inv_step, node_count, rank, k, conv_shape = ctx.meta
        if conv_shape is None:
            grad_out = grad_out.contiguous()
            in_channels, kh, kw = 1, 1, 1
            conv_layout = False
        else:
            grad_out = grad_out.contiguous(memory_format=torch.channels_last)
            in_channels, kh, kw = conv_shape
            conv_layout = True
        grad_pos = torch.empty_like(theta_pos)
        grad_neg = torch.empty_like(theta_neg)
        elements = theta_pos.numel()
        _basis_bwd_kernel[(triton.cdiv(elements, _BLOCK),)](
            grad_out,
            theta_pos,
            theta_neg,
            q,
            grad_pos,
            grad_neg,
            elements,
            k,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2) if grad_out.ndim == 4 else 0,
            grad_out.stride(3) if grad_out.ndim == 4 else 0,
            lo,
            inv_step,
            N_NODES=node_count,
            RANK=rank,
            IN_CHANNELS=in_channels,
            KH=kh,
            KW=kw,
            CONV_LAYOUT=conv_layout,
            BLOCK=_BLOCK,
        )
        return grad_pos, grad_neg, None, None, None, None, None


def _lowrank_basis(
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    nodes: torch.Tensor,
    q: torch.Tensor,
    params: dict,
    conv_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    node_count = nodes.numel()
    lo = float(params["w_min"])
    inv_step = (node_count - 1) / (
        float(params["w_max"]) - float(params["w_min"])
    )
    return _LowRankDifferentialBasis.apply(
        theta_pos,
        theta_neg,
        q,
        lo,
        inv_step,
        node_count,
        conv_shape,
    )


def lowrank_planar_conv2d(
    x: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    nodes: torch.Tensor,
    params: dict,
    device_type: str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    channels_last = bool(params.get("planar_channels_last", True))
    transformed = _LowRankTransform.apply(
        x, nodes, device_type, params, channels_last
    )
    _features, _derivatives, q, _meta = _make_lowrank_tables(
        device_type, params, nodes
    )
    in_channels = x.shape[1]
    kh, kw = kernel_size
    weight = _lowrank_basis(
        theta_pos,
        theta_neg,
        nodes,
        q,
        params,
        conv_shape=(in_channels, kh, kw),
    )
    return F.conv2d(transformed, weight, stride=stride, padding=padding)


def lowrank_planar_linear(
    x: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    nodes: torch.Tensor,
    params: dict,
    device_type: str,
) -> torch.Tensor:
    x4 = x.reshape(x.shape[0], x.shape[1], 1, 1)
    transformed = _LowRankTransform.apply(
        x4, nodes, device_type, params, False
    ).flatten(1)
    _features, _derivatives, q, _meta = _make_lowrank_tables(
        device_type, params, nodes
    )
    basis = _lowrank_basis(theta_pos, theta_neg, nodes, q, params)
    weight = basis.reshape(theta_pos.shape[0], -1)
    return F.linear(transformed, weight)
