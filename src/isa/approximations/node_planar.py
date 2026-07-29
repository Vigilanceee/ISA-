"""State-axis LUT/planar acceleration for non-separable NVM operators.

For a branch current I(V, theta), linearly interpolate theta on fixed physical
state nodes:

    I(V, theta) ~= sum_j basis_j(theta) * I(V, node_j)

The differential operator then becomes a standard convolution/linear layer:

    sum_j I(V, node_j) * (basis_j(theta_pos) - basis_j(theta_neg))

The node curves are evaluated from the device LUT.  A fused Triton kernel
builds the transformed input and supplies its exact piecewise-LUT input
gradient; cuDNN/cuBLAS handles the expensive convolution/GEMM and all state
gradients flow through the piecewise-linear basis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from isa.kernels.device_sweep.lut_triton import (
    _interp1,
    _interp2,
    _make_fefet_tables,
    _make_flash_tables,
)


_BLOCK = 256


def _autocast_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.is_cuda and torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    return tensor.dtype


@triton.jit
def _basis_fwd_kernel(
    THETA_POS, THETA_NEG, OUT,
    E, K: tl.constexpr,
    lo, inv_step,
    N_NODES: tl.constexpr,
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
    pos_frac = tl.maximum(0.0, tl.minimum(pos_coord - pos_left, 1.0))
    neg_frac = tl.maximum(0.0, tl.minimum(neg_coord - neg_left, 1.0))

    for node_idx in range(N_NODES):
        value = tl.zeros((BLOCK,), dtype=tl.float32)
        value += tl.where(node_idx == pos_left, 1.0 - pos_frac, 0.0)
        value += tl.where(node_idx == pos_left + 1, pos_frac, 0.0)
        value -= tl.where(node_idx == neg_left, 1.0 - neg_frac, 0.0)
        value -= tl.where(node_idx == neg_left + 1, neg_frac, 0.0)
        tl.store(
            OUT + n * (N_NODES * K) + node_idx * K + k,
            value,
            mask=mask,
        )


@triton.jit
def _basis_bwd_kernel(
    GO, THETA_POS, THETA_NEG, GPOS, GNEG,
    E, K: tl.constexpr,
    lo, inv_step,
    N_NODES: tl.constexpr,
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

    pos_go_left = tl.load(
        GO + n * (N_NODES * K) + pos_left * K + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    pos_go_right = tl.load(
        GO + n * (N_NODES * K) + (pos_left + 1) * K + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    neg_go_left = tl.load(
        GO + n * (N_NODES * K) + neg_left * K + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    neg_go_right = tl.load(
        GO + n * (N_NODES * K) + (neg_left + 1) * K + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    grad_pos = (pos_go_right - pos_go_left) * inv_step
    grad_neg = (neg_go_left - neg_go_right) * inv_step
    tl.store(GPOS + offsets, grad_pos, mask=mask)
    tl.store(GNEG + offsets, grad_neg, mask=mask)


@triton.jit
def _node_transform_fwd_kernel(
    X, NODES, COEFF, Y, O,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    sx_b, sx_c, sx_h, sx_w,
    so_b, so_c, so_h, so_w,
    x_min: tl.constexpr, inv_dx: tl.constexpr, nx: tl.constexpr,
    w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr,
    I_S, V_sat,
    N_NODES: tl.constexpr, MODE: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = H * W
    inner = C * spatial
    total = B * N_NODES * inner
    mask = offsets < total

    if CHANNELS_LAST:
        out_channels = N_NODES * C
        out_c = offsets % out_channels
        spatial_offset = (offsets // out_channels) % spatial
        b = offsets // (out_channels * spatial)
        node_idx = out_c // C
        c = out_c % C
        h = spatial_offset // W
        w = spatial_offset % W
    else:
        b = offsets // (N_NODES * inner)
        rem = offsets % (N_NODES * inner)
        node_idx = rem // inner
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
    node = tl.load(NODES + node_idx, mask=mask, other=0.0).to(tl.float32)

    if MODE == 1:
        raw = _interp1(Y, x - node, x_min, inv_dx, nx)
        value = I_S * raw / (1.0 + x / V_sat)
    elif MODE == 3:
        coeff = tl.load(COEFF + node_idx, mask=mask, other=0.0).to(tl.float32)
        value = coeff * _interp1(Y, x * node / V_sat, x_min, inv_dx, nx)
    else:
        value = _interp2(
            Y, x, node,
            x_min, inv_dx, nx,
            w_min, inv_dw, nw,
        )

    out_c = node_idx * C + c
    tl.store(
        O + b * so_b + out_c * so_c + h * so_h + w * so_w,
        value,
        mask=mask,
    )


@triton.jit
def _node_transform_bwd_kernel(
    GO, X, NODES, COEFF, Y, DY, GX,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    sgo_b, sgo_c, sgo_h, sgo_w,
    sx_b, sx_c, sx_h, sx_w,
    sgx_b, sgx_c, sgx_h, sgx_w,
    x_min: tl.constexpr, inv_dx: tl.constexpr, nx: tl.constexpr,
    w_min: tl.constexpr, inv_dw: tl.constexpr, nw: tl.constexpr,
    I_S, V_sat,
    N_NODES: tl.constexpr, MODE: tl.constexpr,
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

    for node_idx in range(N_NODES):
        node = tl.load(NODES + node_idx).to(tl.float32)
        grad_out = tl.load(
            GO + b * sgo_b + (node_idx * C + c) * sgo_c + h * sgo_h + w * sgo_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if MODE == 1:
            raw = _interp1(Y, x - node, x_min, inv_dx, nx)
            draw = _interp1(DY, x - node, x_min, inv_dx, nx)
            denom = 1.0 + x / V_sat
            inv_denom = 1.0 / denom
            local = I_S * (
                draw * inv_denom
                - raw * inv_denom * inv_denom / V_sat
            )
        elif MODE == 3:
            coeff = tl.load(COEFF + node_idx).to(tl.float32)
            local = (
                coeff
                * node
                / V_sat
                * _interp1(DY, x * node / V_sat, x_min, inv_dx, nx)
            )
        else:
            local = _interp2(
                DY, x, node,
                x_min, inv_dx, nx,
                w_min, inv_dw, nw,
            )
        grad_x += grad_out * local

    tl.store(
        GX + b * sgx_b + c * sgx_c + h * sgx_h + w * sgx_w,
        grad_x,
        mask=mask,
    )


_pcm_sinh_cache = {}


def _make_pcm_sinh_tables(params: dict, nodes: torch.Tensor):
    size = int(params.get("pcm_lut_z_size", 4096))
    v_t = float(params["V_T_pcm"])
    v_abs = max(
        abs(float(params.get("lut_v_min", -0.5))),
        abs(float(params.get("lut_v_max", 0.5))),
    )
    w_abs = max(abs(float(params["w_min"])), abs(float(params["w_max"])))
    z_max = max(1e-6, v_abs * w_abs / v_t)
    key = (
        str(nodes.device),
        size,
        z_max,
        float(params["I0_pcm"]),
        float(params["I0_pcm_decay"]),
        v_t,
        nodes.numel(),
        float(params["w_min"]),
        float(params["w_max"]),
    )
    cached = _pcm_sinh_cache.get(key)
    if cached is not None:
        return cached

    z = torch.linspace(
        -z_max, z_max, size, device=nodes.device, dtype=torch.float32
    )
    y = torch.sinh(z).contiguous()
    dy = torch.cosh(z).contiguous()
    coeff = (
        float(params["I0_pcm"])
        * torch.exp(-float(params["I0_pcm_decay"]) * nodes.float())
        * nodes.float()
    ).contiguous()
    meta = (-z_max, (size - 1) / (2.0 * z_max), size, 0.0, 1.0, 2)
    _pcm_sinh_cache[key] = (y, dy, coeff, meta)
    return _pcm_sinh_cache[key]


def _device_tables(
    device_type: str,
    params: dict,
    device,
    dtype,
    nodes: torch.Tensor,
):
    if device_type == "flash":
        y, dy, meta = _make_flash_tables(params, device, dtype)
        return (
            y,
            dy,
            nodes,
            (meta[0], meta[1], meta[2], 0.0, 1.0, 2),
            meta[3],
            meta[4],
            1,
        )
    if device_type == "fefet":
        y, dy, meta = _make_fefet_tables(params, device, dtype)
        return (
            y,
            dy,
            nodes,
            (meta[0], meta[1], meta[2], 0.0, 1.0, 2),
            meta[3],
            meta[4],
            1,
        )
    if device_type == "pcm":
        y, dy, coeff, meta = _make_pcm_sinh_tables(params, nodes)
        return y, dy, coeff, meta, 1.0, float(params["V_T_pcm"]), 3
    raise ValueError(f"node-planar backend is not defined for {device_type!r}")


class _NodeTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nodes, device_type: str, params: dict, channels_last: bool):
        if x.ndim != 4:
            raise ValueError(f"node transform expects NCHW input, got shape={tuple(x.shape)}")
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        nodes = nodes.contiguous()
        y, dy, coeff, meta, i_s, v_sat, mode = _device_tables(
            device_type, params, x.device, x.dtype, nodes
        )
        b, c, h, w = x.shape
        shape = (b, nodes.numel() * c, h, w)
        output_dtype = _autocast_dtype(x)
        if channels_last:
            out = torch.empty(
                shape, device=x.device, dtype=output_dtype,
                memory_format=torch.channels_last,
            )
        else:
            out = torch.empty(shape, device=x.device, dtype=output_dtype)

        total = b * nodes.numel() * c * h * w
        _node_transform_fwd_kernel[(triton.cdiv(total, _BLOCK),)](
            x, nodes, coeff, y, out,
            b, c, h, w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
            i_s, v_sat,
            N_NODES=nodes.numel(), MODE=mode,
            CHANNELS_LAST=channels_last, BLOCK=_BLOCK,
        )

        ctx.save_for_backward(x, nodes, coeff, y, dy)
        ctx.meta = (meta, i_s, v_sat, mode, channels_last)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, nodes, coeff, y, dy = ctx.saved_tensors
        meta, i_s, v_sat, mode, channels_last = ctx.meta
        if channels_last:
            grad_out = grad_out.contiguous(memory_format=torch.channels_last)
        else:
            grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        b, c, h, w = x.shape
        total = b * c * h * w
        _node_transform_bwd_kernel[(triton.cdiv(total, _BLOCK),)](
            grad_out, x, nodes, coeff, y, dy, grad_x,
            b, c, h, w,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2), grad_x.stride(3),
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5],
            i_s, v_sat,
            N_NODES=nodes.numel(), MODE=mode,
            CHANNELS_LAST=channels_last, BLOCK=_BLOCK,
        )
        return grad_x, None, None, None, None


def state_nodes(params: dict, device=None) -> torch.Tensor:
    count = int(params.get("planar_nodes", 8))
    if count < 2:
        raise ValueError("planar_nodes must be at least 2")
    return torch.linspace(
        float(params["w_min"]),
        float(params["w_max"]),
        count,
        device=device,
        dtype=torch.float32,
    )


class _DifferentialBasis(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        theta_pos,
        theta_neg,
        node_count: int,
        lo: float,
        inv_step: float,
    ):
        theta_pos = theta_pos.contiguous()
        theta_neg = theta_neg.contiguous()
        n, k = theta_pos.shape
        out = torch.empty(
            n, node_count, k,
            device=theta_pos.device,
            dtype=_autocast_dtype(theta_pos),
        )
        total = theta_pos.numel()
        _basis_fwd_kernel[(triton.cdiv(total, _BLOCK),)](
            theta_pos, theta_neg, out,
            total, k,
            lo, inv_step,
            N_NODES=node_count, BLOCK=_BLOCK,
        )
        ctx.save_for_backward(theta_pos, theta_neg)
        ctx.meta = (lo, inv_step, node_count, k)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        theta_pos, theta_neg = ctx.saved_tensors
        lo, inv_step, node_count, k = ctx.meta
        grad_out = grad_out.contiguous()
        grad_pos = torch.empty_like(theta_pos)
        grad_neg = torch.empty_like(theta_neg)
        total = theta_pos.numel()
        _basis_bwd_kernel[(triton.cdiv(total, _BLOCK),)](
            grad_out, theta_pos, theta_neg, grad_pos, grad_neg,
            total, k,
            lo, inv_step,
            N_NODES=node_count, BLOCK=_BLOCK,
        )
        return grad_pos, grad_neg, None, None, None


def _differential_basis(
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    nodes: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    node_count = nodes.numel()
    lo = float(params["w_min"])
    inv_step = (node_count - 1) / (
        float(params["w_max"]) - float(params["w_min"])
    )
    return _DifferentialBasis.apply(
        theta_pos,
        theta_neg,
        node_count,
        lo,
        inv_step,
    )


def node_planar_conv2d(
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
    transformed = _NodeTransform.apply(
        x, nodes, device_type, params, channels_last
    )
    basis = _differential_basis(theta_pos, theta_neg, nodes, params)
    out_channels, _node_count, flat_k = basis.shape
    in_channels = x.shape[1]
    kh, kw = kernel_size
    if flat_k != in_channels * kh * kw:
        raise ValueError("state tensor and convolution geometry disagree")
    weight = basis.view(out_channels, nodes.numel() * in_channels, kh, kw)
    return F.conv2d(transformed, weight, stride=stride, padding=padding)


def node_planar_linear(
    x: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    nodes: torch.Tensor,
    params: dict,
    device_type: str,
) -> torch.Tensor:
    x4 = x.reshape(x.shape[0], x.shape[1], 1, 1)
    transformed = _NodeTransform.apply(
        x4, nodes, device_type, params, False
    ).flatten(1)
    basis = _differential_basis(theta_pos, theta_neg, nodes, params)
    weight = basis.reshape(theta_pos.shape[0], nodes.numel() * theta_pos.shape[1])
    return F.linear(transformed, weight)


def factorized_conv2d(
    x: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    params: dict,
    device_type: str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    out_channels = theta_pos.shape[0]
    in_channels = x.shape[1]
    kh, kw = kernel_size
    if device_type == "reram":
        transformed = torch.sinh((x / float(params["V0"])).clamp(-10.0, 10.0))
        coefficient = float(params["I0"]) * (
            torch.exp(-theta_pos / float(params["g0"]))
            - torch.exp(-theta_neg / float(params["g0"]))
        )
    elif device_type == "stt":
        alpha = float(params["alpha"])
        transformed = float(params["I0_stt"]) * x * (1.0 + alpha * x.square())
        coefficient = float(params["TMR"]) * (
            torch.cos(theta_pos) - torch.cos(theta_neg)
        )
    else:
        raise ValueError(f"factorized backend is not defined for {device_type!r}")
    weight = coefficient.view(out_channels, in_channels, kh, kw)
    return F.conv2d(transformed, weight, stride=stride, padding=padding)


def factorized_linear(
    x: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    params: dict,
    device_type: str,
) -> torch.Tensor:
    if device_type == "reram":
        transformed = torch.sinh((x / float(params["V0"])).clamp(-10.0, 10.0))
        coefficient = float(params["I0"]) * (
            torch.exp(-theta_pos / float(params["g0"]))
            - torch.exp(-theta_neg / float(params["g0"]))
        )
    elif device_type == "stt":
        alpha = float(params["alpha"])
        transformed = float(params["I0_stt"]) * x * (1.0 + alpha * x.square())
        coefficient = float(params["TMR"]) * (
            torch.cos(theta_pos) - torch.cos(theta_neg)
        )
    else:
        raise ValueError(f"factorized backend is not defined for {device_type!r}")
    return F.linear(transformed, coefficient)
