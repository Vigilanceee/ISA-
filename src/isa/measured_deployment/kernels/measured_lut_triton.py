"""Fused measured-IV lookup and reduction kernels.

The existing implementation expands one slope/intercept matrix per voltage
segment and launches two GEMMs per segment.  This kernel keeps assignments
compact and directly gathers the required curve coefficient while reducing
the input dimension.

Two layouts are supported:

* ``pair_mode=True``: assignment A contains one of 24×24 differential-pair
  ids and the coefficient tables already contain I_pos - I_neg.
* ``pair_mode=False``: assignment A/B contain concrete positive/negative
  member-curve ids and the kernel subtracts their coefficients on the fly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _measured_lut_kernel(
    x_ptr,
    assignment_a_ptr,
    assignment_b_ptr,
    slope_ptr,
    intercept_ptr,
    output_ptr,
    voltage_min,
    inverse_voltage_step,
    lookup_min,
    lookup_max,
    current_scale,
    r_tia,
    signed_min,
    signed_max,
    BATCH: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    SEGMENTS: tl.constexpr,
    PAIR_MODE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_ids = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    output_ids = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
    batch_mask = batch_ids < BATCH
    output_mask = output_ids < OUT_FEATURES
    accumulator = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)

    for k_start in range(0, IN_FEATURES, BLOCK_K):
        input_ids = k_start + tl.arange(0, BLOCK_K)
        input_mask = input_ids < IN_FEATURES
        x_offsets = (
            batch_ids[:, None] * IN_FEATURES + input_ids[None, :]
        )
        x = tl.load(
            x_ptr + x_offsets,
            mask=batch_mask[:, None] & input_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.maximum(tl.minimum(x, lookup_max), lookup_min)
        # x is clamped to the non-negative FG50 lookup range. Integer
        # conversion therefore has the same result as floor, while remaining
        # compatible with the Triton 3.0 build installed on CCI.
        segment = (
            (x - voltage_min) * inverse_voltage_step
        ).to(tl.int32)
        segment = tl.maximum(0, tl.minimum(segment, SEGMENTS - 1))

        assignment_offsets = (
            output_ids[:, None] * IN_FEATURES + input_ids[None, :]
        )
        assignment_mask = output_mask[:, None] & input_mask[None, :]
        assignment_a = tl.load(
            assignment_a_ptr + assignment_offsets,
            mask=assignment_mask,
            other=0,
        ).to(tl.int32)
        coefficient_mask = (
            batch_mask[:, None, None]
            & output_mask[None, :, None]
            & input_mask[None, None, :]
        )
        coefficient_offset_a = (
            assignment_a[None, :, :] * SEGMENTS + segment[:, None, :]
        )
        slope_a = tl.load(
            slope_ptr + coefficient_offset_a,
            mask=coefficient_mask,
            other=0.0,
        ).to(tl.float32)
        intercept_a = tl.load(
            intercept_ptr + coefficient_offset_a,
            mask=coefficient_mask,
            other=0.0,
        ).to(tl.float32)

        if PAIR_MODE:
            value = slope_a * x[:, None, :] + intercept_a
        else:
            assignment_b = tl.load(
                assignment_b_ptr + assignment_offsets,
                mask=assignment_mask,
                other=0,
            ).to(tl.int32)
            coefficient_offset_b = (
                assignment_b[None, :, :] * SEGMENTS + segment[:, None, :]
            )
            slope_b = tl.load(
                slope_ptr + coefficient_offset_b,
                mask=coefficient_mask,
                other=0.0,
            ).to(tl.float32)
            intercept_b = tl.load(
                intercept_ptr + coefficient_offset_b,
                mask=coefficient_mask,
                other=0.0,
            ).to(tl.float32)
            value = (
                (slope_a - slope_b) * x[:, None, :]
                + intercept_a
                - intercept_b
            )
        accumulator += tl.sum(value, axis=2)

    output = accumulator * current_scale * r_tia
    output = tl.maximum(tl.minimum(output, signed_max), signed_min)
    output_offsets = (
        batch_ids[:, None] * OUT_FEATURES + output_ids[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=batch_mask[:, None] & output_mask[None, :],
    )


def _uniform_voltage_grid(voltage_v: torch.Tensor) -> tuple[float, float]:
    if voltage_v.ndim != 1 or voltage_v.numel() < 2:
        raise ValueError("voltage_v must be a 1-D grid with at least two points")
    delta = voltage_v[1:] - voltage_v[:-1]
    if not bool(torch.all(delta > 0)):
        raise ValueError("voltage_v must be strictly increasing")
    reference = delta[0]
    if not bool(torch.allclose(delta, reference, atol=1e-6, rtol=1e-5)):
        raise ValueError("the fused kernel currently requires a uniform voltage grid")
    return float(voltage_v[0].item()), 1.0 / float(reference.item())


@torch.no_grad()
def fused_measured_forward(
    x: torch.Tensor,
    assignment_a: torch.Tensor,
    slopes: torch.Tensor,
    intercepts: torch.Tensor,
    voltage_v: torch.Tensor,
    *,
    assignment_b: torch.Tensor | None = None,
    pair_mode: bool,
    lookup_min: float = 0.0,
    lookup_max: float = 4.0,
    current_scale: float = 1e-9,
    r_tia: float | torch.Tensor = 1.0,
    signed_min: float = -4.0,
    signed_max: float = 4.0,
    block_b: int = 2,
    block_o: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    voltage_min: float | None = None,
    inverse_voltage_step: float | None = None,
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.shape[-1] != assignment_a.shape[1]:
        raise ValueError("x input dimension and assignment width differ")
    if assignment_a.ndim != 2:
        raise ValueError("assignment_a must have shape [out_features, in_features]")
    if not pair_mode and (assignment_b is None or assignment_b.shape != assignment_a.shape):
        raise ValueError("member mode requires assignment_b with matching shape")
    if slopes.ndim != 2 or intercepts.shape != slopes.shape:
        raise ValueError("slope/intercept tables must have matching 2-D shapes")
    if slopes.shape[1] != voltage_v.numel() - 1:
        raise ValueError("coefficient segment count does not match voltage grid")
    if assignment_a.device != x.device:
        raise ValueError("assignments and x must be on the same CUDA device")
    if slopes.device != x.device or intercepts.device != x.device:
        raise ValueError("coefficient tables and x must be on the same CUDA device")

    if (voltage_min is None) != (inverse_voltage_step is None):
        raise ValueError(
            "voltage_min and inverse_voltage_step must be provided together"
        )
    if voltage_min is None:
        # The direct functional API validates the grid. Performance-sensitive
        # callers should validate once during configuration and pass both
        # scalar values, avoiding CUDA-to-CPU synchronization on every layer.
        voltage_min, inverse_step = _uniform_voltage_grid(voltage_v)
    else:
        inverse_step = float(inverse_voltage_step)
        voltage_min = float(voltage_min)
        if inverse_step <= 0:
            raise ValueError("inverse_voltage_step must be positive")
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1]).float().contiguous()
    assignment_a = assignment_a.contiguous()
    if assignment_b is None:
        assignment_b = assignment_a
    else:
        assignment_b = assignment_b.contiguous()
    slopes = slopes.float().contiguous()
    intercepts = intercepts.float().contiguous()
    output = torch.empty(
        (flat.shape[0], assignment_a.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    tensor_gain = isinstance(r_tia, torch.Tensor)
    if tensor_gain:
        if r_tia.numel() != 1:
            raise ValueError("tensor-valued r_tia must contain one scalar")
        if r_tia.device != x.device:
            raise ValueError("tensor-valued r_tia and x must be on the same device")
    # Reading a CUDA scalar with .item() would synchronize every forward.
    # Tensor-valued TIA gain is therefore applied by a device-side pointwise
    # operation after the fused lookup/reduction kernel.
    kernel_gain = 1.0 if tensor_gain else float(r_tia)
    kernel_signed_min = float("-inf") if tensor_gain else float(signed_min)
    kernel_signed_max = float("inf") if tensor_gain else float(signed_max)
    grid = (
        triton.cdiv(flat.shape[0], block_b),
        triton.cdiv(assignment_a.shape[0], block_o),
    )
    _measured_lut_kernel[grid](
        flat,
        assignment_a,
        assignment_b,
        slopes,
        intercepts,
        output,
        voltage_min,
        inverse_step,
        float(lookup_min),
        float(lookup_max),
        float(current_scale),
        kernel_gain,
        kernel_signed_min,
        kernel_signed_max,
        BATCH=flat.shape[0],
        OUT_FEATURES=assignment_a.shape[0],
        IN_FEATURES=assignment_a.shape[1],
        SEGMENTS=slopes.shape[1],
        PAIR_MODE=bool(pair_mode),
        BLOCK_B=int(block_b),
        BLOCK_O=int(block_o),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
    )
    if tensor_gain:
        output.mul_(r_tia.detach().float().reshape(())).clamp_(
            float(signed_min), float(signed_max)
        )
    return output.reshape(*original_shape[:-1], assignment_a.shape[0]).to(x.dtype)
