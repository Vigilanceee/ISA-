"""
Direct-convolution forward kernels for slow transistor devices.

This follows the acceleration pattern used in differential pairs/diff1:
avoid materializing F.unfold in the forward pass and compute the sliding-window
addresses directly inside Triton.  The device equations remain the equations
from this project:

* Flash: exact EKV + velocity-saturation formula from flash_triton.py.
* FeFET: LUT generated from the current FeFET L-K + EKV formula.

Backward intentionally reuses the existing unfold + device backward path.  That
keeps gradients consistent with the existing implementation while removing the
largest forward-side memory traffic for VGG8 convolution layers.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .lut_triton import _make_fefet_tables


@triton.jit
def _sp(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _interp1(table, x, x_min: tl.constexpr, inv_dx: tl.constexpr, size: tl.constexpr):
    xf = (x - x_min) * inv_dx
    xf = tl.minimum(tl.maximum(xf, 0.0), size - 1.001)
    i0 = xf.to(tl.int32)
    t = xf - i0.to(tl.float32)
    y0 = tl.load(table + i0)
    y1 = tl.load(table + i0 + 1)
    return y0 + t * (y1 - y0)


def _configs():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=_configs(), key=["B", "C", "OC", "H", "W", "OH", "OW", "KH", "KW", "STRIDE_H", "STRIDE_W", "PAD_H", "PAD_W"])
@triton.jit
def _flash_conv_fwd_kernel(
    X, WP, WN, O,
    B, C, OC, H, W, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    TOTAL_K: tl.constexpr,
    sx_b, sx_c, sx_h, sx_w,
    sw_o, sw_k,
    so_b, so_o, so_h, so_w,
    I_S, inv_2nUT, VD_over_2nUT, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    total_m = B * OH * OW
    num_n = tl.cdiv(OC, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < total_m
    mask_n = offs_n < OC

    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH
    base_h = oh * STRIDE_H - PAD_H
    base_w = ow * STRIDE_W - PAD_W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(TOTAL_K, BLOCK_K)):
        for ki in range(BLOCK_K):
            k = kb * BLOCK_K + ki
            if k < TOTAL_K:
                c = k // (KH * KW)
                rem = k % (KH * KW)
                kh = rem // KW
                kw = rem % KW
                ih = base_h + kh
                iw = base_w + kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                v = tl.load(X + b * sx_b + c * sx_c + ih * sx_h + iw * sx_w, mask=mask_m & valid, other=0.0).to(tl.float32)
                wp = tl.load(WP + offs_n * sw_o + k * sw_k, mask=mask_n, other=0.0).to(tl.float32)
                wn = tl.load(WN + offs_n * sw_o + k * sw_k, mask=mask_n, other=0.0).to(tl.float32)
                vv = v[:, None]
                denom = 1.0 + vv / V_sat

                up = (vv - wp[None, :]) * inv_2nUT
                sp1 = _sp(up)
                sp2 = _sp(up - VD_over_2nUT)
                ip = sp1 * sp1 - sp2 * sp2

                un = (vv - wn[None, :]) * inv_2nUT
                sn1 = _sp(un)
                sn2 = _sp(un - VD_over_2nUT)
                inn = sn1 * sn1 - sn2 * sn2
                acc += I_S * (ip - inn) / denom

    tl.store(
        O + b[:, None] * so_b + offs_n[None, :] * so_o + oh[:, None] * so_h + ow[:, None] * so_w,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(configs=_configs(), key=["B", "C", "OC", "H", "W", "OH", "OW", "KH", "KW", "STRIDE_H", "STRIDE_W", "PAD_H", "PAD_W"])
@triton.jit
def _fefet_lut_conv_fwd_kernel(
    X, WP, WN, O, Y,
    B, C, OC, H, W, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    TOTAL_K: tl.constexpr,
    sx_b, sx_c, sx_h, sx_w,
    sw_o, sw_k,
    so_b, so_o, so_h, so_w,
    x_min: tl.constexpr, inv_dx: tl.constexpr, lut_size: tl.constexpr,
    I_S, V_sat,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    total_m = B * OH * OW
    num_n = tl.cdiv(OC, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < total_m
    mask_n = offs_n < OC

    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b = tmp // OH
    base_h = oh * STRIDE_H - PAD_H
    base_w = ow * STRIDE_W - PAD_W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(TOTAL_K, BLOCK_K)):
        for ki in range(BLOCK_K):
            k = kb * BLOCK_K + ki
            if k < TOTAL_K:
                c = k // (KH * KW)
                rem = k % (KH * KW)
                kh = rem // KW
                kw = rem % KW
                ih = base_h + kh
                iw = base_w + kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                v = tl.load(X + b * sx_b + c * sx_c + ih * sx_h + iw * sx_w, mask=mask_m & valid, other=0.0).to(tl.float32)
                wp = tl.load(WP + offs_n * sw_o + k * sw_k, mask=mask_n, other=0.0).to(tl.float32)
                wn = tl.load(WN + offs_n * sw_o + k * sw_k, mask=mask_n, other=0.0).to(tl.float32)
                vv = v[:, None]
                scale = I_S / (1.0 + vv / V_sat)
                acc += scale * _interp1(Y, vv - wp[None, :], x_min, inv_dx, lut_size)
                acc -= scale * _interp1(Y, vv - wn[None, :], x_min, inv_dx, lut_size)

    tl.store(
        O + b[:, None] * so_b + offs_n[None, :] * so_o + oh[:, None] * so_h + ow[:, None] * so_w,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


class DirectTransistorConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_pos, w_neg, params, device_type, kernel_size, stride, padding, fallback_func):
        x = x.contiguous()
        w_pos = w_pos.contiguous()
        w_neg = w_neg.contiguous()
        b, c, h, w = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        oc = w_pos.shape[0]
        out = torch.empty((b, oc, oh, ow), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(b * oh * ow, meta["BLOCK_M"]) * triton.cdiv(oc, meta["BLOCK_N"]),)
        if device_type == "flash":
            n_fac = float(params["n"])
            inv_2nUT = 1.0 / (2.0 * n_fac * float(params["U_T"]))
            nvd = float(params["V_D"]) * inv_2nUT
            _flash_conv_fwd_kernel[grid](
                x, w_pos, w_neg, out,
                b, c, oc, h, w, oh, ow,
                kh, kw, sh, sw, ph, pw, c * kh * kw,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                w_pos.stride(0), w_pos.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                float(params["I_S"]), inv_2nUT, nvd, float(params["V_sat"]),
            )
        else:
            y, _dy, meta = _make_fefet_tables(params, x.device, x.dtype)
            _fefet_lut_conv_fwd_kernel[grid](
                x, w_pos, w_neg, out, y,
                b, c, oc, h, w, oh, ow,
                kh, kw, sh, sw, ph, pw, c * kh * kw,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                w_pos.stride(0), w_pos.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                meta[0], meta[1], meta[2],
                meta[3], meta[4],
            )

        ctx.save_for_backward(x, w_pos, w_neg)
        ctx.params = params
        ctx.device_type = device_type
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.fallback_func = fallback_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_pos, w_neg = ctx.saved_tensors
        params = ctx.params
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        fallback_func = ctx.fallback_func
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(ctx.needs_input_grad[0])
            wp_req = w_pos.detach().requires_grad_(ctx.needs_input_grad[1])
            wn_req = w_neg.detach().requires_grad_(ctx.needs_input_grad[2])
            b, _c, h, w = x_req.shape
            kh, kw = kernel_size
            ph, pw = padding
            sh, sw = stride
            oh = (h + 2 * ph - kh) // sh + 1
            ow = (w + 2 * pw - kw) // sw + 1
            cols = F.unfold(x_req, kernel_size=kernel_size, stride=stride, padding=padding)
            flat = cols.permute(0, 2, 1).contiguous().view(b * oh * ow, -1)
            out_flat = fallback_func.apply(flat, wp_req, wn_req, params)
            out = out_flat.view(b, oh * ow, w_pos.shape[0]).permute(0, 2, 1).contiguous().view(b, w_pos.shape[0], oh, ow)
        grad_inputs = []
        grad_slots = []
        if ctx.needs_input_grad[0]:
            grad_inputs.append(x_req)
            grad_slots.append(0)
        if ctx.needs_input_grad[1]:
            grad_inputs.append(wp_req)
            grad_slots.append(1)
        if ctx.needs_input_grad[2]:
            grad_inputs.append(wn_req)
            grad_slots.append(2)
        grads_out = [None, None, None]
        if grad_inputs:
            grads = torch.autograd.grad(
                out,
                tuple(grad_inputs),
                grad_output,
                retain_graph=False,
                allow_unused=False,
            )
            for slot, grad in zip(grad_slots, grads):
                grads_out[slot] = grad
        return grads_out[0], grads_out[1], grads_out[2], None, None, None, None, None, None
