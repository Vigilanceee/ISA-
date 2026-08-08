#!/usr/bin/env python3
"""Benchmark exact direct convolution against unfold + exact matrix formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import yaml

from isa.kernels.device_sweep.direct_conv_triton import DirectFormulaConv2d
from isa.kernels.device_sweep.fefet_triton import FeFETFunction
from isa.kernels.device_sweep.flash_triton import FlashFunction
from isa.kernels.device_sweep.pcm_triton import PCMFunction
from isa.kernels.transformer_ffn.ekv_triton import TritonEKVMatmulFn


ROOT = Path(__file__).resolve().parents[2]
DEVICE_FUNCTIONS = {
    "pcm": PCMFunction,
    "fefet": FeFETFunction,
    "flash": FlashFunction,
}
SHAPES = {
    "conv1": (8, 3, 128, 32, 32),
    "conv3": (2, 128, 256, 16, 16),
    "conv5": (1, 256, 512, 8, 8),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/device_sweeps/device_params.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile-pass", action="store_true")
    parser.add_argument("--devices", default="pcm,fefet,flash")
    parser.add_argument("--shapes", default="conv1,conv3,conv5")
    return parser.parse_args()


def inputs(device: str, shape: tuple[int, int, int, int, int], params: dict):
    batch, channels, outputs, height, width = shape
    if device == "pcm":
        x = torch.empty(batch, channels, height, width, device="cuda").uniform_(-0.5, 0.5)
    else:
        x = torch.empty(batch, channels, height, width, device="cuda").uniform_(0.0, 4.0)
    weights = (outputs, channels * 3 * 3)
    wp = torch.empty(weights, device="cuda").uniform_(params["w_min"], params["w_max"])
    wn = torch.empty(weights, device="cuda").uniform_(params["w_min"], params["w_max"])
    return x, wp, wn


def unfold_exact(x, wp, wn, params, function):
    batch, _channels, height, width = x.shape
    columns = F.unfold(x, kernel_size=3, stride=1, padding=1)
    flat = columns.permute(0, 2, 1).contiguous().view(batch * height * width, -1)
    output = function.apply(flat, wp, wn, params)
    return output.view(batch, height * width, wp.shape[0]).permute(0, 2, 1).reshape(
        batch, wp.shape[0], height, width
    )


def direct_exact(x, wp, wn, params, device, function):
    return DirectFormulaConv2d.apply(
        x, wp, wn, params, device, (3, 3), (1, 1), (1, 1), function
    )


def one_step(route, x, wp, wn):
    for tensor in (x, wp, wn):
        tensor.grad = None
    output = route(x, wp, wn)
    output.square().mean().backward()


def measure(route, x, wp, wn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        one_step(route, x, wp, wn)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one_step(route, x, wp, wn)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260808)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for device in args.devices.split(","):
        params = dict(config[device])
        function = DEVICE_FUNCTIONS[device]
        for shape_name in args.shapes.split(","):
            shape = SHAPES[shape_name]
            x, wp, wn = inputs(device, shape, params)
            x.requires_grad_(True)
            wp.requires_grad_(True)
            wn.requires_grad_(True)
            routes = {
                "unfold_exact": lambda a, b, c: unfold_exact(a, b, c, params, function),
                "direct_exact": lambda a, b, c: direct_exact(a, b, c, params, device, function),
            }
            if device == "flash":
                routes["unfold_ffn_exact"] = lambda a, b, c: unfold_exact(
                    a, b, c, params, TritonEKVMatmulFn
                )
            timings = {}
            for route_name, route in routes.items():
                with torch.cuda.nvtx.range(f"{device}/{shape_name}/{route_name}"):
                    samples = measure(
                        route,
                        x,
                        wp,
                        wn,
                        0 if args.profile_pass else args.warmup,
                        1 if args.profile_pass else args.iters,
                    )
                timings[route_name] = statistics.median(samples)
            with torch.no_grad():
                reference = unfold_exact(x, wp, wn, params, function)
                direct = direct_exact(x, wp, wn, params, device, function)
                max_abs = float((reference - direct).abs().max())
                rel_l2 = float((reference - direct).float().norm() / reference.float().norm().clamp_min(1e-12))
            rows.append(
                {
                    "device": device,
                    "shape": shape_name,
                    "dimensions": shape,
                    "unfold_exact_ms": timings["unfold_exact"],
                    "direct_exact_ms": timings["direct_exact"],
                    "direct_speedup": timings["unfold_exact"] / timings["direct_exact"],
                    "ffn_exact_ms": timings.get("unfold_ffn_exact"),
                    "ffn_speedup": (
                        timings["unfold_exact"] / timings["unfold_ffn_exact"]
                        if "unfold_ffn_exact" in timings
                        else None
                    ),
                    "max_abs_error": max_abs,
                    "relative_l2_error": rel_l2,
                }
            )
    payload = {
        "generated_unix": time.time(),
        "cuda_device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
