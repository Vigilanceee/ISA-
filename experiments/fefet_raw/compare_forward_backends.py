#!/usr/bin/env python3
"""Compare exact legacy and optimized FeFET forward implementations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics

import torch


PARAMS = {
    "I_S": 11.0725,
    "n": 1.112106,
    "U_T": 0.026,
    "V_D": 0.1,
    "A_lk": 0.001369,
    "B_lk": 1.29224,
    "V_sat": 8.2624,
    "raw_kernel_backend": "split",
}


def load_function(path: Path):
    spec = importlib.util.spec_from_file_location("fefet_splitk_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeFETFunction


def make_tensors(shape: tuple[int, int, int], seed: int):
    m, n, k = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    voltage = torch.rand((m, k), device="cuda", generator=generator) * 4.0
    theta_pos = 2.5 + 0.05 * torch.randn((n, k), device="cuda", generator=generator)
    theta_neg = 2.5 + 0.05 * torch.randn((n, k), device="cuda", generator=generator)
    grad = torch.randn((m, n), device="cuda", generator=generator)
    return voltage, theta_pos, theta_neg, grad


def run(function, tensors, forward_backend: str):
    copied = [tensor.detach().clone() for tensor in tensors]
    voltage, theta_pos, theta_neg = [tensor.requires_grad_(True) for tensor in copied[:3]]
    grad = copied[3]
    params = dict(PARAMS, raw_forward_backend=forward_backend)
    output = function.apply(voltage, theta_pos, theta_neg, params)
    output.backward(grad)
    torch.cuda.synchronize()
    return (
        output.detach(),
        voltage.grad.detach(),
        theta_pos.grad.detach(),
        theta_neg.grad.detach(),
    )


def errors(reference: torch.Tensor, candidate: torch.Tensor):
    delta = (candidate.float() - reference.float()).abs()
    denominator = reference.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_rel": float((delta / denominator).max()),
        "mean_rel": float((delta / denominator).mean()),
    }


def time_forward(function, tensors, backend: str, repeats: int):
    voltage, theta_pos, theta_neg, _ = tensors
    params = dict(PARAMS, raw_forward_backend=backend)
    measurements = []
    for index in range(repeats + 2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function.apply(voltage, theta_pos, theta_neg, params)
        end.record()
        torch.cuda.synchronize()
        if index >= 2:
            measurements.append(float(start.elapsed_time(end)))
        del output
    return {
        "median_ms": statistics.median(measurements),
        "minimum_ms": min(measurements),
        "maximum_ms": max(measurements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--shapes", default="1024,128,27;256,256,1152;1,1024,8192")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    function = load_function(args.module)
    records = {}
    for shape_text in args.shapes.split(";"):
        shape = tuple(int(value) for value in shape_text.split(","))
        if len(shape) != 3:
            raise ValueError(shape_text)
        tensors = make_tensors(shape, 20260805 + sum(shape))
        legacy = run(function, tensors, "legacy")
        split = run(function, tensors, "split_k")
        correctness = {
            name: errors(reference, candidate)
            for name, reference, candidate in zip(
                ("forward", "grad_v", "grad_pos", "grad_neg"), legacy, split
            )
        }
        timings = {
            backend: time_forward(function, tensors, backend, args.repeats)
            for backend in ("legacy", "split_k")
        }
        timings["speedup"] = (
            timings["legacy"]["median_ms"] / timings["split_k"]["median_ms"]
        )
        records[shape_text] = {"correctness": correctness, "forward_timings": timings}
        print(json.dumps({shape_text: records[shape_text]}, sort_keys=True), flush=True)
    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
