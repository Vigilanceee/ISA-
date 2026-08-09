#!/usr/bin/env python3
"""Benchmark the exact FeFET L-K/EKV Triton operator.

The benchmark imports the raw kernel directly from a source file, which makes
the selected implementation auditable and prevents an accelerated LUT or
planar route from being selected by configuration.  Both forward and
analytical backward kernels are timed with CUDA events and marked with NVTX
ranges for Nsight Systems/Compute.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERNEL = REPO_ROOT / "src" / "isa" / "kernels" / "device_sweep" / "fefet_triton.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "device_sweeps" / "device_params.yaml"

# Real im2col/linear dimensions from batch-1 VGG8.  Using batch 1 keeps the
# exact operator practical while retaining the layer fan-in and output width.
SHAPES: dict[str, tuple[int, int, int]] = {
    "micro": (64, 64, 64),
    "conv1_b1": (32 * 32, 128, 3 * 3 * 3),
    "conv2_b1": (32 * 32, 128, 128 * 3 * 3),
    "conv3_b1": (16 * 16, 256, 128 * 3 * 3),
    "conv4_b1": (16 * 16, 256, 256 * 3 * 3),
    "conv5_b1": (8 * 8, 512, 256 * 3 * 3),
    "conv6_b1": (8 * 8, 512, 512 * 3 * 3),
    "fc1_b1": (1, 1024, 512 * 4 * 4),
    "fc2_b1": (1, 10, 1024),
}

RAW_OVERRIDES: dict[str, Any] = {
    "raw_kernel_backend": "legacy",
    "raw_forward_backend": "legacy",
    "conv_backend": "reference",
    "linear_backend": "reference",
    "lut_enabled": False,
    "planar_enabled": False,
    "direct_conv_enabled": False,
    "surrogate_backward_enabled": False,
    "weight_quant_enabled": False,
}

EXPECTED_RAW_KERNELS = (
    "_fefet_fwd_kernel",
    "_fefet_grad_v_kernel",
    "_fefet_grad_w_kernel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-file", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("fefet_raw_benchmark.json"))
    parser.add_argument(
        "--shapes",
        default="micro,conv1_b1,conv3_b1,fc1_b1",
        help=f"Comma-separated presets. Choices: {','.join(SHAPES)}",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--raw-kernel-backend",
        choices=("legacy", "split"),
        default="legacy",
        help="Select the exact legacy or split-reduction backward kernels.",
    )
    parser.add_argument(
        "--raw-forward-backend",
        choices=("legacy", "split_k"),
        default="legacy",
        help="Select the exact legacy or FP32 split-K forward kernels.",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_function(kernel_file: Path):
    if not kernel_file.is_file():
        raise FileNotFoundError(kernel_file)
    spec = importlib.util.spec_from_file_location("fefet_raw_profile_kernel", kernel_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import raw FeFET kernel from {kernel_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FeFETFunction


def load_raw_params(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        params = dict(yaml.safe_load(handle)["fefet"])
    configured = {
        key: params.get(key)
        for key in (
            "raw_kernel_backend",
            "raw_forward_backend",
            "conv_backend",
            "linear_backend",
            "lut_enabled",
            "planar_enabled",
            "direct_conv_enabled",
            "surrogate_backward_enabled",
            "weight_quant_enabled",
        )
    }
    params.update(RAW_OVERRIDES)
    assert params["raw_kernel_backend"] == "legacy"
    assert params["raw_forward_backend"] == "legacy"
    assert params["conv_backend"] == "reference"
    assert params["linear_backend"] == "reference"
    for key in (
        "lut_enabled",
        "planar_enabled",
        "direct_conv_enabled",
        "surrogate_backward_enabled",
        "weight_quant_enabled",
    ):
        assert params[key] is False, (key, params[key])
    return params, configured


def raw_torch_formula(
    voltage: torch.Tensor,
    theta_pos: torch.Tensor,
    theta_neg: torch.Tensor,
    params: dict[str, Any],
) -> torch.Tensor:
    """Small-shape PyTorch transcription of the same six-step L-K/EKV formula."""

    def branch(theta: torch.Tensor) -> torch.Tensor:
        v = voltage[:, None, :]
        threshold = theta[None, :, :]
        y = v - threshold
        a_lk = float(params["A_lk"])
        b_lk = float(params["B_lk"])
        x = y / max(b_lk, 1.0e-8)
        for _ in range(6):
            residual = a_lk * x**3 + b_lk * x - y
            derivative = 3.0 * a_lk * x.square() + b_lk
            x = x - residual / derivative.clamp_min(1.0e-8)
        inv_2nut = 1.0 / (2.0 * float(params["n"]) * float(params["U_T"]))
        first = F.softplus(x * inv_2nut)
        second = F.softplus((x - float(params["V_D"])) * inv_2nut)
        basic = float(params["I_S"]) * (first.square() - second.square())
        return basic / (1.0 + v / float(params["V_sat"]))

    return (branch(theta_pos) - branch(theta_neg)).sum(dim=-1)


def validate_formula(raw_function, params: dict[str, Any], seed: int) -> dict[str, float]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 17)
    shape = (8, 7, 11)
    m, n, k = shape
    v_base = torch.rand((m, k), device="cuda", generator=generator) * 4.0
    p_base = 2.45 + torch.rand((n, k), device="cuda", generator=generator) * 0.1
    n_base = 2.45 + torch.rand((n, k), device="cuda", generator=generator) * 0.1
    grad = torch.randn((m, n), device="cuda", generator=generator)

    v_raw, p_raw, n_raw = (item.detach().clone().requires_grad_(True) for item in (v_base, p_base, n_base))
    with torch.cuda.nvtx.range("raw_fefet_validation"):
        out_raw = raw_function.apply(v_raw, p_raw, n_raw, params)
        gradients_raw = torch.autograd.grad(out_raw, (v_raw, p_raw, n_raw), grad)

    v_ref, p_ref, n_ref = (item.detach().clone().requires_grad_(True) for item in (v_base, p_base, n_base))
    out_ref = raw_torch_formula(v_ref, p_ref, n_ref, params)
    gradients_ref = torch.autograd.grad(out_ref, (v_ref, p_ref, n_ref), grad)
    torch.cuda.synchronize()

    def errors(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict[str, float]:
        delta = (actual.float() - expected.float()).abs()
        denominator = expected.float().abs().clamp_min(1.0e-6)
        return {
            f"{name}_max_abs": float(delta.max().item()),
            f"{name}_mean_abs": float(delta.mean().item()),
            f"{name}_max_rel": float((delta / denominator).max().item()),
        }

    result = errors(out_raw, out_ref, "forward")
    for name, actual, expected in zip(("grad_v", "grad_pos", "grad_neg"), gradients_raw, gradients_ref):
        result.update(errors(actual, expected, name))
    return result


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": float(statistics.median(ordered)),
        "minimum_ms": float(ordered[0]),
        "maximum_ms": float(ordered[-1]),
        "mean_ms": float(statistics.fmean(ordered)),
    }


def run_once(raw_function, params, tensors, name: str, record: bool):
    voltage, theta_pos, theta_neg, grad = tensors
    theta_pos.grad = None
    theta_neg.grad = None
    voltage.grad = None

    start = torch.cuda.Event(enable_timing=True)
    after_forward = torch.cuda.Event(enable_timing=True)
    after_backward = torch.cuda.Event(enable_timing=True)
    start.record()
    if record:
        torch.cuda.nvtx.range_push(f"raw_fefet_forward:{name}")
    output = raw_function.apply(voltage, theta_pos, theta_neg, params)
    if record:
        torch.cuda.nvtx.range_pop()
    after_forward.record()
    if record:
        torch.cuda.nvtx.range_push(f"raw_fefet_backward:{name}")
    output.backward(grad)
    if record:
        torch.cuda.nvtx.range_pop()
    after_backward.record()
    return start, after_forward, after_backward, output


def benchmark_shape(
    raw_function,
    params: dict[str, Any],
    name: str,
    shape: tuple[int, int, int],
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    m, n, k = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + sum(ord(char) for char in name))
    voltage = (torch.rand((m, k), device="cuda", generator=generator) * 4.0).requires_grad_(True)
    theta_pos = (2.45 + torch.rand((n, k), device="cuda", generator=generator) * 0.1).requires_grad_(True)
    theta_neg = (2.45 + torch.rand((n, k), device="cuda", generator=generator) * 0.1).requires_grad_(True)
    grad = torch.randn((m, n), device="cuda", generator=generator)
    tensors = (voltage, theta_pos, theta_neg, grad)

    with torch.cuda.nvtx.range(f"raw_fefet_compile_and_warmup:{name}"):
        for _ in range(warmup):
            run_once(raw_function, params, tensors, name, record=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    records = [run_once(raw_function, params, tensors, name, record=True) for _ in range(iterations)]
    torch.cuda.synchronize()
    forward_ms = [float(start.elapsed_time(middle)) for start, middle, _, _ in records]
    backward_ms = [float(middle.elapsed_time(end)) for _, middle, end, _ in records]
    total_ms = [forward + backward for forward, backward in zip(forward_ms, backward_ms)]
    last_output = records[-1][3]
    finite = all(
        bool(torch.isfinite(item).all().item())
        for item in (last_output, voltage.grad, theta_pos.grad, theta_neg.grad)
    )
    interactions = m * n * k
    summary = {
        "shape": {"M": m, "N": n, "K": k},
        "vgg8_batch": 1,
        "device_interactions": interactions,
        "forward": distribution(forward_ms),
        "backward": distribution(backward_ms),
        "forward_backward": distribution(total_ms),
        "median_interactions_per_second": interactions / (statistics.median(total_ms) / 1000.0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_output_and_gradients": finite,
    }
    del records, tensors, voltage, theta_pos, theta_neg, grad, last_output
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    selected = [item.strip() for item in args.shapes.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SHAPES))
    if unknown:
        raise ValueError(f"unknown shapes: {unknown}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    raw_function = load_raw_function(args.kernel_file.resolve())
    params, configured_backends = load_raw_params(args.config.resolve())
    params["raw_kernel_backend"] = args.raw_kernel_backend
    params["raw_forward_backend"] = args.raw_forward_backend

    started = time.time()
    result: dict[str, Any] = {
        "status": "running",
        "method": "exact FeFET L-K solve plus EKV current with analytical backward",
        "kernel_source": str(args.kernel_file.resolve()),
        "kernel_sha256": sha256(args.kernel_file.resolve()),
        "config_source": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "configured_backends_before_override": configured_backends,
        "enforced_raw_backend": {
            **RAW_OVERRIDES,
            "raw_kernel_backend": args.raw_kernel_backend,
            "raw_forward_backend": args.raw_forward_backend,
        },
        "dispatch": "FeFETFunction.apply imported directly from fefet_triton.py",
        "expected_trace_kernels": list(EXPECTED_RAW_KERNELS),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iters,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "formula_validation": None,
        "benchmarks": {},
    }
    if not args.skip_validation:
        result["formula_validation"] = validate_formula(raw_function, params, args.seed)
    for name in selected:
        result["benchmarks"][name] = benchmark_shape(
            raw_function,
            params,
            name,
            SHAPES[name],
            args.warmup,
            args.iters,
            args.seed,
        )
        print(json.dumps({"shape_complete": name, **result["benchmarks"][name]}, sort_keys=True), flush=True)

    result["status"] = "completed"
    result["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"FEFET_RAW_BENCHMARK={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
