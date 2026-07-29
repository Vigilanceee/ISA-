#!/usr/bin/env python3
"""Numerical parity checks for accelerated convolution backends."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from isa.kernels.device_sweep.custom_layers import NVM_Conv2d


DEVICES = ("reram", "pcm", "stt", "fefet", "flash")


def rms_relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    num = (actual.float() - reference.float()).square().mean().sqrt()
    den = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float((num / den).item())


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = actual.float().flatten()
    b = reference.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item())


def make_layer(device_type: str, params: dict, backend: str) -> NVM_Conv2d:
    local = copy.deepcopy(params)
    local["conv_backend"] = backend
    if backend == "reference":
        local["direct_conv_enabled"] = False
        local["lut_enabled"] = False
        local["planar_enabled"] = False
    return NVM_Conv2d(
        2, 3, kernel_size=3, padding=1,
        device_type=device_type,
        device_params=local,
        w_init_max=0.1,
    ).cuda()


def validate(device_type: str, params: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    backend = str(params["conv_backend"])
    reference = make_layer(device_type, params, "reference")
    accelerated = make_layer(device_type, params, backend)

    lo = float(params["w_min"])
    hi = float(params["w_max"])
    with torch.no_grad():
        theta_pos = torch.empty_like(reference.theta_pos).uniform_(lo, hi)
        theta_neg = torch.empty_like(reference.theta_neg).uniform_(lo, hi)
        reference.theta_pos.copy_(theta_pos)
        reference.theta_neg.copy_(theta_neg)
        accelerated.theta_pos.copy_(theta_pos)
        accelerated.theta_neg.copy_(theta_neg)

    if device_type in {"fefet", "flash"}:
        x = torch.empty(2, 2, 5, 5, device="cuda").uniform_(-4.0, 4.0)
    else:
        x = torch.empty(2, 2, 5, 5, device="cuda").uniform_(-0.5, 0.5)
    x_ref = x.detach().requires_grad_(True)
    x_acc = x.detach().requires_grad_(True)
    probe = torch.randn(2, 3, 5, 5, device="cuda")

    y_ref = reference(x_ref)
    loss_ref = (y_ref * probe).sum()
    grads_ref = torch.autograd.grad(
        loss_ref,
        (x_ref, reference.theta_pos, reference.theta_neg),
    )

    y_acc = accelerated(x_acc)
    loss_acc = (y_acc * probe).sum()
    grads_acc = torch.autograd.grad(
        loss_acc,
        (x_acc, accelerated.theta_pos, accelerated.theta_neg),
    )
    torch.cuda.synchronize()

    return {
        "device": device_type,
        "backend": backend,
        "planar_nodes": int(params.get("planar_nodes", 0)),
        "lowrank_rank": int(params.get("lowrank_rank", 0)),
        "forward_rms_relative": rms_relative(y_acc, y_ref),
        "forward_max_absolute": float((y_acc - y_ref).abs().max().item()),
        "grad_x_rms_relative": rms_relative(grads_acc[0], grads_ref[0]),
        "grad_x_cosine": cosine(grads_acc[0], grads_ref[0]),
        "grad_pos_rms_relative": rms_relative(grads_acc[1], grads_ref[1]),
        "grad_pos_cosine": cosine(grads_acc[1], grads_ref[1]),
        "grad_neg_rms_relative": rms_relative(grads_acc[2], grads_ref[2]),
        "grad_neg_cosine": cosine(grads_acc[2], grads_ref[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/device_params.yaml")
    parser.add_argument("--devices", nargs="+", choices=DEVICES, default=list(DEVICES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planar-nodes", type=int)
    parser.add_argument(
        "--backend",
        choices=("reference", "node_planar", "lowrank_planar"),
    )
    parser.add_argument("--lowrank-rank", type=int)
    parser.add_argument("--output", default="reports/backend_validation.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cudnn.benchmark = True
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    rows = []
    for device_type in args.devices:
        if args.backend is not None:
            config[device_type]["conv_backend"] = args.backend
            config[device_type]["linear_backend"] = args.backend
        if (
            args.planar_nodes is not None
            and config[device_type].get("conv_backend")
            in {"node_planar", "lowrank_planar"}
        ):
            config[device_type]["planar_nodes"] = args.planar_nodes
        if args.lowrank_rank is not None:
            config[device_type]["lowrank_rank"] = args.lowrank_rank
        row = validate(device_type, config[device_type], args.seed)
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
