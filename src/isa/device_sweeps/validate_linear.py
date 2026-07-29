#!/usr/bin/env python3
"""Numerical parity checks for accelerated NVM linear backends."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from isa.kernels.device_sweep.custom_layers import NVM_Linear


DEVICES = ("pcm", "fefet", "flash")


def rms_relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (actual.float() - reference.float()).square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
            eps=1e-12,
        ).item()
    )


def make_layer(
    device_type: str,
    params: dict,
    backend: str,
    in_features: int,
    out_features: int,
) -> NVM_Linear:
    local = copy.deepcopy(params)
    local["linear_backend"] = backend
    if backend == "reference":
        local["lut_enabled"] = False
        local["planar_enabled"] = False
    return NVM_Linear(
        in_features,
        out_features,
        device_type=device_type,
        device_params=local,
        w_init_max=0.1,
    ).cuda()


def validate(
    device_type: str,
    params: dict,
    backend: str,
    seed: int,
    batch_size: int,
    in_features: int,
    out_features: int,
) -> dict:
    torch.manual_seed(seed)
    reference = make_layer(
        device_type, params, "reference", in_features, out_features
    )
    accelerated = make_layer(
        device_type, params, backend, in_features, out_features
    )
    lo = float(params["w_min"])
    hi = float(params["w_max"])
    with torch.no_grad():
        theta_pos = torch.empty_like(reference.theta_pos).uniform_(lo, hi)
        theta_neg = torch.empty_like(reference.theta_neg).uniform_(lo, hi)
        reference.theta_pos.copy_(theta_pos)
        reference.theta_neg.copy_(theta_neg)
        accelerated.theta_pos.copy_(theta_pos)
        accelerated.theta_neg.copy_(theta_neg)

    voltage_range = (
        (-4.0, 4.0)
        if device_type in {"fefet", "flash"}
        else (-0.5, 0.5)
    )
    x = torch.empty(
        batch_size, in_features, device="cuda"
    ).uniform_(*voltage_range)
    x_reference = x.detach().requires_grad_(True)
    x_accelerated = x.detach().requires_grad_(True)
    probe = torch.randn(batch_size, out_features, device="cuda")

    y_reference = reference(x_reference)
    loss_reference = (y_reference * probe).sum()
    gradients_reference = torch.autograd.grad(
        loss_reference,
        (x_reference, reference.theta_pos, reference.theta_neg),
    )
    y_accelerated = accelerated(x_accelerated)
    loss_accelerated = (y_accelerated * probe).sum()
    gradients_accelerated = torch.autograd.grad(
        loss_accelerated,
        (x_accelerated, accelerated.theta_pos, accelerated.theta_neg),
    )
    torch.cuda.synchronize()
    return {
        "device": device_type,
        "backend": backend,
        "planar_nodes": int(params.get("planar_nodes", 0)),
        "lowrank_rank": int(params.get("lowrank_rank", 0)),
        "forward_rms_relative": rms_relative(
            y_accelerated, y_reference
        ),
        "forward_max_absolute": float(
            (y_accelerated - y_reference).abs().max().item()
        ),
        "grad_x_rms_relative": rms_relative(
            gradients_accelerated[0], gradients_reference[0]
        ),
        "grad_x_cosine": cosine(
            gradients_accelerated[0], gradients_reference[0]
        ),
        "grad_pos_rms_relative": rms_relative(
            gradients_accelerated[1], gradients_reference[1]
        ),
        "grad_pos_cosine": cosine(
            gradients_accelerated[1], gradients_reference[1]
        ),
        "grad_neg_rms_relative": rms_relative(
            gradients_accelerated[2], gradients_reference[2]
        ),
        "grad_neg_cosine": cosine(
            gradients_accelerated[2], gradients_reference[2]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/device_params.yaml")
    parser.add_argument("--devices", nargs="+", choices=DEVICES, default=list(DEVICES))
    parser.add_argument(
        "--backend",
        choices=("lut", "node_planar", "lowrank_planar"),
        default="lowrank_planar",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--planar-nodes", type=int)
    parser.add_argument("--lowrank-rank", type=int)
    parser.add_argument("--output", default="reports/linear_backend_validation.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    rows = []
    for device_type in args.devices:
        params = copy.deepcopy(config[device_type])
        if args.planar_nodes is not None:
            params["planar_nodes"] = args.planar_nodes
        if args.lowrank_rank is not None:
            params["lowrank_rank"] = args.lowrank_rank
        row = validate(
            device_type,
            params,
            args.backend,
            args.seed,
            args.batch_size,
            args.in_features,
            args.out_features,
        )
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
