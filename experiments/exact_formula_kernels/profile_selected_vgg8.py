#!/usr/bin/env python3
"""Profile one VGG8 training step with the selected exact device backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
import yaml

from isa.device_sweeps.models.vgg8 import VGG8


ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("pcm", "fefet", "flash"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/device_sweeps/device_params.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    params = dict(yaml.safe_load(args.config.read_text(encoding="utf-8"))[args.device])
    transistor = args.device in {"fefet", "flash"}
    v_min, v_max = ((0.0, 4.0) if transistor else (-0.5, 0.5))
    torch.manual_seed(20260808)
    x = torch.empty(args.batch_size, 3, 32, 32, device="cuda").uniform_(v_min, v_max)
    labels = torch.randint(0, 10, (args.batch_size,), device="cuda")
    model = VGG8(
        args.device,
        params,
        tia_r=1.0e5,
        w_init_max=0.02 if args.device == "pcm" else 0.1,
        v_min=v_min,
        v_max=v_max,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, fused=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    samples = []
    with torch.cuda.nvtx.range(f"selected_vgg8/{args.device}/train_step"):
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    payload = {
        "device": args.device,
        "cuda_device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "selected_params": params,
        "step_ms": samples,
        "median_step_ms": statistics.median(samples),
        "final_loss": float(loss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
