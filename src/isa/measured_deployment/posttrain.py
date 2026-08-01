#!/usr/bin/env python3
"""Post-train Physical ViT Vth with a fixed measured-codebook STE forward.

The verify-write method is fixed: one global measured codebook, Q-point
distribution-pairwise assignment, fixed lambda_cm, and measured-LUT inference.
Only the continuous fc1 Vth parameters move.  The discrete assignment is
recomputed from the current Vth at epoch boundaries with the same fixed rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[3]

from isa.device_models.flash_transistor import ekv_current
from isa.measured_deployment.codebook import (
    IVLibrary,
    _compute_distribution_pairwise_indices,
    calibrate_voltages,
    distribution_quadrature,
)
from isa.measured_deployment.operator import (
    iter_deployment_fc1,
    replace_physical_fc1,
    set_fc1_forward_mode,
)
from isa.vision.config import ModelConfig, apply_preset
from isa.vision.data import get_dataloaders
from isa.vision.models import build_model
from isa.vision.scheduler import WarmupCosineScheduler
from isa.vision.utils import clamp_all_vth, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed verify-write + measured-LUT STE post-training"
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "cifar100"))
    parser.add_argument(
        "--init-checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "cifar100_300ep"
            / "small_physical_vit"
            / "physical_vit"
            / "best_checkpoint.pth"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default="auto", help="auto, empty, or checkpoint path")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-scale", default="small", choices=("tiny", "mid", "small", "base"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--digital-lr",
        type=float,
        default=0.0,
        help="Optional LR for non-Vth trainable parameters; 0 uses --lr for all groups",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument(
        "--curve-weight",
        type=float,
        default=0.0,
        help="Weight for normalized continuous-vs-measured curve consistency loss",
    )
    parser.add_argument(
        "--assignment-refresh-epochs",
        type=int,
        default=1,
        help="0 freezes assignment during training; N refreshes every N epochs and at final validation",
    )
    parser.add_argument(
        "--diagnose-ste",
        action="store_true",
        help="Run zero-step checks plus one hard-STE epoch, write diagnostic.json, then exit",
    )
    parser.add_argument(
        "--train-forward",
        default="ste",
        choices=("continuous", "measured", "ste"),
        help=(
            "continuous is the ordinary-post-training control; measured uses the "
            "fixed measured assignment with no fc1 surrogate gradient; ste uses "
            "the continuous EKV surrogate"
        ),
    )
    parser.add_argument(
        "--trainable",
        default="fc1_vth",
        choices=(
            "fc1_vth",
            "fc1_vth_ln_fc2",
            "fc1_vth_ln_fc2_head",
            "fc2_ln_head",
            "attn_fc2_ln_head",
        ),
    )
    parser.add_argument(
        "--fixed-assignment",
        default="",
        help=(
            "Optional activation-aware assignment .pt file. When set, its "
            "pos/neg state indices are loaded after codebook configuration and "
            "Q-point reassignment must remain disabled"
        ),
    )
    parser.add_argument(
        "--codebook-excel",
        default=str(ROOT / "global_codebook_6train2test.xlsx"),
    )
    parser.add_argument("--codebook-sheet", default="cb_all8")
    parser.add_argument("--measured-current-scale", type=float, default=1e-9)
    parser.add_argument("--min-valid-current-na", type=float, default=1.0)
    parser.add_argument("--assignment-q", type=int, default=9)
    parser.add_argument("--lambda-cm", type=float, default=0.0)
    parser.add_argument("--assignment-chunk-devices", type=int, default=8192)
    parser.add_argument(
        "--write-verify-voltage",
        type=float,
        default=4.1,
        help="Fixed hardware audit setting; Q-point pair assignment itself uses Vin quantiles",
    )
    parser.add_argument("--vin-lut-bins", type=int, default=0, help="0 = exact measured segments")
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument("--calib-hist-bins", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return True, rank, local_rank, world_size, device


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def broadcast_object(value: Any, rank: int, distributed: bool) -> Any:
    if not distributed:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def reduce_sums(values: Sequence[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def model_config(scale: str) -> ModelConfig:
    cfg = ModelConfig(
        img_size=32,
        patch_size=4,
        depth=12,
        drop_rate=0.0,
        drop_path_rate=0.05,
        voltage_max=4.0,
    )
    apply_preset(cfg, scale)
    return cfg


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("model", checkpoint)


def resolve_resume(args: argparse.Namespace) -> Path | None:
    text = args.resume.strip()
    if not text:
        return None
    if text == "auto":
        candidate = Path(args.output) / "last_checkpoint.pth"
        return candidate if candidate.exists() else None
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def build_student(
    cfg: ModelConfig, source_checkpoint: Path, device: torch.device
) -> nn.Module:
    model = build_model("physical_vit", cfg)
    model.load_state_dict(checkpoint_state(source_checkpoint), strict=True)
    model.to(device)
    replace_physical_fc1(model)
    return model


def build_teacher(cfg: ModelConfig, checkpoint: Path, device: torch.device) -> nn.Module:
    teacher = build_model("physical_vit", cfg)
    teacher.load_state_dict(checkpoint_state(checkpoint), strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def configure_trainable(model: nn.Module, mode: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names: list[str] = []
    train_fc1 = mode in ("fc1_vth", "fc1_vth_ln_fc2", "fc1_vth_ln_fc2_head")
    train_fc2_ln = mode in (
        "fc1_vth_ln_fc2",
        "fc1_vth_ln_fc2_head",
        "fc2_ln_head",
        "attn_fc2_ln_head",
    )
    train_head = mode in (
        "fc1_vth_ln_fc2_head",
        "fc2_ln_head",
        "attn_fc2_ln_head",
    )
    train_attention = mode == "attn_fc2_ln_head"
    for block_id, block in enumerate(model.blocks):
        if train_fc1:
            for branch in ("vth_pos", "vth_neg"):
                parameter = getattr(block.mlp.fc1, branch)
                parameter.requires_grad_(True)
                names.append(f"blocks.{block_id}.mlp.fc1.{branch}")
        if train_fc2_ln:
            for norm_name in ("norm1", "norm2"):
                for param_name, parameter in getattr(block, norm_name).named_parameters():
                    parameter.requires_grad_(True)
                    names.append(f"blocks.{block_id}.{norm_name}.{param_name}")
            # Physical fc2 remains CIM.  Only its Vth is allowed to move; TIA is fixed.
            for branch in ("vth_pos", "vth_neg"):
                parameter = getattr(block.mlp.fc2, branch)
                parameter.requires_grad_(True)
                names.append(f"blocks.{block_id}.mlp.fc2.{branch}")
        if train_attention:
            for param_name, parameter in block.attn.named_parameters():
                parameter.requires_grad_(True)
                names.append(f"blocks.{block_id}.attn.{param_name}")
    if train_head:
        for param_name, parameter in model.norm.named_parameters():
            parameter.requires_grad_(True)
            names.append(f"norm.{param_name}")
        for param_name, parameter in model.head.named_parameters():
            parameter.requires_grad_(True)
            names.append(f"head.{param_name}")
    return names


def configure_codebook_on_model(
    model: nn.Module, library: IVLibrary, vin_lut_bins: int
) -> None:
    for _, layer in iter_deployment_fc1(model):
        layer.configure_codebook(library.vg, library.currents_a, vin_lut_bins)


def calibration_payload(
    model: nn.Module,
    loader: Iterable,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_fc1_forward_mode(model, "continuous")
    model.eval()
    rows, accumulators = calibrate_voltages(
        model, loader, args.calib_batches, args.calib_hist_bins, device
    )
    points: dict[str, list[float]] = {}
    weights: dict[str, list[float]] = {}
    for block_id in range(len(model.blocks)):
        layer_points, layer_weights, _ = distribution_quadrature(
            accumulators[block_id], args.assignment_q
        )
        points[str(block_id)] = layer_points
        weights[str(block_id)] = layer_weights
    # calibrate_voltages is intentionally wrapped in inference_mode.  If that
    # pass is the first EKV-LUT use on this GPU, the module-level Triton cache
    # would otherwise retain inference tensors that cannot be saved by the
    # later autograd Function.  Rebuild the tiny LUT once in normal grad mode.
    from models import ekv_lut

    ekv_lut._device_cache.clear()
    return {"input_stats": rows, "points": points, "weights": weights}


def codebook_state_spacing(library: IVLibrary, target_current_a: float = 100e-9) -> float:
    coords: list[float] = []
    vg = library.vg.cpu()
    for curve in library.currents_a.cpu():
        clean = torch.cummax(curve, dim=0).values
        hits = torch.nonzero(clean >= target_current_a, as_tuple=False).flatten()
        if hits.numel() == 0 or int(hits[0]) == 0:
            continue
        upper = int(hits[0])
        lower = upper - 1
        y0 = float(clean[lower].item())
        y1 = float(clean[upper].item())
        if y1 <= y0:
            continue
        alpha = (target_current_a - y0) / (y1 - y0)
        coords.append(float(vg[lower].item() + alpha * (vg[upper] - vg[lower]).item()))
    if len(coords) < 2:
        return 0.1
    values = torch.tensor(sorted(coords))
    steps = values[1:] - values[:-1]
    positive = steps[steps > 1e-6]
    return float(positive.median().item()) if positive.numel() else 0.1


@torch.no_grad()
def refresh_assignments(
    model: nn.Module,
    calibration: dict[str, Any],
    library: IVLibrary,
    args: argparse.Namespace,
    rank: int,
    distributed: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_id, layer in iter_deployment_fc1(model):
        shape = (layer.out_features, layer.in_features)
        if rank == 0:
            points = torch.tensor(
                calibration["points"][str(block_id)],
                device=layer.vth_pos.device,
                dtype=torch.float32,
            )
            weights = torch.tensor(
                calibration["weights"][str(block_id)],
                device=layer.vth_pos.device,
                dtype=torch.float32,
            )
            weights = weights / weights.sum()
            flat_pos = layer.vth_pos.detach().float().flatten()
            flat_neg = layer.vth_neg.detach().float().flatten()
            target_pos = ekv_current(points[:, None], flat_pos[None, :], layer.cfg)
            target_neg = ekv_current(points[:, None], flat_neg[None, :], layer.cfg)
            measured_grid = torch.stack(
                [layer.currents_at(point) for point in points], dim=0
            )
            pos, neg = _compute_distribution_pairwise_indices(
                target_pos,
                target_neg,
                measured_grid,
                weights,
                library.i_floor_a,
                args.lambda_cm,
                args.assignment_chunk_devices,
            )
            selected = measured_grid[:, pos] - measured_grid[:, neg]
            target = target_pos - target_neg
            weighted_abs = (
                weights * (selected - target).abs().mean(dim=1)
            ).sum()
            target_abs = (weights * target.abs().mean(dim=1)).sum()
            row = {
                "block": block_id,
                "pair_count": int(pos.numel()),
                "same_pair_ratio": float((pos == neg).float().mean().item()),
                "relative_diff_error": float(
                    weighted_abs.item() / max(target_abs.item(), library.i_floor_a)
                ),
                "used_state_count": int(torch.unique(torch.cat((pos, neg))).numel()),
            }
            # NCCL in the deployment environment does not support int16
            # collectives. Broadcast int32, then compress inside set_assignment.
            pos_idx = pos.reshape(shape).to(torch.int32)
            neg_idx = neg.reshape(shape).to(torch.int32)
        else:
            pos_idx = torch.empty(shape, device=layer.vth_pos.device, dtype=torch.int32)
            neg_idx = torch.empty_like(pos_idx)
            row = {}
        if distributed:
            dist.broadcast(pos_idx, src=0)
            dist.broadcast(neg_idx, src=0)
        layer.set_assignment(pos_idx, neg_idx)
        if rank == 0:
            rows.append(row)
    barrier(distributed)
    return rows


@torch.no_grad()
def load_fixed_assignments(
    model: nn.Module,
    assignment_path: Path,
    rank: int,
    distributed: bool,
) -> list[dict[str, Any]]:
    payload = torch.load(assignment_path, map_location="cpu") if rank == 0 else None
    rows: list[dict[str, Any]] = []
    for block_id, layer in iter_deployment_fc1(model):
        shape = (layer.out_features, layer.in_features)
        if rank == 0:
            pos_key = f"blocks.{block_id}.pos_idx"
            neg_key = f"blocks.{block_id}.neg_idx"
            if pos_key not in payload or neg_key not in payload:
                raise KeyError(
                    f"Fixed assignment is missing {pos_key!r} or {neg_key!r}"
                )
            pos_idx = payload[pos_key].to(
                device=layer.vth_pos.device, dtype=torch.int32
            )
            neg_idx = payload[neg_key].to(
                device=layer.vth_pos.device, dtype=torch.int32
            )
            if tuple(pos_idx.shape) != shape or tuple(neg_idx.shape) != shape:
                raise ValueError(
                    f"Fixed assignment block {block_id} has shape "
                    f"{tuple(pos_idx.shape)}/{tuple(neg_idx.shape)}, expected {shape}"
                )
        else:
            pos_idx = torch.empty(
                shape, device=layer.vth_pos.device, dtype=torch.int32
            )
            neg_idx = torch.empty_like(pos_idx)
        if distributed:
            dist.broadcast(pos_idx, src=0)
            dist.broadcast(neg_idx, src=0)
        layer.set_assignment(pos_idx, neg_idx)
        if rank == 0:
            rows.append(
                {
                    "block": block_id,
                    "source": str(assignment_path.resolve()),
                    "pair_count": int(pos_idx.numel()),
                    "same_pair_ratio": float(
                        (pos_idx == neg_idx).float().mean().item()
                    ),
                    "used_state_count": int(
                        torch.unique(torch.cat((pos_idx.flatten(), neg_idx.flatten()))).numel()
                    ),
                }
            )
    barrier(distributed)
    return rows


def anchor_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (".vth_pos" in name or ".vth_neg" in name)
    }


def anchor_to_device(
    snapshot: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device=device, dtype=torch.float32) for name, tensor in snapshot.items()}


def normalized_anchor_loss(
    model: nn.Module, anchor: dict[str, torch.Tensor], spacing: float
) -> torch.Tensor:
    total = None
    count = 0
    denominator = max(float(spacing) ** 2, 1e-8)
    for name, parameter in model.named_parameters():
        if name not in anchor:
            continue
        value = (parameter.float() - anchor[name]).square().sum() / denominator
        total = value if total is None else total + value
        count += parameter.numel()
    if total is None:
        raise RuntimeError("No fc1 Vth parameters matched the anchor snapshot")
    return total / max(count, 1)


def normalized_curve_consistency_loss(
    model: nn.Module,
    calibration: dict[str, Any],
) -> torch.Tensor:
    total = None
    layers = 0
    for block_id, layer in iter_deployment_fc1(model):
        if layer.pos_idx.numel() == 0:
            raise RuntimeError("Assignment must be initialized before curve loss")
        points = torch.tensor(
            calibration["points"][str(block_id)],
            device=layer.vth_pos.device,
            dtype=torch.float32,
        )
        weights = torch.tensor(
            calibration["weights"][str(block_id)],
            device=layer.vth_pos.device,
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        flat_pos = layer.vth_pos.float().flatten()
        flat_neg = layer.vth_neg.float().flatten()
        d_cont = ekv_current(points[:, None], flat_pos[None, :], layer.cfg) - ekv_current(
            points[:, None], flat_neg[None, :], layer.cfg
        )
        with torch.no_grad():
            measured_grid = torch.stack([layer.currents_at(point) for point in points], dim=0)
            pos = layer.pos_idx.long().flatten()
            neg = layer.neg_idx.long().flatten()
            d_meas = measured_grid[:, pos] - measured_grid[:, neg]
            denom = (weights[:, None] * d_meas.square()).sum() / max(d_meas.shape[1], 1)
            floor = layer.measured_currents.abs().mean().square() * 1e-4 + 1e-24
            denom = denom.clamp_min(floor)
        layer_loss = (weights[:, None] * (d_cont - d_meas).square()).sum() / max(
            d_cont.shape[1], 1
        )
        layer_loss = layer_loss / denom.detach()
        total = layer_loss if total is None else total + layer_loss
        layers += 1
    if total is None:
        raise RuntimeError("No deployment-aware fc1 layers found for curve loss")
    return total / max(layers, 1)


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student.float() / t, dim=1),
        F.softmax(teacher.float() / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def optimizer_parameters(
    model: nn.Module, args: argparse.Namespace
) -> list[nn.Parameter] | list[dict[str, Any]]:
    if args.digital_lr <= 0 or math.isclose(args.digital_lr, args.lr):
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    vth_parameters: list[nn.Parameter] = []
    digital_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".vth_pos" in name or ".vth_neg" in name:
            vth_parameters.append(parameter)
        else:
            digital_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if vth_parameters:
        groups.append({"params": vth_parameters, "lr": args.lr})
    if digital_parameters:
        groups.append({"params": digital_parameters, "lr": args.digital_lr})
    return groups


@torch.no_grad()
def vth_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if ".mlp.fc1.vth_pos" in name or ".mlp.fc1.vth_neg" in name
    }


@torch.no_grad()
def vth_change_summary(
    model: nn.Module, before: dict[str, torch.Tensor]
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for block_id, layer in iter_deployment_fc1(model):
        values = []
        for branch in ("vth_pos", "vth_neg"):
            name = f"blocks.{block_id}.mlp.fc1.{branch}"
            delta = getattr(layer, branch).detach().cpu().float() - before[name].float()
            values.append(delta.flatten())
        merged = torch.cat(values)
        rows.append(
            {
                "block": block_id,
                "mean_abs_delta_vth": float(merged.abs().mean().item()),
                "max_abs_delta_vth": float(merged.abs().max().item()),
                "rms_delta_vth": float(merged.square().mean().sqrt().item()),
            }
        )
    return rows


@torch.no_grad()
def assignment_snapshot(model: nn.Module) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "block": block_id,
            "pos": layer.pos_idx.detach().cpu().clone(),
            "neg": layer.neg_idx.detach().cpu().clone(),
        }
        for block_id, layer in iter_deployment_fc1(model)
    ]


@torch.no_grad()
def assignment_change_summary(
    before: list[dict[str, torch.Tensor]], model: nn.Module
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for old, (block_id, layer) in zip(before, iter_deployment_fc1(model)):
        pos_before = old["pos"].to(layer.pos_idx.device)
        neg_before = old["neg"].to(layer.neg_idx.device)
        pos_change = layer.pos_idx != pos_before
        neg_change = layer.neg_idx != neg_before
        pair_change = pos_change | neg_change
        rows.append(
            {
                "block": block_id,
                "pos_index_change_ratio": float(pos_change.float().mean().item()),
                "neg_index_change_ratio": float(neg_change.float().mean().item()),
                "pair_change_ratio": float(pair_change.float().mean().item()),
                "used_state_count": int(
                    torch.unique(torch.cat((layer.pos_idx.flatten(), layer.neg_idx.flatten()))).numel()
                ),
            }
        )
    return rows


@torch.no_grad()
def max_logit_difference(
    model: nn.Module,
    raw_model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> float:
    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    model.eval()
    set_fc1_forward_mode(raw_model, "measured")
    measured = model(images).detach().float()
    set_fc1_forward_mode(raw_model, "ste")
    ste = model(images).detach().float()
    return float((ste - measured).abs().max().item())


@torch.no_grad()
def fc1_output_discrepancy(
    model: nn.Module,
    raw_model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> list[dict[str, float]]:
    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    model.eval()

    def collect(mode: str) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        hooks = []
        for block_id, layer in iter_deployment_fc1(raw_model):
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, bid=block_id: outputs.__setitem__(
                        bid, output.detach().float().flatten()
                    )
                )
            )
        set_fc1_forward_mode(raw_model, mode)
        _ = model(images)
        for hook in hooks:
            hook.remove()
        return outputs

    continuous = collect("continuous")
    measured = collect("measured")
    rows: list[dict[str, float]] = []
    for block_id in sorted(continuous):
        c = continuous[block_id]
        m = measured[block_id]
        diff = m - c
        denom = c.norm().clamp_min(1e-12)
        cosine = F.cosine_similarity(c, m, dim=0, eps=1e-12)
        rows.append(
            {
                "block": block_id,
                "relative_l2_measured_minus_continuous": float((diff.norm() / denom).item()),
                "cosine_measured_continuous": float(cosine.item()),
            }
        )
    return rows


def train_one_epoch(
    model: nn.Module,
    raw_model: nn.Module,
    teacher: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    anchor: dict[str, torch.Tensor],
    spacing: float,
    calibration: dict[str, Any],
    epoch: int,
    rank: int,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    set_fc1_forward_mode(raw_model, args.train_forward)
    sums = {
        "loss": 0.0,
        "ce": 0.0,
        "kd": 0.0,
        "anchor": 0.0,
        "curve": 0.0,
        "correct": 0.0,
        "n": 0.0,
    }
    optimizer_steps = 0
    for step, (images, targets) in enumerate(loader):
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(enabled=scaler is not None):
            teacher_logits = teacher(images)
        with autocast(enabled=scaler is not None):
            logits = model(images)
            ce = F.cross_entropy(
                logits, targets, label_smoothing=args.label_smoothing
            )
            kd = distillation_loss(logits, teacher_logits, args.kd_temperature)
            anchor_loss = normalized_anchor_loss(raw_model, anchor, spacing)
            if args.curve_weight > 0:
                curve_loss = normalized_curve_consistency_loss(raw_model, calibration)
            else:
                curve_loss = logits.new_zeros(())
            loss = (
                ce
                + args.kd_weight * kd
                + args.anchor_weight * anchor_loss
                + args.curve_weight * curve_loss
            )
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad], args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            # A scale decrease means non-finite gradients and a skipped step.
            if scaler.get_scale() >= scale_before:
                optimizer_steps += 1
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad], args.grad_clip
                )
            optimizer.step()
            optimizer_steps += 1
        clamp_all_vth(raw_model)
        batch = targets.numel()
        sums["loss"] += float(loss.detach().item()) * batch
        sums["ce"] += float(ce.detach().item()) * batch
        sums["kd"] += float(kd.detach().item()) * batch
        sums["anchor"] += float(anchor_loss.detach().item()) * batch
        sums["curve"] += float(curve_loss.detach().item()) * batch
        sums["correct"] += float((logits.detach().argmax(1) == targets).sum().item())
        sums["n"] += batch
        if rank == 0 and (step + 1) % args.print_freq == 0:
            print(
                f"epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={sums['loss']/max(sums['n'],1):.4f} "
                f"acc={100*sums['correct']/max(sums['n'],1):.2f}%",
                flush=True,
            )
    reduced = reduce_sums(
        [sums[key] for key in ("loss", "ce", "kd", "anchor", "curve", "correct", "n")],
        device,
    )
    loss_sum, ce_sum, kd_sum, anchor_sum, curve_sum, correct, count = reduced
    count = max(count, 1.0)
    return {
        "loss": loss_sum / count,
        "ce": ce_sum / count,
        "kd": kd_sum / count,
        "anchor": anchor_sum / count,
        "curve": curve_sum / count,
        "accuracy": 100.0 * correct / count,
        "optimizer_steps": float(optimizer_steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    raw_model: nn.Module,
    loader: Iterable,
    device: torch.device,
    mode: str,
    use_amp: bool,
    max_batches: int,
) -> dict[str, float]:
    set_fc1_forward_mode(raw_model, mode)
    model.eval()
    loss_sum = correct = count = 0.0
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        batch = targets.numel()
        loss_sum += float(loss.item()) * batch
        correct += float((logits.argmax(1) == targets).sum().item())
        count += batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    loss_sum, correct, count, elapsed = reduce_sums(
        [loss_sum, correct, count, elapsed], device
    )
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    wall = elapsed / world
    return {
        "loss": loss_sum / max(count, 1.0),
        "accuracy": 100.0 * correct / max(count, 1.0),
        "images": count,
        "seconds": wall,
        "images_per_second": count / max(wall, 1e-9),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def refresh_due_before_epoch(epoch: int, start_epoch: int, frequency: int) -> bool:
    if frequency <= 0:
        return False
    return (epoch - start_epoch) % frequency == 0


def refresh_due_after_epoch(epoch: int, total_epochs: int, frequency: int) -> bool:
    if frequency <= 0:
        return epoch + 1 >= total_epochs
    return (epoch + 1) % frequency == 0 or epoch + 1 >= total_epochs


def run_ste_diagnostic(
    model: nn.Module,
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    anchor: dict[str, torch.Tensor],
    spacing: float,
    calibration: dict[str, Any],
    library: IVLibrary,
    out_dir: Path,
    rank: int,
    distributed: bool,
) -> None:
    original_forward = args.train_forward
    args.train_forward = "ste"
    continuous_before = evaluate(
        model, student, val_loader, device, "continuous", scaler is not None, args.max_val_batches
    )
    measured_before = evaluate(
        model, student, val_loader, device, "measured", scaler is not None, args.max_val_batches
    )
    ste_before = evaluate(
        model, student, val_loader, device, "ste", scaler is not None, args.max_val_batches
    )
    max_ste_measured_logit_diff = max_logit_difference(model, student, val_loader, device)
    output_discrepancy_before = fc1_output_discrepancy(model, student, val_loader, device)
    assignment_before = assignment_snapshot(student)
    vth_before = vth_snapshot(student)

    train_metrics = train_one_epoch(
        model,
        student,
        teacher,
        train_loader,
        optimizer,
        scaler,
        device,
        args,
        anchor,
        spacing,
        calibration,
        0,
        rank,
    )
    assignment_rows_after = refresh_assignments(
        student, calibration, library, args, rank, distributed
    )
    continuous_after = evaluate(
        model, student, val_loader, device, "continuous", scaler is not None, args.max_val_batches
    )
    measured_after = evaluate(
        model, student, val_loader, device, "measured", scaler is not None, args.max_val_batches
    )
    ste_after = evaluate(
        model, student, val_loader, device, "ste", scaler is not None, args.max_val_batches
    )
    output_discrepancy_after = fc1_output_discrepancy(model, student, val_loader, device)
    vth_change = vth_change_summary(student, vth_before)
    assignment_change = assignment_change_summary(assignment_before, student)
    args.train_forward = original_forward

    if rank == 0:
        result = {
            "status": "completed",
            "mode": "ste_diagnostic_1epoch",
            "continuous_before": continuous_before,
            "measured_before": measured_before,
            "ste_before": ste_before,
            "max_ste_measured_logit_diff": max_ste_measured_logit_diff,
            "train_metrics": train_metrics,
            "continuous_after": continuous_after,
            "measured_after": measured_after,
            "ste_after": ste_after,
            "vth_change_rows": vth_change,
            "assignment_change_rows": assignment_change,
            "assignment_rows_after": assignment_rows_after,
            "fc1_output_discrepancy_before": output_discrepancy_before,
            "fc1_output_discrepancy_after": output_discrepancy_after,
        }
        save_json(out_dir / "diagnostic.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed + rank)
    use_amp = not args.no_amp and device.type == "cuda"
    out_dir = Path(args.output)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    resume_path = resolve_resume(args)
    resume_payload = torch.load(resume_path, map_location="cpu") if resume_path else None
    source = resume_path or Path(args.init_checkpoint)
    fixed_assignment_path = Path(args.fixed_assignment) if args.fixed_assignment else None
    if fixed_assignment_path is not None:
        if not fixed_assignment_path.exists():
            raise FileNotFoundError(
                f"Fixed assignment not found: {fixed_assignment_path}"
            )
        if args.assignment_refresh_epochs != 0:
            raise ValueError(
                "--fixed-assignment requires --assignment-refresh-epochs 0"
            )
        if args.train_forward != "measured" and not args.eval_only:
            raise ValueError(
                "--fixed-assignment training must use --train-forward measured"
            )
    cfg = model_config(args.model_scale)
    student = build_student(cfg, source, device)
    trainable_names = configure_trainable(student, args.trainable)

    library = IVLibrary.load(
        Path(args.codebook_excel),
        args.codebook_sheet,
        args.measured_current_scale,
        args.min_valid_current_na,
    )
    configure_codebook_on_model(student, library, args.vin_lut_bins)

    if fixed_assignment_path is not None:
        calibration = (
            resume_payload.get("calibration")
            if resume_payload and "calibration" in resume_payload
            else {
                "source": "fixed_activation_aware_assignment",
                "assignment_path": str(fixed_assignment_path.resolve()),
                "points": {},
                "weights": {},
            }
        )
    elif resume_payload and "calibration" in resume_payload:
        calibration = resume_payload["calibration"]
    else:
        calibration = None
        if rank == 0:
            calibration_loader, _ = get_dataloaders(
                args.data,
                batch_size=args.calib_batch_size,
                num_workers=args.num_workers,
                distributed=False,
            )
            print("Calibrating fixed Q-point Vin distributions...", flush=True)
            calibration = calibration_payload(student, calibration_loader, args, device)
        calibration = broadcast_object(calibration, rank, distributed)

    if resume_payload and "anchor_state" in resume_payload:
        anchor_cpu = resume_payload["anchor_state"]
    else:
        anchor_cpu = anchor_snapshot(student)
    anchor = anchor_to_device(anchor_cpu, device)
    spacing = float(
        resume_payload.get("anchor_state_spacing", codebook_state_spacing(library))
        if resume_payload
        else codebook_state_spacing(library)
    )

    if rank == 0:
        parameter_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(
            f"device={device} world_size={world_size} codebook={args.codebook_sheet} "
            f"states={library.n_curves} Q={args.assignment_q} lambda_cm={args.lambda_cm:g} "
            f"train_forward={args.train_forward} trainable={args.trainable} "
            f"trainable_parameters={parameter_count:,} anchor_spacing={spacing:.6f}V",
            flush=True,
        )
        save_json(
            out_dir / "fixed_method.json",
            {
                "codebook_excel": str(Path(args.codebook_excel).resolve()),
                "codebook_sheet": args.codebook_sheet,
                "state_count": library.n_curves,
                "measured_current_scale": args.measured_current_scale,
                "min_valid_current_na": args.min_valid_current_na,
                "assignment": (
                    "fixed_activation_aware"
                    if fixed_assignment_path is not None
                    else "distribution_pairwise"
                ),
                "fixed_assignment": (
                    str(fixed_assignment_path.resolve())
                    if fixed_assignment_path is not None
                    else None
                ),
                "Q": args.assignment_q,
                "lambda_cm": args.lambda_cm,
                "write_verify_voltage_V": args.write_verify_voltage,
                "vin_lut_bins": args.vin_lut_bins,
                "curve_weight": args.curve_weight,
                "vth_lr": args.lr,
                "digital_lr": args.digital_lr,
                "assignment_refresh_epochs": args.assignment_refresh_epochs,
                "diagnose_ste": args.diagnose_ste,
                "trainable_parameter_names": trainable_names,
                "frozen_voltage_mapping": True,
                "frozen_tia_gain": True,
            },
        )

    if fixed_assignment_path is not None:
        assignment_rows = load_fixed_assignments(
            student, fixed_assignment_path, rank, distributed
        )
    else:
        assignment_rows = refresh_assignments(
            student, calibration, library, args, rank, distributed
        )

    if distributed:
        model: nn.Module = DDP(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        model = student

    _, val_loader = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
    )

    if args.eval_only:
        continuous = evaluate(
            model, student, val_loader, device, "continuous", use_amp, args.max_val_batches
        )
        measured = evaluate(
            model, student, val_loader, device, "measured", use_amp, args.max_val_batches
        )
        if rank == 0:
            result = {
                "status": "completed",
                "mode": "eval_only",
                "checkpoint": str(source),
                "continuous": continuous,
                "measured": measured,
                "accuracy_drop": continuous["accuracy"] - measured["accuracy"],
                "assignment_rows": assignment_rows,
            }
            save_json(out_dir / "metrics.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    teacher = build_teacher(cfg, Path(args.init_checkpoint), device)
    train_loader, _ = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters(student, args), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    start_epoch = 0
    # Always materialize a best checkpoint after the first completed epoch,
    # even for a deliberately tiny smoke subset whose measured accuracy is 0.
    best_measured = -math.inf
    if resume_payload:
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        best_measured = float(resume_payload.get("best_measured_accuracy", 0.0))
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        if "scheduler" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler"])
        if scaler is not None and resume_payload.get("scaler"):
            scaler.load_state_dict(resume_payload["scaler"])
        if rank == 0:
            print(
                f"Resumed {resume_path} at epoch={start_epoch}, "
                f"best_measured={best_measured:.4f}%",
                flush=True,
            )

    if args.diagnose_ste:
        run_ste_diagnostic(
            model,
            student,
            teacher,
            train_loader,
            val_loader,
            optimizer,
            scaler,
            device,
            args,
            anchor,
            spacing,
            calibration,
            library,
            out_dir,
            rank,
            distributed,
        )
        if distributed:
            dist.destroy_process_group()
        return

    if start_epoch >= args.epochs:
        if rank == 0:
            print("Requested epochs already completed; running final evaluation only.", flush=True)
        continuous = evaluate(
            model, student, val_loader, device, "continuous", use_amp, args.max_val_batches
        )
        measured = evaluate(
            model, student, val_loader, device, "measured", use_amp, args.max_val_batches
        )
        if rank == 0:
            save_json(
                out_dir / "metrics.json",
                {
                    "status": "completed",
                    "epoch": start_epoch - 1,
                    "best_measured_accuracy": best_measured,
                    "continuous": continuous,
                    "measured": measured,
                },
            )
        if distributed:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        # Required semantics: fixed assignment rule, with a configurable refresh cadence.
        if (
            fixed_assignment_path is None
            and refresh_due_before_epoch(
                epoch, start_epoch, args.assignment_refresh_epochs
            )
        ):
            assignment_rows = refresh_assignments(
                student, calibration, library, args, rank, distributed
            )
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            student,
            teacher,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            anchor,
            spacing,
            calibration,
            epoch,
            rank,
        )
        if (
            fixed_assignment_path is None
            and refresh_due_after_epoch(
                epoch, args.epochs, args.assignment_refresh_epochs
            )
        ):
            assignment_rows = refresh_assignments(
                student, calibration, library, args, rank, distributed
            )
        continuous = evaluate(
            model, student, val_loader, device, "continuous", use_amp, args.max_val_batches
        )
        measured = evaluate(
            model, student, val_loader, device, "measured", use_amp, args.max_val_batches
        )
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        elif rank == 0:
            print(
                "WARNING: all AMP optimizer steps were skipped; scheduler was not advanced",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        is_best = measured["accuracy"] > best_measured
        best_measured = max(best_measured, measured["accuracy"])

        if rank == 0:
            row = {
                "epoch": epoch,
                "train_forward": args.train_forward,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_kd": train_metrics["kd"],
                "train_anchor": train_metrics["anchor"],
                "train_curve": train_metrics["curve"],
                "train_accuracy": train_metrics["accuracy"],
                "optimizer_steps": int(train_metrics["optimizer_steps"]),
                "continuous_val_accuracy": continuous["accuracy"],
                "measured_val_accuracy": measured["accuracy"],
                "best_measured_accuracy": best_measured,
                "lr": optimizer.param_groups[0]["lr"],
                "digital_lr": optimizer.param_groups[-1]["lr"],
                "epoch_seconds": elapsed,
            }
            append_csv(out_dir / "training_log.csv", row)
            payload = {
                "epoch": epoch,
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_measured_accuracy": best_measured,
                "args": vars(args),
                "calibration": calibration,
                "anchor_state": anchor_cpu,
                "anchor_state_spacing": spacing,
                "last_assignment_rows": assignment_rows,
            }
            atomic_torch_save(payload, out_dir / "last_checkpoint.pth")
            if is_best:
                atomic_torch_save(payload, out_dir / "best_checkpoint.pth")
            save_json(
                out_dir / "metrics.json",
                {
                    "status": "running" if epoch + 1 < args.epochs else "completed",
                    "epoch": epoch,
                    "best_measured_accuracy": best_measured,
                    "continuous": continuous,
                    "measured": measured,
                    "assignment_rows": assignment_rows,
                },
            )
            print(
                f"epoch={epoch}/{args.epochs-1} train={train_metrics['accuracy']:.2f}% "
                f"continuous={continuous['accuracy']:.2f}% "
                f"measured={measured['accuracy']:.2f}% best={best_measured:.2f}% "
                f"time={elapsed:.1f}s",
                flush=True,
            )

    if rank == 0:
        print(f"Training completed. Best measured accuracy={best_measured:.4f}%", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
