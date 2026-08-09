#!/usr/bin/env python3
"""Quick activation-aware measured assignment validation for Physical ViT fc1.

The hardware/program-verify path is unchanged: the script only changes the
offline choice of pos/neg measured state ids.  It first builds the usual Q-point
curve-matching Top-K candidates, then runs a residual-aware coordinate pass on
real calibration fc1 inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

from isa.device_models.flash_transistor import ekv_current
from isa.measured_deployment.codebook import IVLibrary
from isa.measured_deployment.operator import (
    iter_deployment_fc1,
    set_fc1_forward_mode,
)
from isa.measured_deployment.posttrain import (
    build_student,
    configure_codebook_on_model,
    evaluate,
    model_config,
    save_json,
)
from isa.vision.data import get_dataloaders
from isa.vision.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-aware measured assignment quick eval")
    parser.add_argument("--data", default=str(ROOT / "data" / "cifar100"))
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "cifar100_300ep"
            / "small_physical_vit"
            / "physical_vit"
            / "best_checkpoint.pth"
        ),
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "activation_aware_assignment_quick"))
    parser.add_argument("--model-scale", default="small", choices=("tiny", "mid", "small", "base"))
    parser.add_argument("--codebook-excel", default=str(ROOT / "global_codebook_6train2test.xlsx"))
    parser.add_argument("--codebook-sheet", default="cb_all8")
    parser.add_argument("--measured-current-scale", type=float, default=1e-9)
    parser.add_argument("--min-valid-current-na", type=float, default=1.0)
    parser.add_argument("--assignment-q", type=int, default=9)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cm", type=float, default=0.0)
    parser.add_argument("--assignment-chunk-devices", type=int, default=8192)
    parser.add_argument("--calib-batches", type=int, default=2)
    parser.add_argument("--calib-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens-per-layer", type=int, default=512)
    parser.add_argument("--coord-block", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vin-lut-bins", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-baseline-eval", action="store_true")
    parser.add_argument(
        "--qpoint-only",
        action="store_true",
        help=(
            "Save and evaluate the independent Q-point Top-1 assignment, then "
            "exit before activation-aware coordinate descent"
        ),
    )
    return parser.parse_args()


def save_assignment(path: Path, model: torch.nn.Module, rows: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {"rows": rows}
    for block_id, layer in iter_deployment_fc1(model):
        payload[f"blocks.{block_id}.pos_idx"] = layer.pos_idx.detach().cpu()
        payload[f"blocks.{block_id}.neg_idx"] = layer.neg_idx.detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def capture_fc1_inputs(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    batches: int,
    max_tokens: int,
) -> dict[int, torch.Tensor]:
    set_fc1_forward_mode(model, "continuous")
    model.eval()
    chunks: dict[int, list[torch.Tensor]] = {i: [] for i, _ in iter_deployment_fc1(model)}
    counts: dict[int, int] = {i: 0 for i, _ in iter_deployment_fc1(model)}
    handles = []

    def make_hook(block_id: int):
        def hook(_module, inputs):
            if counts[block_id] >= max_tokens:
                return
            flat = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            need = max_tokens - counts[block_id]
            take = min(need, flat.shape[0])
            if take > 0:
                chunks[block_id].append(flat[:take].cpu())
                counts[block_id] += take

        return hook

    for block_id, layer in iter_deployment_fc1(model):
        handles.append(layer.register_forward_pre_hook(make_hook(block_id)))
    try:
        for batch_id, (images, _) in enumerate(loader):
            if batch_id >= batches:
                break
            model(images.to(device, non_blocking=True))
            print(f"  captured calibration batch {batch_id + 1}/{batches}", flush=True)
            if all(count >= max_tokens for count in counts.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for block_id in counts:
        if not chunks[block_id]:
            raise RuntimeError(f"No calibration inputs captured for block {block_id}")
        result[block_id] = torch.cat(chunks[block_id], dim=0)[:max_tokens].contiguous()
    return result


@torch.no_grad()
def quantile_points(x: torch.Tensor, q: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.tensor([(i + 0.5) / q for i in range(q)], device=device, dtype=torch.float32)
    values = torch.quantile(x.to(device=device, dtype=torch.float32).flatten(), probs).unique(sorted=True)
    weights = torch.full((values.numel(),), 1.0 / max(values.numel(), 1), device=device)
    return values, weights


@torch.no_grad()
def qpoint_topk_candidates(
    layer,
    x_calib: torch.Tensor,
    library: IVLibrary,
    q: int,
    topk: int,
    lambda_cm: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    points, weights = quantile_points(x_calib, q, device)
    flat_pos = layer.vth_pos.detach().float().flatten()
    flat_neg = layer.vth_neg.detach().float().flatten()
    target_pos = ekv_current(points[:, None], flat_pos[None, :], layer.cfg)
    target_neg = ekv_current(points[:, None], flat_neg[None, :], layer.cfg)
    measured_grid = torch.stack([layer.currents_at(point) for point in points], dim=0)
    state_count = measured_grid.shape[1]
    pos_pair_ids = torch.arange(state_count, device=device).repeat_interleave(state_count)
    neg_pair_ids = torch.arange(state_count, device=device).repeat(state_count)
    pair_count = pos_pair_ids.numel()
    keep = min(topk, pair_count)
    scale = max(
        float(measured_grid.abs().max().item()),
        float(target_pos.abs().max().item()),
        float(target_neg.abs().max().item()),
        float(library.i_floor_a),
    )
    sqrt_w = torch.sqrt(weights.float() / weights.sum()).view(1, -1)
    pair_diff = ((measured_grid[:, pos_pair_ids] - measured_grid[:, neg_pair_ids]).t() / scale)
    pair_diff_w = pair_diff * sqrt_w
    pair_diff_norm = pair_diff_w.square().sum(dim=1)
    target_diff = ((target_pos - target_neg).t() / scale).contiguous()
    if lambda_cm != 0:
        pair_sum = ((measured_grid[:, pos_pair_ids] + measured_grid[:, neg_pair_ids]).t() / scale)
        pair_sum_w = pair_sum * sqrt_w
        pair_sum_norm = pair_sum_w.square().sum(dim=1)
        target_sum = ((target_pos + target_neg).t() / scale).contiguous()
    n_weights = target_diff.shape[0]
    top_pos = torch.empty(n_weights, keep, dtype=torch.int16, device=device)
    top_neg = torch.empty_like(top_pos)
    best_loss_sum = 0.0
    for start in range(0, n_weights, chunk_size):
        end = min(start + chunk_size, n_weights)
        target_diff_w = target_diff[start:end] * sqrt_w
        loss = pair_diff_norm.unsqueeze(0) - 2.0 * (target_diff_w @ pair_diff_w.t())
        if lambda_cm != 0:
            target_sum_w = target_sum[start:end] * sqrt_w
            common_loss = pair_sum_norm.unsqueeze(0) - 2.0 * (target_sum_w @ pair_sum_w.t())
            loss.add_(common_loss, alpha=lambda_cm)
        values, ids = torch.topk(loss, k=keep, largest=False, dim=1)
        best_loss_sum += float(values[:, 0].sum().item())
        top_pos[start:end] = pos_pair_ids[ids].to(torch.int16)
        top_neg[start:end] = neg_pair_ids[ids].to(torch.int16)
    shape = (layer.out_features, layer.in_features, keep)
    row = {
        "q_points": [float(v) for v in points.detach().cpu().tolist()],
        "topk": keep,
        "mean_top1_curve_loss": best_loss_sum / max(n_weights, 1),
    }
    return top_pos.reshape(shape).cpu(), top_neg.reshape(shape).cpu(), row


@torch.no_grad()
def measured_values_by_input(layer, x_calib: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x_calib.to(device=device, dtype=torch.float32).clamp(layer.lookup_min, layer.lookup_max)
    values = []
    for input_id in range(layer.in_features):
        values.append(layer.currents_at(x[:, input_id]).t().contiguous())
    return torch.stack(values, dim=0)


@torch.no_grad()
def optimize_layer_assignment(
    block_id: int,
    layer,
    x_calib: torch.Tensor,
    cand_pos_cpu: torch.Tensor,
    cand_neg_cpu: torch.Tensor,
    sweeps: int,
    coord_block: int,
    min_improvement: float,
    device: torch.device,
) -> dict[str, Any]:
    cand_pos = cand_pos_cpu.to(device=device, dtype=torch.long)
    cand_neg = cand_neg_cpu.to(device=device, dtype=torch.long)
    initial_pos = cand_pos[:, :, 0].to(torch.int32)
    initial_neg = cand_neg[:, :, 0].to(torch.int32)
    layer.set_assignment(initial_pos, initial_neg)

    x = x_calib.to(device=device, dtype=torch.float32)
    layer.set_forward_mode("continuous")
    y_cont = layer(x.unsqueeze(0)).squeeze(0).float()
    y_meas_initial = layer.measured_forward(x.unsqueeze(0)).squeeze(0).float()
    residual_initial = (y_meas_initial - y_cont).t().contiguous()
    initial_loss = float(residual_initial.square().sum().item())

    values = measured_values_by_input(layer, x, device)
    r_tia = layer.r_tia.detach().float().view(()).to(device)
    final_pos = initial_pos.clone()
    final_neg = initial_neg.clone()
    rows_changed = 0
    total_changed = 0
    final_loss_sum = 0.0
    output_count, input_count, topk = cand_pos.shape
    input_indices = torch.arange(input_count, device=device)

    for output_id in range(output_count):
        residual = residual_initial[output_id].clone()
        choices = torch.zeros(input_count, device=device, dtype=torch.long)
        changed_this_row = 0
        for _ in range(sweeps):
            for start in range(0, input_count, coord_block):
                end = min(start + coord_block, input_count)
                block_values = values[start:end]  # [B, M, states]
                pos_ids = cand_pos[output_id, start:end]  # [B, K]
                neg_ids = cand_neg[output_id, start:end]
                bsz = end - start
                pos_vals = block_values.gather(
                    2, pos_ids[:, None, :].expand(bsz, block_values.shape[1], topk)
                )
                neg_vals = block_values.gather(
                    2, neg_ids[:, None, :].expand(bsz, block_values.shape[1], topk)
                )
                contrib = (pos_vals - neg_vals).permute(1, 0, 2).contiguous() * r_tia
                current_choice = choices[start:end]
                current = contrib.gather(
                    2, current_choice[None, :, None].expand(contrib.shape[0], bsz, 1)
                ).squeeze(2)
                delta = contrib - current.unsqueeze(2)
                score = 2.0 * torch.einsum("m,mbk->bk", residual, delta)
                score = score + delta.square().sum(dim=0)
                best_score, best_choice = score.min(dim=1)
                improve = best_score < -float(min_improvement)
                if bool(improve.any()):
                    selected = delta[:, torch.arange(bsz, device=device), best_choice]
                    residual.add_(selected[:, improve].sum(dim=1))
                    local = torch.arange(start, end, device=device)[improve]
                    choices[local] = best_choice[improve]
                    changed_this_row += int(improve.sum().item())
        final_loss_sum += float(residual.square().sum().item())
        if changed_this_row:
            rows_changed += 1
            total_changed += changed_this_row
            final_pos[output_id] = cand_pos[output_id, input_indices, choices].to(torch.int32)
            final_neg[output_id] = cand_neg[output_id, input_indices, choices].to(torch.int32)

    layer.set_assignment(final_pos, final_neg)
    return {
        "block": block_id,
        "initial_recon_loss": initial_loss,
        "final_recon_loss": final_loss_sum,
        "relative_recon_loss": final_loss_sum / max(initial_loss, 1e-30),
        "changed_weight_count": total_changed,
        "changed_weight_ratio": total_changed / float(output_count * input_count),
        "changed_output_rows": rows_changed,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model_config(args.model_scale)
    model = build_student(cfg, Path(args.checkpoint), device)
    library = IVLibrary.load(
        Path(args.codebook_excel),
        args.codebook_sheet,
        args.measured_current_scale,
        args.min_valid_current_na,
    )
    configure_codebook_on_model(model, library, args.vin_lut_bins)

    train_loader, _val_loader = get_dataloaders(
        args.data,
        batch_size=args.calib_batch_size,
        num_workers=args.num_workers,
        distributed=False,
    )
    _, eval_loader = get_dataloaders(
        args.data,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        distributed=False,
    )
    started = time.perf_counter()
    print("Capturing fc1 calibration inputs...", flush=True)
    calib_inputs = capture_fc1_inputs(
        model,
        train_loader,
        device,
        args.calib_batches,
        args.max_tokens_per_layer,
    )

    candidate_rows: list[dict[str, Any]] = []
    candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    print("Building Q-point Top-K candidates and setting top-1 assignment...", flush=True)
    for block_id, layer in iter_deployment_fc1(model):
        cand_pos, cand_neg, row = qpoint_topk_candidates(
            layer,
            calib_inputs[block_id],
            library,
            args.assignment_q,
            args.topk,
            args.lambda_cm,
            args.assignment_chunk_devices,
            device,
        )
        layer.set_assignment(cand_pos[:, :, 0].to(device=device, dtype=torch.int32), cand_neg[:, :, 0].to(device=device, dtype=torch.int32))
        row.update({"block": block_id})
        candidate_rows.append(row)
        candidates[block_id] = (cand_pos, cand_neg)
        print(f"  block {block_id}: top-{row['topk']} candidates ready", flush=True)

    qpoint_assignment_path = out_dir / "qpoint_assignment.pt"
    save_assignment(qpoint_assignment_path, model, candidate_rows)
    print(f"Saved Q-point Top-1 assignment: {qpoint_assignment_path}", flush=True)

    baseline = None
    if not args.no_baseline_eval:
        print("Evaluating top-1 Q-point measured baseline...", flush=True)
        baseline = evaluate(model, model, eval_loader, device, "measured", False, args.max_val_batches)
        print(f"  baseline measured accuracy={baseline['accuracy']:.2f}%", flush=True)

    if args.qpoint_only:
        elapsed = time.perf_counter() - started
        result = {
            "status": "completed",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "codebook_excel": str(Path(args.codebook_excel).resolve()),
            "codebook_sheet": args.codebook_sheet,
            "assignment": "independent_qpoint_top1",
            "assignment_path": str(qpoint_assignment_path.resolve()),
            "args": vars(args),
            "baseline_measured": baseline,
            "candidate_rows": candidate_rows,
            "elapsed_seconds": elapsed,
        }
        save_json(out_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    print("Running activation-aware residual coordinate descent...", flush=True)
    opt_rows: list[dict[str, Any]] = []
    for block_id, layer in iter_deployment_fc1(model):
        cand_pos, cand_neg = candidates[block_id]
        row = optimize_layer_assignment(
            block_id,
            layer,
            calib_inputs[block_id],
            cand_pos,
            cand_neg,
            args.sweeps,
            args.coord_block,
            args.min_improvement,
            device,
        )
        opt_rows.append(row)
        print(
            f"  block {block_id}: recon {row['relative_recon_loss']:.4f}x, "
            f"changed {100*row['changed_weight_ratio']:.2f}%",
            flush=True,
        )

    print("Evaluating activation-aware measured deployment...", flush=True)
    activation_aware = evaluate(model, model, eval_loader, device, "measured", False, args.max_val_batches)
    elapsed = time.perf_counter() - started
    result = {
        "status": "completed",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "codebook_excel": str(Path(args.codebook_excel).resolve()),
        "codebook_sheet": args.codebook_sheet,
        "assignment": "activation_aware_topk_residual_cd",
        "args": vars(args),
        "baseline_measured": baseline,
        "activation_aware_measured": activation_aware,
        "candidate_rows": candidate_rows,
        "optimization_rows": opt_rows,
        "elapsed_seconds": elapsed,
    }
    save_json(out_dir / "metrics.json", result)
    save_assignment(out_dir / "activation_aware_assignment.pt", model, opt_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
