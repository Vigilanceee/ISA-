#!/usr/bin/env python3
"""Train Physical ViT, Hybrid ViT, or Standard ViT baseline on CIFAR-100."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from isa.vision.config import ModelConfig, apply_preset  # noqa: E402
from isa.vision.models import build_model  # noqa: E402
from isa.vision.initialization import (  # noqa: E402
    load_shared_baseline_weights,
    match_physical_ffns_to_baseline,
    initialize_ekv_model,
)
from isa.vision.data import get_dataloaders, get_imagenet_loaders, mixup_data, cutmix_data, mixup_cutmix_target  # noqa: E402
from isa.vision.metrics import AverageMeter, accuracy, cross_entropy_loss  # noqa: E402
from isa.vision.utils import (  # noqa: E402
    CSVLogger,
    clamp_all_vth,
    count_parameters,
    save_checkpoint,
    set_seed,
    collect_vth_stats,
)
from isa.vision.scheduler import WarmupCosineScheduler  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT on CIFAR-100")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--output", type=str, default="./outputs/physical_vit_cifar100")
    p.add_argument("--model", type=str, default="physical_vit",
                   choices=["physical_vit", "standard_vit", "hybrid_vit"])
    p.add_argument("--model-scale", type=str, default="tiny",
                   choices=["tiny", "mid", "small", "base"],
                   help="Model scale preset: tiny(192), mid(256), small(384)")
    p.add_argument("--dataset", type=str, default="cifar100",
                   choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128, help="per-rank batch size under DDP")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.0)
    # These override the preset when set explicitly (0 means use preset default)
    p.add_argument("--embed-dim", type=int, default=0, help="override embed_dim (0 = use preset)")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=0, help="override num_heads (0 = use preset)")
    p.add_argument("--mlp-ratio", type=float, default=0.0, help="override mlp_ratio (0 = use preset)")
    p.add_argument("--img-size", type=int, default=0, help="image size (0=auto: 32 for cifar, 224 for imagenet)")
    p.add_argument("--patch-size", type=int, default=0, help="patch size (0=auto)")
    p.add_argument("--voltage-max", type=float, default=4.0, help="first-layer DAC voltage range")
    p.add_argument("--mixup-alpha", type=float, default=0.8, help="MixUp Beta alpha (0=disable)")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix Beta alpha (0=disable)")
    p.add_argument("--mixup-prob", type=float, default=0.5, help="probability of MixUp vs CutMix")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--print-freq", type=int, default=50)
    # Debug / testing
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="debug: stop each epoch after N train batches")
    p.add_argument("--max-val-batches", type=int, default=0,
                   help="debug: stop validation after N batches")
    # Baseline initialization (for physical_vit / hybrid_vit)
    p.add_argument("--baseline-init-checkpoint", type=str, default="",
                   help="standard ViT checkpoint path for weight sharing + FFN matching")
    p.add_argument("--baseline-match-steps", type=int, default=20,
                   help="FFN matching optimization steps")
    p.add_argument("--baseline-match-batches", type=int, default=2,
                   help="calibration batches for FFN matching")
    p.add_argument("--baseline-match-max-rows", type=int, default=4096)
    p.add_argument("--baseline-match-batch-size", type=int, default=256)
    p.add_argument("--baseline-match-lr", type=float, default=1e-3)
    # EKV initialization strategy
    p.add_argument("--ekv-init", type=str, default="default",
                   choices=["default", "random_fixed", "centered_fixed",
                            "random_reverse_tia", "centered_reverse_tia", "calibrated"])
    p.add_argument("--ekv-eps-k", type=float, default=0.5)
    p.add_argument("--ekv-rho", type=float, default=0.8)
    p.add_argument("--ekv-quantile", type=float, default=0.99)
    p.add_argument("--ekv-clip-threshold", type=float, default=0.05)
    p.add_argument("--ekv-candidates", type=int, default=1)
    p.add_argument("--ekv-calib-batches", type=int, default=1)
    p.add_argument("--ekv-calib-max-rows", type=int, default=4096)
    # Voltage mapping init
    p.add_argument("--voltage-map-init", type=str, default="data",
                   choices=["data", "identity", "none"])
    p.add_argument("--voltage-map-quantile", type=float, default=0.995)
    p.add_argument("--voltage-map-low", type=float, default=0.2)
    p.add_argument("--voltage-map-high", type=float, default=3.8)
    p.add_argument("--voltage-map-iters", type=int, default=2)
    return p.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return True, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_metrics(loss_sum: float, correct_sum: float, total: int, device: torch.device) -> tuple[float, float]:
    values = torch.tensor([loss_sum, correct_sum, float(total)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denom = max(values[2].item(), 1.0)
    return values[0].item() / denom, values[1].item() / denom


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler | None,
    use_amp: bool,
    label_smoothing: float,
    grad_clip: float,
    print_freq: int,
    rank: int,
    max_train_batches: int = 0,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    loss_sum = 0.0
    correct_sum = 0.0
    total = 0

    for i, (images, targets) in enumerate(loader):
        if max_train_batches > 0 and i >= max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---- MixUp / CutMix ----
        use_mixup = mixup_alpha > 0 or cutmix_alpha > 0
        if use_mixup:
            if np.random.rand() < mixup_prob:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                mixed = True
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
                mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            if mixed:
                loss = mixup_cutmix_target(
                    lambda logits, t: cross_entropy_loss(logits, t, label_smoothing),
                    logits, targets_a, targets_b, lam,
                )
            else:
                loss = cross_entropy_loss(logits, targets, label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        clamp_all_vth(model.module if isinstance(model, DDP) else model)

        acc1 = accuracy(logits.detach(), targets, topk=(1,))[0]
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc1, batch_size)
        loss_sum += loss.item() * batch_size
        correct_sum += acc1 / 100.0 * batch_size
        total += batch_size

        if is_main_process(rank) and (i + 1) % print_freq == 0:
            print(
                f"  Epoch[{epoch}] Step[{i+1}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f} acc={acc_meter.avg:.2f}%",
                flush=True,
            )

    loss_avg, acc_avg = reduce_metrics(loss_sum, correct_sum, total, device)
    return loss_avg, acc_avg * 100.0


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    max_val_batches: int = 0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    total = 0

    for i, (images, targets) in enumerate(loader):
        if max_val_batches > 0 and i >= max_val_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = cross_entropy_loss(logits, targets, label_smoothing=0.0)
        acc1 = accuracy(logits, targets, topk=(1,))[0]
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        correct_sum += acc1 / 100.0 * batch_size
        total += batch_size

    loss_avg, acc_avg = reduce_metrics(loss_sum, correct_sum, total, device)
    return loss_avg, acc_avg * 100.0


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    set_seed(args.seed + rank)
    use_amp = not args.no_amp and device.type == "cuda"

    # Build model config from scale preset, optionally overridden by explicit args
    # Determine image/patch size
    if args.img_size > 0:
        img_size = args.img_size
    elif args.dataset == "imagenet":
        img_size = 224
    else:
        img_size = 32
    if args.patch_size > 0:
        patch_size = args.patch_size
    elif args.dataset == "imagenet":
        patch_size = 16
    else:
        patch_size = 4

    model_cfg = ModelConfig(
        img_size=img_size,
        patch_size=patch_size,
        depth=args.depth,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path,
        voltage_max=args.voltage_max,
    )
    apply_preset(model_cfg, args.model_scale)
    if args.embed_dim > 0:
        model_cfg.embed_dim = args.embed_dim
    if args.num_heads > 0:
        model_cfg.num_heads = args.num_heads
    if args.mlp_ratio > 0:
        model_cfg.mlp_ratio = args.mlp_ratio
    if args.dataset == "imagenet":
        model_cfg.num_classes = 200

    model = build_model(args.model, model_cfg).to(device)
    hidden = int(model_cfg.embed_dim * model_cfg.mlp_ratio)
    if is_main_process(rank):
        print(f"Model scale: {args.model_scale} | embed_dim={model_cfg.embed_dim} "
              f"heads={model_cfg.num_heads} hidden={hidden} depth={args.depth} "
              f"mlp_ratio={model_cfg.mlp_ratio}", flush=True)

    # --- Baseline weight sharing + FFN matching ---
    baseline_match_info = {}
    if args.baseline_init_checkpoint and args.model != "standard_vit" and not args.resume:
        if is_main_process(rank):
            print(f"Loading baseline checkpoint: {args.baseline_init_checkpoint}", flush=True)
        baseline_ckpt = torch.load(args.baseline_init_checkpoint, map_location=device)
        baseline_state = baseline_ckpt.get("model", baseline_ckpt)

        # Step 1: load shape-compatible weights (attention, embedding, fc2 for hybrid, etc.)
        shared_info = load_shared_baseline_weights(model, baseline_state)
        if is_main_process(rank):
            print(f"  Shared baseline tensors loaded: {shared_info['baseline_shared_tensors']} "
                  f"({shared_info['baseline_shared_parameters']:,} params)", flush=True)

        # Step 2: build a temporary baseline model for FFN input/output collection
        baseline_model = build_model("standard_vit", model_cfg).to(device)
        baseline_model.load_state_dict(baseline_state)
        if args.dataset == "imagenet":
            temp_loader, _ = get_imagenet_loaders(
                args.data, batch_size=args.batch_size, num_workers=args.num_workers,
                distributed=False, img_size=img_size,
            )
        else:
            temp_loader, _ = get_dataloaders(
                args.data, batch_size=args.batch_size, num_workers=args.num_workers, distributed=False,
            )

        # Step 3: match each physical/hybrid FFN to its baseline counterpart
        match_info = match_physical_ffns_to_baseline(
            model, baseline_model, temp_loader, device,
            steps=args.baseline_match_steps,
            max_batches=args.baseline_match_batches,
            max_rows=args.baseline_match_max_rows,
            batch_size=args.baseline_match_batch_size,
            lr=args.baseline_match_lr,
            use_amp=use_amp,
            seed=args.seed,
        )
        baseline_match_info = match_info
        del baseline_model
        if is_main_process(rank):
            layers = match_info.get("baseline_match_layers", 0)
            init_mse = match_info.get("baseline_match_initial_mse", float("nan"))
            final_mse = match_info.get("baseline_match_final_mse", float("nan"))
            print(f"  FFN matching: {layers} layers, MSE {init_mse:.6g} -> {final_mse:.6g}", flush=True)

        # Step 4: EKV initialization (voltage mapping calibration + Vth init)
        ekv_info = initialize_ekv_model(
            model, temp_loader, device,
            strategy=args.ekv_init,
            eps_k=args.ekv_eps_k,
            candidates=args.ekv_candidates,
            rho=args.ekv_rho,
            quantile=args.ekv_quantile,
            clip_threshold=args.ekv_clip_threshold,
            max_batches=args.ekv_calib_batches,
            max_rows=args.ekv_calib_max_rows,
            seed=args.seed,
            use_amp=use_amp,
            voltage_map_init=args.voltage_map_init,
            voltage_map_quantile=args.voltage_map_quantile,
            voltage_map_low=args.voltage_map_low,
            voltage_map_high=args.voltage_map_high,
            voltage_map_iters=args.voltage_map_iters,
        )
        if is_main_process(rank) and ekv_info.get("ekv_init") != "default":
            print(f"  EKV init: {ekv_info.get('ekv_init')} | "
                  f"clip_max={ekv_info.get('init_clip_max', float('nan')):.6f} | "
                  f"JS={ekv_info.get('init_js', float('nan')):.6f}", flush=True)
    elif args.resume and args.baseline_init_checkpoint and args.model != "standard_vit":
        if is_main_process(rank):
            print("Resume requested: skipping baseline FFN matching and EKV initialization", flush=True)

    # Output directory
    out_dir = Path(args.output) / args.model
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_acc = float(resume_ckpt.get("best_acc", 0.0))

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    if is_main_process(rank):
        raw_model = model.module if isinstance(model, DDP) else model
        n_params = count_parameters(raw_model)
        print(f"Model: {args.model} | Params: {n_params:,} | world_size={world_size}", flush=True)
        if args.resume:
            print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_acc={best_acc:.2f}", flush=True)

    if args.dataset == "imagenet":
        train_loader, val_loader = get_imagenet_loaders(
            args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
            img_size=img_size,
        )
    else:
        train_loader, val_loader = get_dataloaders(
            args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])

    logger = CSVLogger(
        out_dir / "training_log.csv",
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "lr", "epoch_seconds"],
    ) if is_main_process(rank) else None

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler, use_amp,
            args.label_smoothing, args.grad_clip, args.print_freq, rank,
            max_train_batches=args.max_train_batches,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
        )
        val_loss, val_acc = validate(model, val_loader, device, use_amp,
                                     max_val_batches=args.max_val_batches)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.time() - t0

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        if is_main_process(rank):
            assert logger is not None
            logger.log({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.4f}",
                "best_acc": f"{best_acc:.4f}",
                "lr": f"{lr:.8f}",
                "epoch_seconds": f"{epoch_seconds:.2f}",
            })

            raw_model = model.module if isinstance(model, DDP) else model
            ckpt_args = vars(args)
            save_checkpoint(out_dir / "last_checkpoint.pth", raw_model, optimizer, scheduler, epoch, best_acc, ckpt_args)
            if is_best:
                save_checkpoint(out_dir / "best_checkpoint.pth", raw_model, optimizer, scheduler, epoch, best_acc, ckpt_args)

            vth_stats = collect_vth_stats(raw_model)
            vth_msg = ""
            if vth_stats:
                vth_msg = f" | Vth_pos={vth_stats['vth_pos_mean']:.3f} Vth_neg={vth_stats['vth_neg_mean']:.3f}"
            print(
                f"Epoch {epoch}/{args.epochs-1} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"best={best_acc:.2f}% lr={lr:.2e}{vth_msg} "
                f"time={epoch_seconds:.1f}s",
                flush=True,
            )

    if is_main_process(rank):
        print(f"Training done. Best val acc: {best_acc:.2f}%", flush=True)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
