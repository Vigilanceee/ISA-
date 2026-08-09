#!/usr/bin/env python3
"""Evaluate checkpoint and generate analysis plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from isa.vision.config import ModelConfig  # noqa: E402
from isa.vision.models import build_model  # noqa: E402
from isa.operators.cim import VoltageMapping, CIMLinear  # noqa: E402
from isa.vision.data import get_dataloaders  # noqa: E402
from isa.vision.metrics import accuracy, cross_entropy_loss  # noqa: E402
from isa.vision.utils import count_parameters, get_device, load_checkpoint, collect_vth_stats  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--model", type=str, default="physical_vit", choices=["physical_vit", "standard_vit"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", type=str, default="./outputs/physical_vit_cifar100")
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--voltage-max", type=float, default=4.0)
    p.add_argument("--plot", action="store_true", help="Save Vth and voltage histograms")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = cross_entropy_loss(logits, targets)
        acc1 = accuracy(logits, targets, topk=(1,))[0]
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += acc1 * bs / 100.0
        total += bs
    return total_loss / total, 100.0 * total_correct / total


def collect_distributions(model: nn.Module, loader, device, max_batches: int = 20):
    vth_pos, vth_neg, voltages = [], [], []
    for module in model.modules():
        if isinstance(module, CIMLinear):
            vth_pos.append(module.vth_pos.detach().cpu().flatten())
            vth_neg.append(module.vth_neg.detach().cpu().flatten())

    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        x = model.patch_embed(images).flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(images.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + model.pos_embed

        for blk in model.blocks:
            normed = blk.norm2(x)
            if hasattr(blk.mlp, "vm1"):
                voltages.append(blk.mlp.vm1(normed).detach().cpu().flatten())
            x = blk(x)

    return {
        "vth_pos": torch.cat(vth_pos).numpy() if vth_pos else None,
        "vth_neg": torch.cat(vth_neg).numpy() if vth_neg else None,
        "voltage": torch.cat(voltages).numpy() if voltages else None,
    }


def plot_results(out_dir: Path, dist: dict, log_csv: Path | None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plotting")
        return

    if dist.get("vth_pos") is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(dist["vth_pos"], bins=50, alpha=0.7, label="Vth_pos")
        axes[0].hist(dist["vth_neg"], bins=50, alpha=0.7, label="Vth_neg")
        axes[0].set_title("Vth distribution")
        axes[0].legend()
        if dist.get("voltage") is not None:
            axes[1].hist(dist["voltage"], bins=50, alpha=0.8)
            axes[1].set_title("VoltageMapping output (V_gs)")
        fig.tight_layout()
        fig.savefig(out_dir / "vth_voltage_hist.png", dpi=150)
        plt.close(fig)

    if log_csv and log_csv.exists():
        import csv

        epochs, train_acc, val_acc = [], [], []
        with open(log_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))
        if epochs:
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, train_acc, label="train_acc")
            plt.plot(epochs, val_acc, label="val_acc")
            plt.xlabel("epoch")
            plt.ylabel("accuracy (%)")
            plt.legend()
            plt.title("Training curves")
            plt.savefig(out_dir / "accuracy_curve.png", dpi=150)
            plt.close()


def main():
    args = parse_args()
    device = get_device("cuda")
    out_dir = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(mlp_ratio=args.mlp_ratio, voltage_max=args.voltage_max)
    model = build_model(args.model, model_cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded {args.checkpoint} | epoch={ckpt.get('epoch')} best_acc={ckpt.get('best_acc', 0):.2f}")
    print(f"Parameters: {count_parameters(model):,}")

    _, val_loader = get_dataloaders(args.data, batch_size=args.batch_size)
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Val loss={val_loss:.4f} | Val acc={val_acc:.2f}%")

    stats = collect_vth_stats(model)
    if stats:
        print(f"Vth stats: {stats}")

    if args.plot and args.model == "physical_vit":
        dist = collect_distributions(model, val_loader, device)
        log_csv = out_dir / "training_log.csv"
        plot_results(out_dir, dist, log_csv if log_csv.exists() else None)
        print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
