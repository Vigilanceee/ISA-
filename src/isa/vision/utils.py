"""Misc utilities: seeding, logging, checkpoints, Vth stats."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clamp_all_vth(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "clamp_vth"):
            module.clamp_vth()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_acc: float,
    args: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": args,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = False

    def log(self, row: Dict[str, Any]) -> None:
        write_header = not self._initialized and not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)


def collect_vth_stats(model: nn.Module) -> Dict[str, float]:
    pos_vals, neg_vals = [], []
    for module in model.modules():
        if hasattr(module, "vth_pos") and hasattr(module, "vth_neg"):
            pos_vals.append(module.vth_pos.detach().cpu().flatten())
            neg_vals.append(module.vth_neg.detach().cpu().flatten())
    if not pos_vals:
        return {}
    pos = torch.cat(pos_vals)
    neg = torch.cat(neg_vals)
    return {
        "vth_pos_mean": pos.mean().item(),
        "vth_pos_std": pos.std().item(),
        "vth_neg_mean": neg.mean().item(),
        "vth_neg_std": neg.std().item(),
    }
