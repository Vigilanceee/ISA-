"""CIFAR-100 dataloaders with MixUp, CutMix, and RandAugment."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from isa.vision.config import CIFAR100_MEAN, CIFAR100_STD


def _ensure_cifar_layout(data_dir: str) -> Path:
    """Ensure torchvision-compatible layout (cifar-100-python)."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical = root / "cifar-100-python"
    alt = root / "cifar100"
    if not canonical.exists() and alt.exists() and (alt / "train").is_file():
        canonical.symlink_to(alt.resolve())
    return root


def build_transforms(train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    root = _ensure_cifar_layout(data_dir)
    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_set = datasets.CIFAR100(root=str(root), train=True, download=False, transform=train_tf)
    val_set = datasets.CIFAR100(root=str(root), train=False, download=False, transform=val_tf)

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# MixUp / CutMix (applied per-batch in the training loop)
# ---------------------------------------------------------------------------

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8):
    """MixUp: linearly interpolate pairs of examples and their labels.

    Returns (mixed_x, y_a, y_b, lam) where the loss should be computed as
        lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b).
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # symmetric sampling — always use the larger side
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix: replace a rectangular region with another image's patch.

    Returns (mixed_x, y_a, y_b, lam) — same interface as mixup_data.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    W = x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(W * cut_rat))
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, W)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]

    # True lambda based on actual cut area
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * W))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_cutmix_target(loss_fn, logits: torch.Tensor, y_a: torch.Tensor,
                        y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Weighted cross-entropy for MixUp / CutMix targets."""
    return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)


# ---------------------------------------------------------------------------
# ImageNet / ImageNet-subset dataloader
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_imagenet_transforms(train: bool = True, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    distributed: bool = False,
    img_size: int = 224,
):
    from torchvision.datasets import ImageFolder
    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/val'
    train_tf = build_imagenet_transforms(train=True, img_size=img_size)
    val_tf = build_imagenet_transforms(train=False, img_size=img_size)
    train_set = ImageFolder(train_dir, transform=train_tf)
    val_set = ImageFolder(val_dir, transform=val_tf)
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
