"""Training / evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    maxk = max(topk)
    b = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0).item() * 100.0 / b for k in topk]


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
    )
