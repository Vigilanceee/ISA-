"""Deterministic health checks for long device-training runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HealthDecision:
    """A terminal health decision made from metrics available after an epoch."""

    should_stop: bool
    reason: str = ""
    best_val_acc: float = 0.0
    recent_gain: float | None = None


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def evaluate_training_health(
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_history: Sequence[float],
    threshold_pruning: bool,
    epoch8_min_best: float = 0.15,
    epoch20_min_best: float = 0.35,
    plateau_window: int = 5,
    plateau_min_gain: float = 0.02,
) -> HealthDecision:
    """Evaluate non-finite metrics and the VGG8 early-health rules.

    Accuracy values are fractions rather than percentages. ``recent_gain`` is
    the improvement from the first point in the trailing window to the best
    point in that window; a declining window therefore has zero gain.
    """

    current_metrics = (train_loss, train_acc, val_loss, val_acc)
    if not _all_finite(current_metrics) or not _all_finite(val_history):
        return HealthDecision(True, "non_finite_metric")

    best_val = max((float(value) for value in val_history), default=0.0)
    if not threshold_pruning:
        return HealthDecision(False, best_val_acc=best_val)

    if epoch >= 8 and best_val < float(epoch8_min_best):
        return HealthDecision(
            True,
            f"epoch8_best_below_{float(epoch8_min_best):.4f}",
            best_val_acc=best_val,
        )

    window = max(2, int(plateau_window))
    recent_gain = None
    if len(val_history) >= window:
        recent = [float(value) for value in val_history[-window:]]
        recent_gain = max(0.0, max(recent) - recent[0])

    if (
        epoch >= 20
        and best_val < float(epoch20_min_best)
        and recent_gain is not None
        and recent_gain < float(plateau_min_gain)
    ):
        return HealthDecision(
            True,
            (
                f"epoch20_best_below_{float(epoch20_min_best):.4f}"
                f"_and_gain_below_{float(plateau_min_gain):.4f}"
            ),
            best_val_acc=best_val,
            recent_gain=recent_gain,
        )

    return HealthDecision(
        False,
        best_val_acc=best_val,
        recent_gain=recent_gain,
    )
