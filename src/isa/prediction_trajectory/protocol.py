"""Shared configuration and probe-set helpers for prediction trajectories."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEVICES = ("reram", "pcm", "stt", "fefet", "flash")
TRANSISTOR_DEVICES = {"fefet", "flash"}


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {source}")
    unknown = sorted(set(payload.get("devices", ())) - set(DEVICES))
    if unknown:
        raise ValueError(f"Unknown devices in {source}: {', '.join(unknown)}")
    return payload


def stratified_probe_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic near-equal number of examples from each class."""

    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    classes = np.unique(targets)
    if size <= 0 or size > targets.size:
        raise ValueError("probe size must be positive and no larger than the dataset")
    base, remainder = divmod(size, len(classes))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for position, class_id in enumerate(classes):
        count = base + int(position < remainder)
        candidates = np.flatnonzero(targets == class_id)
        if candidates.size < count:
            raise ValueError(f"class {class_id} has only {candidates.size} examples")
        selected.append(np.sort(rng.choice(candidates, size=count, replace=False)))
    indices = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    if indices.size != size or np.unique(indices).size != size:
        raise RuntimeError("stratified probe selection produced duplicate or missing indices")
    return indices


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_probe_file(
    *,
    labels: Sequence[int] | np.ndarray,
    path: Path,
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create or validate the immutable shared probe definition."""

    targets = np.asarray(labels, dtype=np.int64)
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            indices = np.asarray(payload["indices"], dtype=np.int64)
            saved_labels = np.asarray(payload["labels"], dtype=np.int64)
            saved_seed = int(payload["seed"])
        if indices.shape != (size,) or saved_labels.shape != (size,):
            raise ValueError(f"probe definition has unexpected shape: {path}")
        if saved_seed != seed or not np.array_equal(saved_labels, targets[indices]):
            raise ValueError(f"probe definition does not match current protocol: {path}")
        return indices, saved_labels

    indices = stratified_probe_indices(targets, size=size, seed=seed)
    saved_labels = targets[indices]
    atomic_npz(
        path,
        indices=indices,
        labels=saved_labels,
        seed=np.asarray(seed, dtype=np.int64),
    )
    return indices, saved_labels
