"""Train one VGG8/device/seed run and save probe predictions every five epochs."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from isa.device_sweeps.models.quantization import set_quantization_config
from isa.device_sweeps.models.vgg8 import VGG8
from isa.prediction_trajectory.protocol import (
    DEVICES,
    TRANSISTOR_DEVICES,
    atomic_json,
    atomic_npz,
    load_config,
)


@torch.no_grad()
def clamp_states(model: nn.Module) -> None:
    """Project every differential device state back to its physical range."""

    for module in model.modules():
        if hasattr(module, "theta_pos") and hasattr(module, "theta_neg"):
            module.theta_pos.clamp_(module.w_min, module.w_max)
            module.theta_neg.clamp_(module.w_min, module.w_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=DEVICES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device-config", default="configs/device_sweeps/device_params.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def input_normalization(device_name: str) -> transforms.Normalize:
    if device_name in TRANSISTOR_DEVICES:
        return transforms.Normalize((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    return transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))


def datasets_for_device(data_dir: str, device_name: str):
    norm = input_normalization(device_name)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), norm])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    validation_set = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=evaluation_transform
    )
    return train_set, validation_set


def data_loader(
    dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        generator=generator,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    bf16: bool,
    channels_last: bool,
    max_steps: int = 0,
    return_probabilities: bool = False,
) -> tuple[float, float, np.ndarray | None]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    probabilities: list[np.ndarray] = []
    for step, (inputs, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break
        inputs = inputs.to(device=device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device=device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(inputs)
        loss = criterion(logits.float(), labels)
        batch_count = labels.numel()
        loss_sum += float(loss.item()) * batch_count
        correct += int(logits.argmax(1).eq(labels).sum().item())
        count += batch_count
        if return_probabilities:
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if count == 0:
        raise RuntimeError("evaluation loader produced no examples")
    prediction_array = (
        np.concatenate(probabilities, axis=0).astype(np.float32, copy=False) if return_probabilities else None
    )
    return loss_sum / count, correct / count, prediction_array


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    bf16: bool,
    channels_last: bool,
    grad_clip: float,
    max_steps: int,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for step, (inputs, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break
        inputs = inputs.to(device=device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(inputs)
            loss = criterion(logits.float(), labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
        optimizer.step()
        clamp_states(model)
        batch_count = labels.numel()
        loss_sum += float(loss.detach().item()) * batch_count
        correct += int(logits.detach().argmax(1).eq(labels).sum().item())
        count += batch_count
    if count == 0:
        raise RuntimeError("training loader produced no examples")
    return loss_sum / count, correct / count


def append_history(path: Path, row: dict[str, object]) -> None:
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def truncate_history(path: Path, last_epoch: int) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= last_epoch]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the physical VGG8 experiment")
    config = load_config(args.config)
    training = dict(config["training"])
    epochs = args.epochs or int(training["epochs"])
    snapshot_interval = int(training["snapshot_interval"])
    if epochs <= 0 or snapshot_interval <= 0:
        raise ValueError("epochs and snapshot interval must be positive")

    output_root = Path(args.output_root)
    run_dir = output_root / "raw" / args.device / f"seed_{args.seed}"
    snapshots_dir = run_dir / "snapshots"
    completion_path = run_dir / "completed.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "train_history.csv"
    if completion_path.is_file():
        print(f"RUN_SKIP_COMPLETE device={args.device} seed={args.seed}", flush=True)
        return
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    cuda_device = torch.device("cuda:0")
    torch.cuda.set_device(cuda_device)

    with Path(args.device_config).open(encoding="utf-8") as handle:
        device_config = yaml.safe_load(handle)
    quantization = device_config.get("quantization", {})
    set_quantization_config(
        enabled=bool(quantization.get("enabled", False)),
        input_bits=int(quantization.get("input_bits", 8)),
        output_bits=int(quantization.get("output_bits", 8)),
    )
    physical = dict(config["device_hyperparameters"][args.device])
    model_params = dict(device_config[args.device])
    model_params["init_center"] = float(physical["init_center"])
    model_params.pop("init_std", None)
    v_min, v_max = (0.0, 4.0) if args.device in TRANSISTOR_DEVICES else (-0.5, 0.5)
    model = VGG8(
        device_type=args.device,
        device_params=model_params,
        tia_r=float(physical["tia_r"]),
        w_init_max=float(physical["init_half_width"]),
        v_min=v_min,
        v_max=v_max,
    ).to(cuda_device)
    channels_last = bool(training.get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    train_set, validation_set = datasets_for_device(args.data_dir, args.device)
    with np.load(output_root / "probe_indices.npz", allow_pickle=False) as probe_payload:
        probe_indices = np.asarray(probe_payload["indices"], dtype=np.int64)
        probe_labels = np.asarray(probe_payload["labels"], dtype=np.int64)
        probe_seed = int(probe_payload["seed"])
    if not np.array_equal(probe_labels, np.asarray(validation_set.targets)[probe_indices]):
        raise ValueError("shared probe labels do not match CIFAR-10")
    validation_loader = data_loader(
        validation_set,
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        shuffle=False,
        seed=args.seed,
    )
    probe_loader = data_loader(
        Subset(validation_set, probe_indices.tolist()),
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        shuffle=False,
        seed=probe_seed,
    )

    start_epoch = 1
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=cuda_device)
        if state["device"] != args.device or int(state["seed"]) != args.seed:
            raise ValueError("checkpoint identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        truncate_history(history_path, int(state["epoch"]))
        print(
            f"RUN_RESUME device={args.device} seed={args.seed} epoch={start_epoch}",
            flush=True,
        )

    run_metadata = {
        "device": args.device,
        "seed": args.seed,
        "epochs": epochs,
        "snapshot_interval": snapshot_interval,
        "probe_size": int(probe_indices.size),
        "probe_seed": probe_seed,
        "training": training,
        "device_hyperparameters": physical,
        "device_parameters": model_params,
        "constraint": "device state clamp after every optimizer step",
    }
    atomic_json(run_dir / "run_config.json", run_metadata)

    bf16 = bool(training.get("bf16", True))
    if start_epoch == 1 and not (snapshots_dir / "epoch_000.npz").is_file():
        val_loss, val_accuracy, _ = evaluate(
            model,
            validation_loader,
            criterion,
            cuda_device,
            bf16=bf16,
            channels_last=channels_last,
            max_steps=args.max_val_steps,
        )
        _, _, probabilities = evaluate(
            model,
            probe_loader,
            criterion,
            cuda_device,
            bf16=bf16,
            channels_last=channels_last,
            return_probabilities=True,
        )
        atomic_npz(
            snapshots_dir / "epoch_000.npz",
            probabilities=probabilities,
            epoch=np.asarray(0, dtype=np.int32),
            validation_accuracy=np.asarray(val_accuracy, dtype=np.float64),
            validation_loss=np.asarray(val_loss, dtype=np.float64),
        )
        atomic_torch_save(
            {
                "device": args.device,
                "seed": args.seed,
                "epoch": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_path,
        )

    run_started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_seed = args.seed * 1_000_003 + epoch
        seed_all(epoch_seed)
        epoch_started = time.perf_counter()
        train_loader = data_loader(
            train_set,
            batch_size=int(training["batch_size"]),
            workers=int(training["workers"]),
            shuffle=True,
            seed=epoch_seed,
        )
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            cuda_device,
            bf16=bf16,
            channels_last=channels_last,
            grad_clip=float(training["grad_clip"]),
            max_steps=args.max_train_steps,
        )
        scheduler.step()

        snapshot_now = epoch % snapshot_interval == 0 or epoch == epochs
        val_loss = float("nan")
        val_accuracy = float("nan")
        if snapshot_now:
            val_loss, val_accuracy, _ = evaluate(
                model,
                validation_loader,
                criterion,
                cuda_device,
                bf16=bf16,
                channels_last=channels_last,
                max_steps=args.max_val_steps,
            )
            _, _, probabilities = evaluate(
                model,
                probe_loader,
                criterion,
                cuda_device,
                bf16=bf16,
                channels_last=channels_last,
                return_probabilities=True,
            )
            atomic_npz(
                snapshots_dir / f"epoch_{epoch:03d}.npz",
                probabilities=probabilities,
                epoch=np.asarray(epoch, dtype=np.int32),
                validation_accuracy=np.asarray(val_accuracy, dtype=np.float64),
                validation_loss=np.asarray(val_loss, dtype=np.float64),
            )
            atomic_torch_save(
                {
                    "device": args.device,
                    "seed": args.seed,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_path,
            )

        row = {
            "device": args.device,
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "snapshot": int(snapshot_now),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        append_history(history_path, row)
        print(
            f"TRAJECTORY_PROGRESS device={args.device} seed={args.seed} "
            f"epoch={epoch}/{epochs} train_acc={train_accuracy:.6f} "
            f"val_acc={val_accuracy:.6f} snapshot={int(snapshot_now)} "
            f"epoch_seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )

    snapshot_paths = sorted(snapshots_dir.glob("epoch_*.npz"))
    expected_snapshots = epochs // snapshot_interval + 1
    if epochs % snapshot_interval:
        expected_snapshots += 1
    if len(snapshot_paths) != expected_snapshots:
        raise RuntimeError(f"expected {expected_snapshots} snapshots, found {len(snapshot_paths)}")
    atomic_json(
        completion_path,
        {
            **run_metadata,
            "status": "completed",
            "snapshot_count": len(snapshot_paths),
            "elapsed_seconds": time.perf_counter() - run_started,
        },
    )
    print(
        f"TRAJECTORY_RUN_DONE device={args.device} seed={args.seed} snapshots={len(snapshot_paths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
