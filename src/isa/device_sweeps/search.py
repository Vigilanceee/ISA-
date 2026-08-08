#!/usr/bin/env python3
"""Resumable MLP/VGG8 paper search.

Learning rate, physical state initialization, and TIA resistance are searched
for each device/model pair. Regularization and scheduler settings remain fixed
per architecture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from isa.device_sweeps.backend import (
    BACKENDS,
    normalize_backend_parameters,
)
from isa.device_sweeps.health import evaluate_training_health
from isa.device_sweeps.models.mlp import MLP
from isa.device_sweeps.models.quantization import set_quantization_config
from isa.device_sweeps.models.vgg8 import VGG8
from isa.device_sweeps.utils.data_loader import get_cifar10_loaders, get_mnist_loaders

DEVICES = ("reram", "pcm", "stt", "fefet", "flash")
TRANSISTOR_DEVICES = {"fefet", "flash"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if distributed() else 1


def main_process() -> bool:
    return rank() == 0


def barrier() -> None:
    if distributed():
        dist.barrier()


def broadcast(value):
    payload = [value]
    if distributed():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def clamp_states(model: nn.Module) -> None:
    for module in raw_model(model).modules():
        if hasattr(module, "theta_pos") and hasattr(module, "theta_neg"):
            module.theta_pos.clamp_(module.w_min, module.w_max)
            module.theta_neg.clamp_(module.w_min, module.w_max)


def aggregate_metrics(loss_sum, correct, count) -> tuple[float, float]:
    loss_sum = reduce_sum(loss_sum)
    correct = reduce_sum(correct)
    count = reduce_sum(count)
    return (loss_sum / count).item(), (correct / count).item()


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    bf16: bool,
    channels_last: bool,
    grad_clip: float,
    max_steps: int,
) -> tuple[float, float]:
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = torch.zeros((), device=device, dtype=torch.long)
    for step, (inputs, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(inputs)
            loss = criterion(logits.float(), labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, foreach=True
            )
        optimizer.step()
        clamp_states(model)
        batch = labels.numel()
        loss_sum += loss.detach() * batch
        correct += logits.detach().argmax(1).eq(labels).sum()
        count += batch
    return aggregate_metrics(loss_sum, correct, count)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    bf16: bool,
    channels_last: bool,
    max_steps: int,
) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = torch.zeros((), device=device, dtype=torch.long)
    for step, (inputs, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(inputs)
            loss = criterion(logits.float(), labels)
        batch = labels.numel()
        loss_sum += loss * batch
        correct += logits.argmax(1).eq(labels).sum()
        count += batch
    return aggregate_metrics(loss_sum, correct, count)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sample_hyperparameters(trial: optuna.Trial, ranges: dict) -> dict:
    center_lo, center_hi = map(float, ranges["init_center"])
    width_lo, width_hi = map(float, ranges["init_half_width"])
    tia_lo, tia_hi = map(float, ranges["tia_r"])
    lr_lo, lr_hi = map(float, ranges["lr"])
    return {
        "lr": trial.suggest_float("lr", lr_lo, lr_hi, log=True),
        "init_center": trial.suggest_float(
            "init_center", center_lo, center_hi
        ),
        "init_half_width": trial.suggest_float(
            "init_half_width", width_lo, width_hi, log=True
        ),
        "tia_r": trial.suggest_float("tia_r", tia_lo, tia_hi, log=True),
    }


def trial_checkpoint(directory: Path, trial_number: int) -> Path:
    return directory / "checkpoints" / f"trial_{trial_number}_last.pt"


def get_or_create_trial(
    study: optuna.Study,
    directory: Path,
    ranges: dict,
) -> tuple[optuna.Trial, dict, bool]:
    running = [
        frozen
        for frozen in study.get_trials(deepcopy=False)
        if frozen.state == optuna.trial.TrialState.RUNNING
    ]
    for frozen in running:
        checkpoint = trial_checkpoint(directory, frozen.number)
        if checkpoint.exists():
            trial = optuna.Trial(study, frozen._trial_id)
            return trial, dict(frozen.params), True
        study.tell(frozen.number, state=optuna.trial.TrialState.FAIL)

    trial = study.ask()
    return trial, sample_hyperparameters(trial, ranges), False


def write_summary(study: optuna.Study, output: Path) -> None:
    rows = []
    for trial in study.get_trials(deepcopy=False):
        values = list(trial.intermediate_values.values())
        if trial.value is not None:
            values.append(trial.value)
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "best_val_acc": max(values)
                if values
                else trial.user_attrs.get("best_val_acc", ""),
                "last_epoch": max(trial.intermediate_values)
                if trial.intermediate_values
                else trial.user_attrs.get("last_epoch", ""),
                "init_center": trial.params.get("init_center", ""),
                "init_half_width": trial.params.get("init_half_width", ""),
                "tia_r": trial.params.get("tia_r", ""),
                "lr": trial.params.get("lr", ""),
                "stop_reason": trial.user_attrs.get("stop_reason", ""),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "trial", "state", "best_val_acc", "last_epoch",
            "init_center", "init_half_width", "tia_r", "lr", "stop_reason",
        ])
        writer.writeheader()
        writer.writerows(rows)


def truncate_trial_log(csv_path: Path, last_committed_epoch: int) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [
            row
            for row in reader
            if int(row["epoch"]) <= last_committed_epoch
        ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_validation_history(
    csv_path: Path, last_committed_epoch: int
) -> list[float]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [
            float(row["val_acc"])
            for row in csv.DictReader(handle)
            if int(row["epoch"]) <= last_committed_epoch
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=DEVICES, required=True)
    parser.add_argument("--model", choices=("mlp", "vgg8"), default="vgg8")
    parser.add_argument("--device-config", default="configs/device_params.yaml")
    parser.add_argument("--search-config", default="configs/paper_search.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-root", default="logs/vgg8_paper_baseline")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument(
        "--conv-backend",
        choices=BACKENDS,
        default="",
        help="Temporarily override the configured convolution backend.",
    )
    parser.add_argument(
        "--linear-backend",
        choices=BACKENDS,
        default="",
        help="Temporarily override the configured linear backend.",
    )
    parser.add_argument(
        "--lowrank-rank",
        type=int,
        default=0,
        help="Temporarily override the configured low-rank basis size.",
    )
    parser.add_argument("--fixed-init-center", type=float)
    parser.add_argument("--fixed-init-half-width", type=float)
    parser.add_argument("--fixed-tia-r", type=float)
    parser.add_argument("--fixed-lr", type=float)
    parser.add_argument(
        "--health-pruning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable VGG8 early-health thresholds; defaults on for every VGG8 device.",
    )
    parser.add_argument("--health-epoch8-min-best", type=float)
    parser.add_argument("--health-epoch20-min-best", type=float)
    parser.add_argument("--health-plateau-window", type=int)
    parser.add_argument("--health-plateau-min-gain", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with open(args.device_config, "r", encoding="utf-8") as handle:
        device_config = yaml.safe_load(handle)
    with open(args.search_config, "r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    training_config = experiment_config["training"]
    fixed = (
        training_config[args.model]
        if args.model in training_config
        else training_config
    )
    ranges = dict(experiment_config["search"][args.device])
    ranges["lr"] = ranges["lr"][args.model]
    if args.fixed_init_center is not None:
        ranges["init_center"] = [
            args.fixed_init_center,
            args.fixed_init_center,
        ]
    if args.fixed_init_half_width is not None:
        ranges["init_half_width"] = [
            args.fixed_init_half_width,
            args.fixed_init_half_width,
        ]
    if args.fixed_tia_r is not None:
        ranges["tia_r"] = [args.fixed_tia_r, args.fixed_tia_r]
    if args.fixed_lr is not None:
        ranges["lr"] = [args.fixed_lr, args.fixed_lr]
    epochs = args.epochs or int(fixed["epochs"])
    target_trials = args.trials or int(fixed["trials"])
    bf16 = bool(fixed.get("bf16", True))
    channels_last = (
        args.model == "vgg8"
        and bool(fixed.get("channels_last", True))
    )

    quantization = device_config.get("quantization", {})
    set_quantization_config(
        enabled=bool(quantization.get("enabled", False)),
        input_bits=int(quantization.get("input_bits", 8)),
        output_bits=int(quantization.get("output_bits", 8)),
    )

    run_directory = Path(args.run_root) / args.device
    if main_process():
        run_directory.mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(run_directory).free / (1024 ** 3)
        if free_gib < 5.0:
            raise RuntimeError(
                f"Only {free_gib:.2f} GiB is free under {run_directory}; "
                "at least 5 GiB is required for checkpoints and weights."
            )
    barrier()

    loader_kwargs = {
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "num_workers": args.workers,
        "device_type": args.device,
        "distributed": ddp,
    }
    loader_function = (
        get_cifar10_loaders
        if args.model == "vgg8"
        else get_mnist_loaders
    )
    if ddp:
        train_loader, val_loader, train_sampler = loader_function(
            **loader_kwargs
        )
    else:
        train_loader, val_loader = loader_function(**loader_kwargs)
        train_sampler = None

    study = None
    if main_process():
        database_path = (run_directory / "optuna.db").resolve()
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{database_path}",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        study = optuna.create_study(
            study_name=(
                args.study_name
                or f"{args.model}_{args.device}_paper_search"
            ),
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=int(fixed["prune_start_epoch"]),
            ),
        )

    while True:
        if main_process():
            finished = [
                trial
                for trial in study.get_trials(deepcopy=False)
                if trial.state
                in {
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                }
            ]
            done = len(finished) >= target_trials
        else:
            done = None
        done = broadcast(done)
        if done:
            break

        if main_process():
            trial, hp, resuming = get_or_create_trial(
                study, run_directory, ranges
            )
            trial_info = {
                "number": trial.number,
                "hp": hp,
                "resuming": resuming,
            }
        else:
            trial = None
            trial_info = None
        trial_info = broadcast(trial_info)
        trial_number = int(trial_info["number"])
        hp = trial_info["hp"]
        seed_everything(args.seed + trial_number)

        params = dict(device_config[args.device])
        params["init_center"] = float(hp["init_center"])
        params.pop("init_std", None)
        params = normalize_backend_parameters(
            params,
            conv_backend=args.conv_backend,
            linear_backend=args.linear_backend,
        )
        if args.lowrank_rank:
            params["lowrank_rank"] = int(args.lowrank_rank)
        v_min, v_max = (
            (0.0, 4.0)
            if args.device in TRANSISTOR_DEVICES
            else (-0.5, 0.5)
        )
        model_class = VGG8 if args.model == "vgg8" else MLP
        model = model_class(
            device_type=args.device,
            device_params=params,
            tia_r=float(hp["tia_r"]),
            w_init_max=float(hp["init_half_width"]),
            v_min=v_min,
            v_max=v_max,
        ).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        if ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["lr"]),
            weight_decay=float(fixed["weight_decay"]),
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(fixed["label_smoothing"])
        )
        checkpoint_path = trial_checkpoint(run_directory, trial_number)
        csv_path = run_directory / f"trial_{trial_number}.csv"
        start_epoch = 1
        best_val = 0.0
        val_history: list[float] = []
        if bool(trial_info["resuming"]):
            state = torch.load(checkpoint_path, map_location=device)
            raw_model(model).load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            best_val = float(state["best_val"])
            if main_process():
                truncate_trial_log(csv_path, int(state["epoch"]))
                val_history = read_validation_history(
                    csv_path, int(state["epoch"])
                )
            val_history = broadcast(val_history)

        if main_process():
            metadata = {
                "trial": trial_number,
                "model": args.model,
                "dataset": "cifar10" if args.model == "vgg8" else "mnist",
                "device": args.device,
                "searched": hp,
                "fixed_training": fixed,
                "device_params": params,
                "world_size": world_size(),
            }
            source_root = Path(__file__).resolve().parents[1]
            matrix_kernel = source_root / "kernels/device_sweep" / f"{args.device}_triton.py"
            kernel_paths = [matrix_kernel]
            if params.get("direct_conv_enabled", False):
                kernel_paths.append(
                    source_root / "kernels/device_sweep/direct_conv_triton.py"
                )
            if params.get("conv_backend") == "factorized":
                kernel_paths.append(source_root / "approximations/node_planar.py")
            if (
                args.device == "flash"
                and params.get("exact_matrix_backend", "split")
                in {"auto", "ffn_tiled"}
            ):
                kernel_paths.append(
                    source_root / "kernels/transformer_ffn/ekv_triton.py"
                )
            metadata["device_kernels"] = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in kernel_paths
                if path.is_file()
            ]
            with (run_directory / f"trial_{trial_number}_config.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
            print(
                f"trial={trial_number} resume={trial_info['resuming']} "
                f"start_epoch={start_epoch} hp={hp}",
                flush=True,
            )

        pruned = False
        prune_reason = ""
        threshold_pruning = (
            args.health_pruning
            if args.health_pruning is not None
            else args.model == "vgg8"
        )
        trial_start = time.perf_counter()
        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_start = time.perf_counter()
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                bf16,
                channels_last,
                float(fixed["grad_clip"]),
                args.max_train_steps,
            )
            val_loss, val_acc = validate(
                model,
                val_loader,
                criterion,
                device,
                bf16,
                channels_last,
                args.max_val_steps,
            )
            scheduler.step()
            val_history.append(val_acc)
            health = evaluate_training_health(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_history=val_history,
                threshold_pruning=threshold_pruning,
                epoch8_min_best=(
                    args.health_epoch8_min_best
                    if args.health_epoch8_min_best is not None
                    else float(fixed.get("health_epoch8_min_best", 0.15))
                ),
                epoch20_min_best=(
                    args.health_epoch20_min_best
                    if args.health_epoch20_min_best is not None
                    else float(fixed.get("health_epoch20_min_best", 0.35))
                ),
                plateau_window=(
                    args.health_plateau_window
                    if args.health_plateau_window is not None
                    else int(fixed.get("health_plateau_window", 5))
                ),
                plateau_min_gain=(
                    args.health_plateau_min_gain
                    if args.health_plateau_min_gain is not None
                    else float(fixed.get("health_plateau_min_gain", 0.02))
                ),
            )
            improved = math.isfinite(val_acc) and val_acc > best_val
            if math.isfinite(val_acc):
                best_val = max(best_val, val_acc)

            prune_now = False
            if main_process():
                if math.isfinite(val_acc):
                    trial.report(val_acc, epoch)
                optuna_prune = (
                    math.isfinite(val_acc) and trial.should_prune()
                )
                prune_now = health.should_stop or optuna_prune
                prune_reason = (
                    health.reason
                    if health.should_stop
                    else "optuna_pruner" if optuna_prune else ""
                )
                if prune_reason:
                    trial.set_user_attr("stop_reason", prune_reason)
                trial.set_user_attr("last_epoch", epoch)
                trial.set_user_attr("best_val_acc", best_val)
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
                write_header = not csv_path.exists()
                with csv_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                if (
                    epoch == 1
                    or epoch % args.checkpoint_interval == 0
                    or prune_now
                ):
                    atomic_torch_save(
                        {
                            "epoch": epoch,
                            "best_val": best_val,
                            "model": raw_model(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "hp": hp,
                            "stop_reason": prune_reason,
                        },
                        checkpoint_path,
                    )
                if improved:
                    atomic_torch_save(
                        {
                            "epoch": epoch,
                            "val_acc": val_acc,
                            "model": raw_model(model).state_dict(),
                            "hp": hp,
                        },
                        run_directory
                        / "weights"
                        / f"trial_{trial_number}_best.pt",
                    )
                print(
                    f"[T{trial_number:02d}] E{epoch:03d}/{epochs} "
                    f"train={train_acc:.4f} val={val_acc:.4f} "
                    f"best={best_val:.4f} "
                    f"health={prune_reason or 'ok'} "
                    f"epoch={row['epoch_seconds']:.1f}s "
                    f"total={time.perf_counter() - trial_start:.0f}s",
                    flush=True,
                )

            prune_now = broadcast(prune_now)
            barrier()
            if prune_now:
                pruned = True
                break

        if main_process():
            if pruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, best_val)
            checkpoint_path.unlink(missing_ok=True)
            write_summary(study, run_directory / "summary.csv")
        barrier()
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    if main_process():
        write_summary(study, run_directory / "summary.csv")
        completed = [
            trial
            for trial in study.best_trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            best = max(completed, key=lambda trial: trial.value)
            print(
                f"complete model={args.model} device={args.device} "
                f"best_trial={best.number} "
                f"best_val_acc={best.value:.6f}",
                flush=True,
            )
    if distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
