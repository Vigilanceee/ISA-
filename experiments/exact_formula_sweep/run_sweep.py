#!/usr/bin/env python3
"""Resumable two-GPU scheduler for the five-device, five-trial sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    ("vgg8", "fefet"),
    ("vgg8", "flash"),
    ("vgg8", "pcm"),
    ("vgg8", "reram"),
    ("vgg8", "stt"),
    ("mlp", "fefet"),
    ("mlp", "flash"),
    ("mlp", "pcm"),
    ("mlp", "reram"),
    ("mlp", "stt"),
)
TERMINAL_STATES = {"COMPLETE", "PRUNED"}
DISTRIBUTED_ENV = {
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "WORLD_SIZE",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel-gpus", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    return parser.parse_args()


def visible_gpu_ids(count: int) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    available = (
        [item.strip() for item in configured.split(",") if item.strip()]
        if configured and configured != "-1"
        else [str(index) for index in range(count)]
    )
    if len(available) < count:
        raise ValueError(
            f"parallel-gpus={count} requires {count} visible GPUs, got {available}"
        )
    return available[:count]


def summary_path(output_root: Path, model: str, device: str) -> Path:
    return output_root / model / device / "summary.csv"


def completed(output_root: Path, model: str, device: str, trials: int) -> bool:
    path = summary_path(output_root, model, device)
    if not path.is_file():
        return False
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    terminal = [row for row in rows if row.get("state") in TERMINAL_STATES]
    return len(terminal) >= trials


def environment_for(gpu_id: str) -> dict[str, str]:
    environment = dict(os.environ)
    for key in DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def command(args: argparse.Namespace, model: str, device: str) -> list[str]:
    result = [
        sys.executable,
        "-m",
        "isa.device_sweeps.search",
        "--device",
        device,
        "--model",
        model,
        "--device-config",
        str(ROOT / "configs/device_sweeps/device_params.yaml"),
        "--search-config",
        str(ROOT / "configs/device_sweeps/search_matrix.yaml"),
        "--data-dir",
        str(args.data),
        "--run-root",
        str(args.output_root / model),
        "--study-name",
        f"exact_formula_{model}_{device}_5trial",
        "--seed",
        str(args.seed),
        "--trials",
        str(args.trials),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
    ]
    if args.max_train_steps:
        result.extend(("--max-train-steps", str(args.max_train_steps)))
    if args.max_val_steps:
        result.extend(("--max-val-steps", str(args.max_val_steps)))
    return result


def write_status(output_root: Path, payload: dict) -> None:
    status = output_root / "scheduler_status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    temporary = status.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status)


def main() -> None:
    args = arguments()
    if args.trials != 5:
        raise ValueError("The formal sweep is fixed to exactly five trials per study")
    if not args.data.is_dir():
        raise FileNotFoundError(f"Existing dataset directory is required: {args.data}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    skipped = []
    for task in TASKS:
        if completed(args.output_root, *task, args.trials):
            skipped.append("/".join(task))
        else:
            pending.put(task)
    gpu_ids = visible_gpu_ids(args.parallel_gpus)
    lock = threading.Lock()
    failures: list[dict] = []
    finished: list[str] = []
    running: dict[str, str] = {}

    def snapshot() -> None:
        write_status(
            args.output_root,
            {
                "updated_unix": time.time(),
                "running": dict(running),
                "finished": list(finished),
                "skipped": skipped,
                "failures": list(failures),
                "remaining": pending.qsize(),
            },
        )

    def worker(gpu_id: str) -> None:
        while not failures:
            try:
                model, device = pending.get_nowait()
            except queue.Empty:
                return
            task_name = f"{model}/{device}"
            task_dir = args.output_root / model / device
            task_dir.mkdir(parents=True, exist_ok=True)
            with lock:
                running[gpu_id] = task_name
                snapshot()
            with (task_dir / "launcher.log").open("a", encoding="utf-8") as log:
                log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] GPU={gpu_id}\n")
                log.flush()
                result = subprocess.run(
                    command(args, model, device),
                    cwd=ROOT,
                    env=environment_for(gpu_id),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                running.pop(gpu_id, None)
                if result.returncode == 0 and completed(
                    args.output_root, model, device, args.trials
                ):
                    finished.append(task_name)
                else:
                    failures.append(
                        {"task": task_name, "gpu": gpu_id, "returncode": result.returncode}
                    )
                snapshot()
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,)) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
