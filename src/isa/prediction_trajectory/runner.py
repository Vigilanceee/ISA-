"""Resume-aware multi-GPU scheduler for independent device/seed runs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from torchvision import datasets

from isa.prediction_trajectory.protocol import atomic_json, ensure_probe_file, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device-config", default="configs/device_sweeps/device_params.yaml")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    return parser.parse_args()


def prepare_probe(config: dict, data_dir: str, output_root: Path) -> None:
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=False)
    probe = dict(config["probe"])
    indices, labels = ensure_probe_file(
        labels=dataset.targets,
        path=output_root / "probe_indices.npz",
        size=int(probe["size"]),
        seed=int(probe["seed"]),
    )
    counts = {int(class_id): int(np.count_nonzero(labels == class_id)) for class_id in np.unique(labels)}
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError(f"probe is not stratified: {counts}")
    print(
        "TRAJECTORY_PROBE="
        + json.dumps(
            {
                "size": int(indices.size),
                "seed": int(probe["seed"]),
                "class_counts": counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_worker(
    gpu: str,
    tasks: queue.Queue[tuple[str, int]],
    *,
    args: argparse.Namespace,
    epochs: int,
) -> list[str]:
    failures: list[str] = []
    output_root = Path(args.output_root)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            device, seed = tasks.get_nowait()
        except queue.Empty:
            break
        task_name = f"{device}_seed_{seed}"
        completion = output_root / "raw" / device / f"seed_{seed}" / "completed.json"
        if completion.is_file():
            print(f"MATRIX_SKIP task={task_name}", flush=True)
            tasks.task_done()
            continue
        command = [
            sys.executable,
            "-m",
            "isa.prediction_trajectory.train",
            "--config",
            args.config,
            "--device",
            device,
            "--seed",
            str(seed),
            "--data-dir",
            args.data_dir,
            "--output-root",
            args.output_root,
            "--device-config",
            args.device_config,
            "--epochs",
            str(epochs),
        ]
        if args.max_train_steps > 0:
            command.extend(("--max-train-steps", str(args.max_train_steps)))
        if args.max_val_steps > 0:
            command.extend(("--max-val-steps", str(args.max_val_steps)))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"MATRIX_START gpu={gpu} task={task_name}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            tasks.task_done()
            continue
        with (logs / f"{task_name}.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\nMATRIX_ATTEMPT_START={time.strftime('%Y-%m-%dT%H:%M:%S%z')} gpu={gpu}\n")
            handle.flush()
            result = subprocess.run(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            failures.append(task_name)
            print(
                f"MATRIX_FAIL gpu={gpu} task={task_name} returncode={result.returncode}",
                flush=True,
            )
        else:
            print(f"MATRIX_DONE gpu={gpu} task={task_name}", flush=True)
        tasks.task_done()
    return failures


def validate_smoke(output_root: Path, device: str, seed: int) -> None:
    snapshot_paths = sorted((output_root / "raw" / device / f"seed_{seed}" / "snapshots").glob("epoch_*.npz"))
    if len(snapshot_paths) != 3:
        raise RuntimeError(f"10-epoch smoke expected three snapshots, found {len(snapshot_paths)}")
    for path in snapshot_paths:
        with np.load(path, allow_pickle=False) as payload:
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        if probabilities.shape != (1000, 10):
            raise RuntimeError(f"invalid smoke probability shape: {path}")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4):
            raise RuntimeError(f"invalid smoke probabilities: {path}")
    print("TRAJECTORY_SMOKE_VALIDATED snapshots=3 shape=1000x10", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    devices = [str(device) for device in config["devices"]]
    seeds = [int(seed) for seed in config["seeds"]]
    epochs = int(config["training"]["epochs"])
    if args.smoke:
        devices = ["flash"]
        seeds = [0]
        epochs = 10

    task_queue: queue.Queue[tuple[str, int]] = queue.Queue()
    for device in devices:
        for seed in seeds:
            task_queue.put((device, seed))
    if args.dry_run:
        for gpu, task in zip(
            (gpu_ids[index % len(gpu_ids)] for index in range(task_queue.qsize())),
            list(task_queue.queue),
        ):
            print(f"DRY_RUN gpu={gpu} device={task[0]} seed={task[1]}")
        return

    prepare_probe(config, args.data_dir, output_root)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_worker, gpu, task_queue, args=args, epochs=epochs) for gpu in gpu_ids]
        failures = [failure for future in futures for failure in future.result()]
    if failures:
        raise RuntimeError("failed prediction-trajectory runs: " + ", ".join(failures))
    if args.smoke:
        validate_smoke(output_root, devices[0], seeds[0])
    atomic_json(
        output_root / "matrix_completed.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "devices": devices,
            "seeds": seeds,
            "epochs": epochs,
            "gpus": gpu_ids,
        },
    )
    print("TRAJECTORY_MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
