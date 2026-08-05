"""Resumable FeFET VGG8 health-run launcher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

TERMINAL_STATES = {"COMPLETE", "PRUNED"}
INHERITED_DISTRIBUTED_ENV = {
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


def validate_execution_options(
    *, parallel_seeds: int, nproc_per_node: int
) -> None:
    if parallel_seeds < 1:
        raise ValueError("parallel_seeds must be at least 1")
    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be at least 1")
    if parallel_seeds > 1 and nproc_per_node > 1:
        raise ValueError(
            "--parallel-seeds and --nproc-per-node>1 are mutually exclusive; "
            "seed parallelism runs one independent process per GPU"
        )


def worker_gpu_ids(
    worker_count: int, environ: dict[str, str] | None = None
) -> list[str]:
    """Resolve one explicit CUDA-visible identifier for every seed worker."""

    environment = os.environ if environ is None else environ
    configured = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    available = (
        [value.strip() for value in configured.split(",") if value.strip()]
        if configured and configured != "-1"
        else [str(index) for index in range(worker_count)]
    )
    if len(available) < worker_count:
        raise ValueError(
            f"parallel_seeds={worker_count} requires {worker_count} visible GPUs, "
            f"but CUDA_VISIBLE_DEVICES exposes {available}"
        )
    return available[:worker_count]


def independent_seed_environment(
    gpu_id: str, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Create a single-process environment inside an ACP multi-GPU worker.

    ACP exports torch-distributed rendezvous variables for the worker even when
    the user command is not launched with torchrun.  Independent seed workers
    must not inherit those variables, otherwise both processes identify as
    rank zero and contend for the platform master port.
    """

    environment = dict(os.environ if environ is None else environ)
    for key in INHERITED_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def load_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a mapping: {path}")
    return payload


def validate_manifest(manifest: dict, repo_root: Path) -> None:
    if manifest.get("device") != "fefet" or manifest.get("model") != "vgg8":
        raise ValueError("This runner only accepts device=fefet and model=vgg8")
    seeds = manifest.get("seeds", [])
    if len(seeds) != 5 or len(set(map(int, seeds))) != 5:
        raise ValueError("Exactly five unique seeds are required")

    config_path = repo_root / str(manifest["device_config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fefet = config["fefet"]
    if fefet.get("conv_backend") != "reference":
        raise ValueError("FeFET conv_backend must be reference")
    if fefet.get("linear_backend") != "reference":
        raise ValueError("FeFET linear_backend must be reference")
    if fefet.get("raw_kernel_backend") != "split":
        raise ValueError("FeFET raw_kernel_backend must be split")
    if fefet.get("raw_forward_backend") != "split_k":
        raise ValueError("FeFET raw_forward_backend must be split_k")
    if bool(fefet.get("lut_enabled", False)):
        raise ValueError("FeFET LUT must be disabled")


def build_command(
    manifest: dict,
    *,
    repo_root: Path,
    data_dir: Path,
    seed_dir: Path,
    seed: int,
    nproc_per_node: int,
) -> list[str]:
    hp = manifest["fixed_hyperparameters"]
    health = manifest["health"]
    execution = manifest.get("execution", {})
    python_prefix = [sys.executable]
    if nproc_per_node > 1:
        python_prefix += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(nproc_per_node),
        ]
    command = python_prefix + [
        "-m",
        "isa.device_sweeps.search",
        "--device",
        "fefet",
        "--model",
        "vgg8",
        "--device-config",
        str(repo_root / manifest["device_config"]),
        "--search-config",
        str(repo_root / manifest["search_config"]),
        "--data-dir",
        str(data_dir),
        "--run-root",
        str(seed_dir),
        "--study-name",
        f"fefet_raw_vgg8_seed_{seed}",
        "--seed",
        str(seed),
        "--trials",
        "1",
        "--epochs",
        str(manifest.get("epochs", 200)),
        "--batch-size",
        str(execution.get("batch_size", 128)),
        "--workers",
        str(execution.get("workers", 4)),
        "--checkpoint-interval",
        str(execution.get("checkpoint_interval", 5)),
        "--conv-backend",
        "reference",
        "--linear-backend",
        "reference",
        "--fixed-lr",
        str(hp["lr"]),
        "--fixed-init-center",
        str(hp["init_center"]),
        "--fixed-init-half-width",
        str(hp["init_half_width"]),
        "--fixed-tia-r",
        str(hp["tia_r"]),
        "--health-pruning",
        "--health-epoch8-min-best",
        str(health["epoch8_min_best"]),
        "--health-epoch20-min-best",
        str(health["epoch20_min_best"]),
        "--health-plateau-window",
        str(health["plateau_window"]),
        "--health-plateau-min-gain",
        str(health["plateau_min_gain"]),
    ]
    return command


def terminal_record(seed_dir: Path) -> dict | None:
    marker = seed_dir / "terminal.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text(encoding="utf-8"))
    return record if record.get("state") in TERMINAL_STATES else None


def write_terminal_record(seed_dir: Path, seed: int) -> dict:
    summary_path = seed_dir / "fefet" / "summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Missing terminal summary: {summary_path}")
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["state"] not in TERMINAL_STATES:
        raise RuntimeError(f"Expected one terminal trial in {summary_path}: {rows}")
    row = rows[0]
    record = {
        "seed": seed,
        "state": row["state"],
        "best_val_acc": float(row["best_val_acc"]),
        "last_epoch": int(row["last_epoch"]),
        "stop_reason": row.get("stop_reason", ""),
        "summary": str(summary_path),
    }
    marker = seed_dir / "terminal.json"
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return record


def run_seed(
    manifest: dict,
    *,
    repo_root: Path,
    data_dir: Path,
    output_root: Path,
    seed: int,
    nproc_per_node: int,
    gpu_id: str | None,
    dry_run: bool,
) -> dict:
    """Run or resume one seed and persist its launcher output."""

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    existing = terminal_record(seed_dir)
    if existing is not None:
        print(f"skip seed={seed} state={existing['state']}", flush=True)
        return existing

    command = build_command(
        manifest,
        repo_root=repo_root,
        data_dir=data_dir,
        seed_dir=seed_dir,
        seed=seed,
        nproc_per_node=nproc_per_node,
    )
    gpu_message = gpu_id if gpu_id is not None else "DDP/inherited"
    print(
        f"launch seed={seed} gpu={gpu_message} command={' '.join(command)}",
        flush=True,
    )
    if dry_run:
        return {"seed": seed, "state": "DRY_RUN", "gpu_id": gpu_id}

    environment = (
        independent_seed_environment(gpu_id)
        if gpu_id is not None
        else os.environ.copy()
    )
    log_path = seed_dir / "launcher.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(
            f"\n=== seed={seed} CUDA_VISIBLE_DEVICES="
            f"{environment.get('CUDA_VISIBLE_DEVICES', '')} ===\n"
        )
        log_handle.write("command=" + " ".join(command) + "\n")
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record = write_terminal_record(seed_dir, seed)
    print(
        f"terminal seed={seed} state={record['state']} "
        f"best={record['best_val_acc']:.4f} "
        f"reason={record['stop_reason'] or 'completed'}",
        flush=True,
    )
    return record


def run_seed_group(
    seeds: list[int],
    *,
    gpu_id: str,
    stop_event: threading.Event,
    run_kwargs: dict,
) -> list[dict]:
    records = []
    for seed in seeds:
        if stop_event.is_set():
            break
        try:
            records.append(
                run_seed(
                    seed=seed,
                    gpu_id=gpu_id,
                    nproc_per_node=1,
                    **run_kwargs,
                )
            )
        except Exception:
            stop_event.set()
            raise
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=1,
        help="Run independent seeds concurrently, one per visible GPU.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_execution_options(
        parallel_seeds=args.parallel_seeds,
        nproc_per_node=args.nproc_per_node,
    )
    manifest_path = args.manifest.resolve()
    repo_root = manifest_path.parents[2]
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, repo_root)
    data_dir = (args.data_dir or repo_root / manifest["data_dir"]).resolve()
    output_root = (
        args.output_root or repo_root / manifest["output_root"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in manifest["seeds"]]
    shared_kwargs = {
        "manifest": manifest,
        "repo_root": repo_root,
        "data_dir": data_dir,
        "output_root": output_root,
        "dry_run": args.dry_run,
    }

    if args.parallel_seeds == 1:
        gpu_id = (
            None
            if args.nproc_per_node > 1
            else worker_gpu_ids(1)[0]
        )
        for seed in seeds:
            run_seed(
                seed=seed,
                gpu_id=gpu_id,
                nproc_per_node=args.nproc_per_node,
                **shared_kwargs,
            )
        return

    gpu_ids = worker_gpu_ids(args.parallel_seeds)
    seed_groups = [
        seeds[index :: args.parallel_seeds]
        for index in range(args.parallel_seeds)
    ]
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=args.parallel_seeds) as executor:
        futures = [
            executor.submit(
                run_seed_group,
                seed_group,
                gpu_id=gpu_id,
                stop_event=stop_event,
                run_kwargs=shared_kwargs,
            )
            for seed_group, gpu_id in zip(seed_groups, gpu_ids, strict=True)
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
