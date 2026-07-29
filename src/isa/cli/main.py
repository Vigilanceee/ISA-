"""Unified launcher for training matrices and benchmark evaluation."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {source}")
    return payload


def format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(context)
    if isinstance(value, list):
        return [format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_value(item, context) for key, item in value.items()}
    return value


def to_cli_args(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        option = f"--{key}"
        if value is True:
            result.append(option)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            result.append(option)
            result.extend(str(item) for item in value)
        else:
            result.extend((option, str(value)))
    return result


def launcher_prefix(module: str, gpus: int) -> list[str]:
    if gpus <= 1:
        return [sys.executable, "-m", module]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        module,
    ]


def experiment_context(
    matrix: dict[str, Any],
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> dict[str, Any]:
    defaults = dict(matrix.get("paths", {}))
    defaults.update(profile.get("paths", {}))
    if args.data:
        defaults["data"] = args.data
    if args.output_root:
        defaults["output_root"] = args.output_root
    return defaults


def build_experiment(
    matrix_path: str,
    experiment_id: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    matrix = load_yaml(matrix_path)
    profile = load_yaml(args.profile) if args.profile else {}
    experiments = matrix.get("experiments", {})
    if experiment_id not in experiments:
        choices = ", ".join(experiments)
        raise KeyError(f"Unknown experiment {experiment_id!r}; choose one of: {choices}")

    spec = experiments[experiment_id]
    gpus = int(args.gpus or profile.get("gpus", 1))
    if gpus < 1:
        raise ValueError("GPU count must be positive")

    context = experiment_context(matrix, args, profile)
    context["gpus"] = gpus
    context["experiment"] = experiment_id
    defaults = format_value(matrix.get("defaults", {}), context)
    values = dict(defaults.get("args", {}))
    values.update(format_value(spec.get("args", {}), context))

    global_batch = spec.get("global_batch_size", defaults.get("global_batch_size"))
    batch_arg = spec.get("batch_arg", defaults.get("batch_arg"))
    if global_batch is not None and batch_arg:
        global_batch = int(global_batch)
        if global_batch % gpus:
            raise ValueError(
                f"global_batch_size={global_batch} is not divisible by gpus={gpus}"
            )
        values[batch_arg] = global_batch // gpus

    artifact = format_value(spec.get("completion", ""), context)
    success_marker = format_value(
        spec.get("success_marker", defaults.get("success_marker", "")),
        context,
    )
    completion = success_marker or artifact
    resume_pattern = format_value(spec.get("resume", ""), context)
    resume_candidates = [
        Path(candidate)
        for candidate in glob.glob(resume_pattern)
        if Path(candidate).is_file()
    ]
    resume = (
        str(max(resume_candidates, key=lambda path: path.stat().st_mtime))
        if resume_candidates
        else resume_pattern
    )
    if args.resume == "auto" and resume and Path(resume).is_file() and not (
        completion and Path(completion).is_file()
    ):
        values[spec.get("resume_arg", defaults.get("resume_arg", "resume"))] = resume

    module = str(spec.get("module", defaults.get("module", "")))
    if not module:
        raise ValueError(f"No Python module configured for {experiment_id}")

    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in matrix.get("environment", {}).items()})
    environment.update({str(k): str(v) for k, v in profile.get("environment", {}).items()})
    environment.update({str(k): str(v) for k, v in spec.get("environment", {}).items()})

    command = launcher_prefix(module, gpus) + to_cli_args(values)
    metadata = {
        "id": experiment_id,
        "gpus": gpus,
        "global_batch_size": global_batch,
        "completion": completion,
        "artifact": artifact,
        "resume": resume,
        "command": command,
        "context": context,
    }
    return command, environment, metadata


def write_manifest(metadata: dict[str, Any], status: str, returncode: int | None) -> None:
    context = metadata.get("context", {})
    output_root = Path(context.get("output_root", "outputs"))
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "status": status,
        "returncode": returncode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    target = manifest_dir / f"{metadata['id']}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_experiment(matrix_path: str, experiment_id: str, args: argparse.Namespace) -> int:
    command, environment, metadata = build_experiment(matrix_path, experiment_id, args)
    completion = metadata.get("completion")
    if completion and Path(completion).is_file() and args.resume == "auto":
        print(f"[skip] {experiment_id}: {completion}")
        write_manifest(metadata, "completed", 0)
        return 0

    print(f"[run] {experiment_id}")
    print(shlex.join(command))
    if args.dry_run:
        return 0

    write_manifest(metadata, "running", None)
    result = subprocess.run(command, env=environment, check=False)
    status = "completed" if result.returncode == 0 else "failed"
    if result.returncode == 0 and completion:
        marker = Path(completion)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "experiment": experiment_id,
                    "artifact": metadata.get("artifact"),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    write_manifest(metadata, status, result.returncode)
    return result.returncode


def matrix_ids(matrix: dict[str, Any], selection: list[str] | None) -> list[str]:
    configured = list(matrix.get("order", matrix.get("experiments", {}).keys()))
    if not selection:
        return configured
    unknown = sorted(set(selection) - set(configured))
    if unknown:
        raise KeyError(f"Unknown experiments: {', '.join(unknown)}")
    return [item for item in configured if item in selection]


def command_train(args: argparse.Namespace) -> int:
    return run_experiment(args.config, args.experiment, args)


def command_matrix(args: argparse.Namespace) -> int:
    matrix = load_yaml(args.config)
    failures: list[str] = []
    for experiment_id in matrix_ids(matrix, args.experiments):
        returncode = run_experiment(args.config, experiment_id, args)
        if returncode:
            failures.append(experiment_id)
            if not args.keep_going:
                break
    if failures:
        print(f"[failed] {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def command_list(args: argparse.Namespace) -> int:
    matrix = load_yaml(args.config)
    for experiment_id in matrix_ids(matrix, None):
        description = matrix["experiments"][experiment_id].get("description", "")
        print(f"{experiment_id:20s} {description}")
    return 0


def evaluation_command(
    spec: dict[str, Any],
    model_id: str,
    gpu: int,
    context: dict[str, Any],
    dry_run: bool,
) -> int:
    model = format_value(spec["models"][model_id], context)
    values = dict(format_value(spec.get("args", {}), context))
    values.update(model.get("args", {}))
    command = [sys.executable, "-m", spec["module"]] + to_cli_args(values)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[eval:{gpu}] {model_id}: {shlex.join(command)}")
    if dry_run:
        return 0
    return subprocess.run(command, env=environment, check=False).returncode


def evaluation_worker(
    spec: dict[str, Any],
    model_ids: list[str],
    gpu: int,
    context: dict[str, Any],
    dry_run: bool,
) -> list[str]:
    failures: list[str] = []
    for model_id in model_ids:
        if evaluation_command(spec, model_id, gpu, context, dry_run):
            failures.append(model_id)
    return failures


def command_evaluate(args: argparse.Namespace) -> int:
    matrix = load_yaml(args.config)
    spec = matrix.get("evaluation")
    if not spec:
        raise ValueError(f"No evaluation section in {args.config}")
    profile = load_yaml(args.profile) if args.profile else {}
    gpus = int(args.gpus or profile.get("gpus", 1))
    context = experiment_context(matrix, args, profile)
    model_ids = list(spec["models"])
    if args.experiments:
        model_ids = [item for item in model_ids if item in args.experiments]

    assignments = [model_ids[gpu::gpus] for gpu in range(gpus)]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=gpus) as executor:
        futures = [
            executor.submit(
                evaluation_worker,
                spec,
                assigned,
                gpu,
                context,
                args.dry_run,
            )
            for gpu, assigned in enumerate(assignments)
            if assigned
        ]
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        print(f"[failed] {', '.join(sorted(failures))}", file=sys.stderr)
        return 1
    for postprocess in spec.get("postprocess", []):
        values = format_value(postprocess.get("args", {}), context)
        command = [sys.executable, "-m", postprocess["module"]] + to_cli_args(values)
        print(f"[postprocess] {shlex.join(command)}")
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode
    return 0


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment-matrix YAML")
    parser.add_argument("--profile", default="", help="Optional resource-profile YAML")
    parser.add_argument("--gpus", type=int, default=0, help="Override GPU count")
    parser.add_argument("--data", default="", help="Override the matrix data path")
    parser.add_argument("--output-root", default="", help="Override output root")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run one configured experiment")
    add_shared_options(train)
    train.add_argument("--experiment", required=True)
    train.set_defaults(handler=command_train)

    matrix = subparsers.add_parser("matrix", help="Run a configured experiment matrix")
    add_shared_options(matrix)
    matrix.add_argument("--experiments", nargs="*")
    matrix.add_argument("--keep-going", action="store_true")
    matrix.set_defaults(handler=command_matrix)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate configured checkpoints")
    add_shared_options(evaluate)
    evaluate.add_argument("--experiments", nargs="*")
    evaluate.set_defaults(handler=command_evaluate)

    listing = subparsers.add_parser("list", help="List matrix experiments")
    listing.add_argument("--config", required=True)
    listing.set_defaults(handler=command_list)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))
