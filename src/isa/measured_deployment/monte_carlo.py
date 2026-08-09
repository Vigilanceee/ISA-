#!/usr/bin/env python3
"""Cell-wise empirical Monte Carlo evaluation for measured-state deployment.

The activation-aware state assignment is kept fixed.  For every Monte Carlo
realization, each positive and negative physical cell independently draws one
complete raw measured IV curve from the member set of its assigned global
state.  The draw is fixed for the complete validation set; curves are never
resampled per image, token, voltage point, or batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from isa.measured_deployment.codebook import IVLibrary
from isa.measured_deployment.operator import iter_deployment_fc1
from isa.measured_deployment.posttrain import (
    build_student,
    configure_codebook_on_model,
    evaluate,
    model_config,
)
from isa.vision.data import get_dataloaders
from isa.vision.utils import set_seed


@dataclass
class MemberCurveLibrary:
    vg: torch.Tensor
    currents_a: torch.Tensor
    curve_names: list[str]
    state_to_curve_ids: dict[int, tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cell-wise raw-member IV Monte Carlo evaluation"
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "cifar100"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed-assignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-scale", default="small", choices=("tiny", "mid", "small", "base"))
    parser.add_argument(
        "--codebook-excel", default=str(ROOT / "global_codebook_6train2test.xlsx")
    )
    parser.add_argument("--codebook-sheet", default="cb_all8")
    parser.add_argument("--measured-excel", default=str(ROOT / "measured_iv.xlsx"))
    parser.add_argument(
        "--member-archive",
        default="",
        help=(
            "Optional FG50 member NPZ containing voltage_v, curve_ids, and "
            "raw_current_na. When set, raw member curves are loaded from this "
            "archive instead of legacy per-sheet Excel tabs."
        ),
    )
    parser.add_argument(
        "--state-csv", default=str(ROOT / "global_codebook_maps" / "codebook_states.csv")
    )
    parser.add_argument("--fold-id", default="all8")
    parser.add_argument("--measured-current-scale", type=float, default=1e-9)
    parser.add_argument("--min-valid-current-na", type=float, default=1.0)
    parser.add_argument("--vin-lut-bins", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--evaluate-center", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_state_members(
    state_csv: Path, fold_id: str, codebook_sheet: str
) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    with state_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("fold_id")) != str(fold_id):
                continue
            if row.get("codebook_sheet") != codebook_sheet:
                continue
            state_id = int(row["state_id"])
            members = [item.strip() for item in row["members"].split(";") if item.strip()]
            if not members:
                raise ValueError(f"State {state_id} has no measured members")
            if len(members) != int(row["n_members"]):
                raise ValueError(
                    f"State {state_id} member-count mismatch: {len(members)} vs {row['n_members']}"
                )
            result[state_id] = members
    if not result:
        raise ValueError(
            f"No states found for fold_id={fold_id!r}, codebook_sheet={codebook_sheet!r}"
        )
    expected = list(range(max(result) + 1))
    if sorted(result) != expected:
        raise ValueError("State ids must be contiguous and start at zero")
    return result


def load_member_curve_library(args: argparse.Namespace) -> MemberCurveLibrary:
    state_members = load_state_members(
        Path(args.state_csv), args.fold_id, args.codebook_sheet
    )
    requested_names = {name for members in state_members.values() for name in members}
    if args.member_archive:
        payload = np.load(Path(args.member_archive), allow_pickle=False)
        required = {"voltage_v", "curve_ids", "raw_current_na"}
        missing_keys = required - set(payload.files)
        if missing_keys:
            raise KeyError(
                f"Member archive is missing required arrays: {sorted(missing_keys)}"
            )
        common_vg = torch.from_numpy(
            np.asarray(payload["voltage_v"], dtype=np.float32)
        ).contiguous()
        archive_names = np.asarray(payload["curve_ids"]).reshape(-1).astype(str)
        raw_current = np.asarray(payload["raw_current_na"], dtype=np.float32).reshape(
            archive_names.size, common_vg.numel()
        )
        threshold = float(args.min_valid_current_na)
        clean_current = np.where(raw_current >= threshold, raw_current, 0.0)
        name_to_archive_id = {
            name: curve_id for curve_id, name in enumerate(archive_names.tolist())
        }
        missing = sorted(requested_names - set(name_to_archive_id))
        if missing:
            raise KeyError(
                f"Missing {len(missing)} state-member curves in NPZ archive, "
                f"first entries: {missing[:8]}"
            )

        # Preserve state/member CSV order so Monte Carlo seeds are stable.
        curve_names: list[str] = []
        curve_name_to_id: dict[str, int] = {}
        state_to_curve_ids: dict[int, tuple[int, ...]] = {}
        selected_archive_ids: list[int] = []
        for state_id in sorted(state_members):
            ids: list[int] = []
            for name in state_members[state_id]:
                if name not in curve_name_to_id:
                    curve_name_to_id[name] = len(curve_names)
                    curve_names.append(name)
                    selected_archive_ids.append(name_to_archive_id[name])
                ids.append(curve_name_to_id[name])
            state_to_curve_ids[state_id] = tuple(ids)
        currents = torch.from_numpy(
            clean_current[np.asarray(selected_archive_ids, dtype=np.int64)]
            * float(args.measured_current_scale)
        ).float().contiguous()
        return MemberCurveLibrary(
            vg=common_vg,
            currents_a=currents,
            curve_names=curve_names,
            state_to_curve_ids=state_to_curve_ids,
        )

    sheets = sorted({name.split(":", 1)[0] for name in requested_names})
    raw_curves: dict[str, torch.Tensor] = {}
    common_vg: torch.Tensor | None = None
    for sheet in sheets:
        library = IVLibrary.load(
            Path(args.measured_excel),
            sheet,
            args.measured_current_scale,
            args.min_valid_current_na,
        )
        if common_vg is None:
            common_vg = library.vg
        elif library.vg.shape != common_vg.shape or not torch.allclose(
            library.vg, common_vg, atol=1e-7, rtol=0
        ):
            raise ValueError(f"Measured Vg grid differs in sheet {sheet}")
        for curve_id, value_name in enumerate(library.value_names):
            key = f"{sheet}:{value_name}"
            if key in requested_names:
                raw_curves[key] = library.currents_a[curve_id].clone()
    missing = sorted(requested_names - set(raw_curves))
    if missing:
        raise KeyError(f"Missing {len(missing)} state-member curves, first entries: {missing[:8]}")
    if common_vg is None:
        raise RuntimeError("No raw member curves were loaded")

    # Preserve state/member CSV order so seeds remain reproducible across runs.
    curve_names: list[str] = []
    curve_name_to_id: dict[str, int] = {}
    state_to_curve_ids: dict[int, tuple[int, ...]] = {}
    for state_id in sorted(state_members):
        ids: list[int] = []
        for name in state_members[state_id]:
            if name not in curve_name_to_id:
                curve_name_to_id[name] = len(curve_names)
                curve_names.append(name)
            ids.append(curve_name_to_id[name])
        state_to_curve_ids[state_id] = tuple(ids)
    currents = torch.stack([raw_curves[name] for name in curve_names], dim=0)
    return MemberCurveLibrary(
        vg=common_vg.float().contiguous(),
        currents_a=currents.float().contiguous(),
        curve_names=curve_names,
        state_to_curve_ids=state_to_curve_ids,
    )


def load_state_assignment(path: Path, model: torch.nn.Module) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_id, layer in iter_deployment_fc1(model):
        pos_key = f"blocks.{block_id}.pos_idx"
        neg_key = f"blocks.{block_id}.neg_idx"
        if pos_key not in payload or neg_key not in payload:
            raise KeyError(f"Fixed assignment is missing {pos_key!r} or {neg_key!r}")
        pos = payload[pos_key].long().contiguous()
        neg = payload[neg_key].long().contiguous()
        expected = (layer.out_features, layer.in_features)
        if tuple(pos.shape) != expected or tuple(neg.shape) != expected:
            raise ValueError(
                f"Block {block_id} assignment shape is {tuple(pos.shape)}/{tuple(neg.shape)}, expected {expected}"
            )
        result[block_id] = (pos, neg)
    return result


def sample_curve_ids(
    state_ids: torch.Tensor,
    state_to_curve_ids: dict[int, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    flat_state = state_ids.reshape(-1).cpu().long()
    sampled = torch.empty_like(flat_state, dtype=torch.int32)
    for state_id in torch.unique(flat_state, sorted=True).tolist():
        if state_id not in state_to_curve_ids:
            raise KeyError(f"Assignment references unknown state {state_id}")
        mask = flat_state == state_id
        members = torch.tensor(state_to_curve_ids[state_id], dtype=torch.int32)
        if members.numel() == 1:
            sampled[mask] = members[0]
        else:
            choices = torch.randint(
                members.numel(), (int(mask.sum().item()),), generator=generator
            )
            sampled[mask] = members[choices]
    return sampled.reshape(state_ids.shape)


@torch.no_grad()
def set_center_assignment(
    model: torch.nn.Module,
    state_assignment: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for block_id, layer in iter_deployment_fc1(model):
        pos, neg = state_assignment[block_id]
        layer.set_assignment(pos.to(layer.vth_pos.device), neg.to(layer.vth_pos.device))


@torch.no_grad()
def set_random_member_realization(
    model: torch.nn.Module,
    state_assignment: dict[int, tuple[torch.Tensor, torch.Tensor]],
    member_library: MemberCurveLibrary,
    seed: int,
) -> dict[str, int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    all_sampled: list[torch.Tensor] = []
    total_cells = 0
    singleton_cells = 0
    for block_id, layer in iter_deployment_fc1(model):
        pos_state, neg_state = state_assignment[block_id]
        pos_curve = sample_curve_ids(
            pos_state, member_library.state_to_curve_ids, generator
        )
        neg_curve = sample_curve_ids(
            neg_state, member_library.state_to_curve_ids, generator
        )
        layer.set_assignment(
            pos_curve.to(layer.vth_pos.device), neg_curve.to(layer.vth_pos.device)
        )
        all_sampled.extend((pos_curve.flatten(), neg_curve.flatten()))
        total_cells += pos_state.numel() + neg_state.numel()
        for state_tensor in (pos_state, neg_state):
            for state_id in torch.unique(state_tensor).tolist():
                if len(member_library.state_to_curve_ids[int(state_id)]) == 1:
                    singleton_cells += int((state_tensor == state_id).sum().item())
    unique_curves = int(torch.unique(torch.cat(all_sampled)).numel())
    return {
        "total_physical_cells": int(total_cells),
        "singleton_state_cells": int(singleton_cells),
        "unique_raw_curves_used": unique_curves,
    }


def completed_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]) for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    if args.num_seeds < 0 or args.seed_stride <= 0:
        raise ValueError("--num-seeds must be non-negative and --seed-stride positive")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed_start)
    random.seed(args.seed_start)
    np.random.seed(args.seed_start)
    use_amp = not args.no_amp and device.type == "cuda"

    cfg = model_config(args.model_scale)
    model = build_student(cfg, Path(args.checkpoint), device)
    state_assignment = load_state_assignment(Path(args.fixed_assignment), model)
    center_library = IVLibrary.load(
        Path(args.codebook_excel),
        args.codebook_sheet,
        args.measured_current_scale,
        args.min_valid_current_na,
    )
    state_count = len(load_state_members(Path(args.state_csv), args.fold_id, args.codebook_sheet))
    if center_library.currents_a.shape[0] != state_count:
        raise ValueError(
            f"Center codebook has {center_library.currents_a.shape[0]} curves but state CSV has {state_count} states"
        )

    _, val_loader = get_dataloaders(
        args.data,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        distributed=False,
    )
    center_metrics_path = out_dir / "center_metrics.json"
    if args.evaluate_center and not center_metrics_path.exists():
        configure_codebook_on_model(model, center_library, args.vin_lut_bins)
        set_center_assignment(model, state_assignment)
        print("Evaluating deterministic center-curve deployment...", flush=True)
        metrics = evaluate(
            model, model, val_loader, device, "measured", use_amp, args.max_val_batches
        )
        atomic_json(
            center_metrics_path,
            {
                "status": "completed",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "assignment": str(Path(args.fixed_assignment).resolve()),
                "curve_realization": "global_state_center",
                **metrics,
            },
        )
        print(f"center accuracy={metrics['accuracy']:.4f}%", flush=True)

    member_library = load_member_curve_library(args)
    for _, layer in iter_deployment_fc1(model):
        layer.configure_codebook(
            member_library.vg, member_library.currents_a, args.vin_lut_bins
        )

    samples_path = out_dir / "samples.csv"
    done = completed_seeds(samples_path)
    seeds = [args.seed_start + i * args.seed_stride for i in range(args.num_seeds)]
    for run_id, seed in enumerate(seeds, start=1):
        if seed in done:
            print(f"[SKIP] seed={seed} already completed", flush=True)
            continue
        started = time.perf_counter()
        realization = set_random_member_realization(
            model, state_assignment, member_library, seed
        )
        metrics = evaluate(
            model, model, val_loader, device, "measured", use_amp, args.max_val_batches
        )
        row: dict[str, Any] = {
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "images": int(metrics["images"]),
            "inference_seconds": metrics["seconds"],
            "realization_seconds": time.perf_counter() - started - metrics["seconds"],
            **realization,
        }
        append_csv(samples_path, row)
        done.add(seed)
        print(
            f"seed={seed} ({run_id}/{len(seeds)}) accuracy={metrics['accuracy']:.4f}% "
            f"unique_curves={realization['unique_raw_curves_used']}",
            flush=True,
        )

    member_counts = [len(member_library.state_to_curve_ids[i]) for i in range(state_count)]
    atomic_json(
        out_dir / "run_metadata.json",
        {
            "status": "completed",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "fixed_assignment": str(Path(args.fixed_assignment).resolve()),
            "codebook_excel": str(Path(args.codebook_excel).resolve()),
            "codebook_sheet": args.codebook_sheet,
            "measured_excel": str(Path(args.measured_excel).resolve()),
            "state_csv": str(Path(args.state_csv).resolve()),
            "sampling": "independent_per_physical_cell_uniform_over_state_members",
            "sampling_lifetime": "one_complete_IV_curve_fixed_for_full_validation_set",
            "state_count": state_count,
            "raw_member_curve_count": len(member_library.curve_names),
            "state_member_count_min": min(member_counts),
            "state_member_count_max": max(member_counts),
            "state_member_count_mean": float(np.mean(member_counts)),
            "seed_start": args.seed_start,
            "seed_stride": args.seed_stride,
            "num_seeds_requested": args.num_seeds,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
