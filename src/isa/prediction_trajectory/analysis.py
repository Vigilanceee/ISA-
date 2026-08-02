"""Offline prediction-trajectory analysis using PCA, cosine, and Euclidean distance."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isa.prediction_trajectory.protocol import atomic_json, load_config


@dataclass(frozen=True)
class RunTrajectory:
    device: str
    seed: int
    epochs: np.ndarray
    accuracies: np.ndarray
    losses: np.ndarray
    probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def joint_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact PCA coordinates and variance ratios via the checkpoint Gram matrix."""

    observations = np.asarray(matrix, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[0] < 2:
        raise ValueError("PCA requires at least two two-dimensional observations")
    centered = observations - observations.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > np.finfo(np.float64).eps * max(eigenvalues[0], 1.0)
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("PCA observations have zero or non-finite total variance")
    coordinates = eigenvectors * np.sqrt(eigenvalues)[None, :]
    return coordinates, eigenvalues / total


def interpolate_first_crossing(
    epochs: np.ndarray,
    accuracies: np.ndarray,
    vectors: np.ndarray,
    milestone: float,
) -> tuple[np.ndarray, float]:
    """Linearly interpolate the first pair of snapshots that crosses a milestone."""

    epoch_values = np.asarray(epochs, dtype=np.float64)
    accuracy_values = np.asarray(accuracies, dtype=np.float64)
    prediction_values = np.asarray(vectors, dtype=np.float64)
    if prediction_values.shape[0] != accuracy_values.size or epoch_values.size != accuracy_values.size:
        raise ValueError("epoch, accuracy, and prediction lengths differ")
    reached = np.flatnonzero(accuracy_values >= milestone)
    if reached.size == 0:
        raise ValueError(f"trajectory never reaches accuracy {milestone:.6f}")
    upper = int(reached[0])
    if upper == 0:
        return prediction_values[0].copy(), float(epoch_values[0])
    lower = upper - 1
    low_accuracy = float(accuracy_values[lower])
    high_accuracy = float(accuracy_values[upper])
    if high_accuracy <= low_accuracy:
        weight = 1.0
    else:
        weight = (milestone - low_accuracy) / (high_accuracy - low_accuracy)
        weight = float(np.clip(weight, 0.0, 1.0))
    vector = (1.0 - weight) * prediction_values[lower] + weight * prediction_values[upper]
    epoch = (1.0 - weight) * epoch_values[lower] + weight * epoch_values[upper]
    return vector, float(epoch)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def normalized_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return float(np.linalg.norm(difference) / math.sqrt(difference.size))


def pairwise_distances(vectors: Iterable[np.ndarray]) -> np.ndarray:
    values = list(vectors)
    return np.asarray(
        [normalized_distance(values[i], values[j]) for i, j in itertools.combinations(range(len(values)), 2)],
        dtype=np.float64,
    )


def select_endpoint_index(accuracies: np.ndarray, endpoint: str) -> int:
    """Select the best-validation or final snapshot from one trajectory."""

    values = np.asarray(accuracies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("endpoint selection requires finite one-dimensional accuracies")
    if endpoint == "best":
        return int(np.argmax(values))
    if endpoint == "final":
        return int(values.size - 1)
    raise ValueError(f"unknown endpoint: {endpoint}")


def ratio_and_bootstrap(
    between: np.ndarray,
    within: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    between_values = np.asarray(between, dtype=np.float64)
    within_values = np.asarray(within, dtype=np.float64)
    if between_values.size == 0 or within_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    denominator = float(within_values.mean())
    ratio = float(between_values.mean() / denominator) if denominator > 0 else float("nan")
    samples = np.full(repeats, np.nan, dtype=np.float64)
    for index in range(repeats):
        sampled_between = rng.choice(between_values, size=between_values.size, replace=True)
        sampled_within = rng.choice(within_values, size=within_values.size, replace=True)
        sampled_denominator = float(sampled_within.mean())
        if sampled_denominator > 0:
            samples[index] = float(sampled_between.mean() / sampled_denominator)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return ratio, float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return ratio, float(low), float(high)


def load_run(run_dir: Path, device: str, seed: int) -> RunTrajectory:
    snapshot_paths = sorted((run_dir / "snapshots").glob("epoch_*.npz"))
    if not snapshot_paths:
        raise FileNotFoundError(f"no prediction snapshots in {run_dir}")
    epochs: list[int] = []
    accuracies: list[float] = []
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    for path in snapshot_paths:
        with np.load(path, allow_pickle=False) as payload:
            epoch = int(payload["epoch"])
            probability = np.asarray(payload["probabilities"], dtype=np.float32)
            accuracy = float(payload["validation_accuracy"])
            loss = float(payload["validation_loss"])
        if probability.shape != (1000, 10):
            raise ValueError(f"unexpected probability shape in {path}: {probability.shape}")
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=2e-4):
            raise ValueError(f"invalid probability values in {path}")
        epochs.append(epoch)
        accuracies.append(accuracy)
        losses.append(loss)
        predictions.append(probability.reshape(-1))
    if len(set(epochs)) != len(epochs) or epochs != sorted(epochs):
        raise ValueError(f"duplicate or unsorted snapshots in {run_dir}")
    return RunTrajectory(
        device=device,
        seed=seed,
        epochs=np.asarray(epochs, dtype=np.int32),
        accuracies=np.asarray(accuracies, dtype=np.float64),
        losses=np.asarray(losses, dtype=np.float64),
        probabilities=np.stack(predictions).astype(np.float32, copy=False),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_milestones(runs: list[RunTrajectory], analysis: dict) -> np.ndarray:
    common_low = max(float(run.accuracies[0]) for run in runs)
    common_high = min(float(np.max(run.accuracies)) for run in runs)
    if common_high <= common_low + 1e-6:
        raise ValueError(f"no common accuracy interval: low={common_low:.6f}, high={common_high:.6f}")
    candidates = np.asarray(analysis["candidate_accuracy_milestones"], dtype=np.float64)
    selected = candidates[(candidates > common_low + 1e-6) & (candidates <= common_high + 1e-12)]
    if selected.size < 4:
        count = int(analysis.get("milestone_count_fallback", 5))
        margin = 0.08 * (common_high - common_low)
        selected = np.linspace(common_low + margin, common_high, count)
    if selected.size > 6:
        selected = selected[:6]
    return selected


def effect_components(
    vectors_by_device: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    within: list[float] = []
    centroids: list[np.ndarray] = []
    for device in sorted(vectors_by_device):
        values = vectors_by_device[device]
        within.extend(pairwise_distances(values).tolist())
        centroids.append(np.mean(np.stack(values), axis=0))
    between = pairwise_distances(centroids)
    return between, np.asarray(within, dtype=np.float64)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root)
    devices = tuple(config["devices"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    expected_snapshot_count = (
        int(config["training"]["epochs"]) // int(config["training"]["snapshot_interval"]) + 1
    )
    runs: list[RunTrajectory] = []
    missing: list[str] = []
    for device in devices:
        for seed in seeds:
            run_dir = output_root / "raw" / device / f"seed_{seed}"
            if not (run_dir / "completed.json").is_file():
                missing.append(f"{device}/seed_{seed}")
                continue
            run = load_run(run_dir, device, seed)
            if len(run.epochs) != expected_snapshot_count:
                raise ValueError(
                    f"{device}/seed_{seed}: expected {expected_snapshot_count} snapshots, "
                    f"found {len(run.epochs)}"
                )
            runs.append(run)
    if missing and not args.allow_incomplete:
        raise RuntimeError("incomplete runs: " + ", ".join(missing))
    if len(runs) < 2:
        raise RuntimeError("analysis requires at least two completed runs")

    matrix = np.concatenate([run.probabilities for run in runs], axis=0)
    coordinates, explained_ratio = joint_pca(matrix)
    metadata_rows: list[dict[str, object]] = []
    cursor = 0
    for run in runs:
        for index, epoch in enumerate(run.epochs):
            metadata_rows.append(
                {
                    "device": run.device,
                    "seed": run.seed,
                    "epoch": int(epoch),
                    "validation_accuracy": float(run.accuracies[index]),
                    "validation_loss": float(run.losses[index]),
                    "pc1": float(coordinates[cursor, 0]),
                    "pc2": float(coordinates[cursor, 1]) if coordinates.shape[1] > 1 else 0.0,
                    "pc3": float(coordinates[cursor, 2]) if coordinates.shape[1] > 2 else 0.0,
                }
            )
            cursor += 1
    analysis_dir = output_root / "analysis"
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "pca_coordinates.csv", metadata_rows)
    cumulative = np.cumsum(explained_ratio)
    explained_rows = [
        {
            "component": index + 1,
            "explained_variance_ratio": float(value),
            "cumulative_explained_variance": float(cumulative[index]),
        }
        for index, value in enumerate(explained_ratio)
    ]
    write_csv(analysis_dir / "pca_explained_variance.csv", explained_rows)
    top3 = float(explained_ratio[:3].sum())
    k90 = int(np.searchsorted(cumulative, 0.90, side="left") + 1)

    available_devices = sorted({run.device for run in runs}, key=devices.index)
    available_seeds = sorted({run.seed for run in runs})
    full_matrix = len(available_devices) >= 2 and len(available_seeds) >= 2
    similarity_rows: list[dict[str, object]] = []
    similarity_summary_rows: list[dict[str, object]] = []
    ratio_rows: list[dict[str, object]] = []
    same_epoch_rows: list[dict[str, object]] = []
    endpoint_run_rows: list[dict[str, object]] = []
    endpoint_similarity_rows: list[dict[str, object]] = []
    endpoint_similarity_summary_rows: list[dict[str, object]] = []
    endpoint_effect_rows: list[dict[str, object]] = []
    milestones = np.asarray([], dtype=np.float64)
    analysis_config = dict(config["analysis"])
    repeats = int(analysis_config["bootstrap_repeats"])
    rng = np.random.default_rng(int(analysis_config["bootstrap_seed"]))

    if full_matrix and not missing:
        milestones = choose_milestones(runs, analysis_config)
        run_lookup = {(run.device, run.seed): run for run in runs}
        for milestone in milestones:
            vectors: dict[str, list[np.ndarray]] = {device: [] for device in devices}
            deltas: dict[str, list[np.ndarray]] = {device: [] for device in devices}
            crossing_epochs: list[float] = []
            for device in devices:
                for seed in seeds:
                    run = run_lookup[(device, seed)]
                    vector, crossing_epoch = interpolate_first_crossing(
                        run.epochs, run.accuracies, run.probabilities, float(milestone)
                    )
                    vectors[device].append(vector)
                    deltas[device].append(vector - run.probabilities[0])
                    crossing_epochs.append(crossing_epoch)

            device_delta = {device: np.mean(np.stack(deltas[device]), axis=0) for device in devices}
            pair_similarities: list[float] = []
            for first_index, first in enumerate(devices):
                for second_index, second in enumerate(devices):
                    similarity = cosine_similarity(device_delta[first], device_delta[second])
                    similarity_rows.append(
                        {
                            "milestone_accuracy": float(milestone),
                            "device_i": first,
                            "device_j": second,
                            "cosine_similarity": similarity,
                        }
                    )
                    if second_index > first_index:
                        pair_similarities.append(similarity)

            bootstrap_similarity = np.empty(repeats, dtype=np.float64)
            for repeat in range(repeats):
                sampled_delta = {
                    device: np.mean(
                        np.stack(
                            [deltas[device][index] for index in rng.integers(0, len(seeds), len(seeds))]
                        ),
                        axis=0,
                    )
                    for device in devices
                }
                values = [
                    cosine_similarity(sampled_delta[first], sampled_delta[second])
                    for first, second in itertools.combinations(devices, 2)
                ]
                bootstrap_similarity[repeat] = float(np.nanmean(values))
            similarity_low, similarity_high = np.nanpercentile(bootstrap_similarity, [2.5, 97.5])
            similarity_summary_rows.append(
                {
                    "milestone_accuracy": float(milestone),
                    "mean_cross_device_cosine": float(np.nanmean(pair_similarities)),
                    "ci95_low": float(similarity_low),
                    "ci95_high": float(similarity_high),
                    "device_pair_count": len(pair_similarities),
                    "seed_count_per_device": len(seeds),
                }
            )

            between_accuracy, within_accuracy = effect_components(vectors)
            ratio_accuracy, ratio_accuracy_low, ratio_accuracy_high = ratio_and_bootstrap(
                between_accuracy, within_accuracy, repeats=repeats, rng=rng
            )
            snapshot_interval = int(config["training"]["snapshot_interval"])
            aligned_epoch = int(
                round(float(np.median(crossing_epochs)) / snapshot_interval) * snapshot_interval
            )
            aligned_epoch = int(np.clip(aligned_epoch, 0, int(config["training"]["epochs"])))
            epoch_vectors: dict[str, list[np.ndarray]] = {device: [] for device in devices}
            for device in devices:
                for seed in seeds:
                    run = run_lookup[(device, seed)]
                    index = int(np.argmin(np.abs(run.epochs - aligned_epoch)))
                    epoch_vectors[device].append(run.probabilities[index])
            between_epoch, within_epoch = effect_components(epoch_vectors)
            ratio_epoch, ratio_epoch_low, ratio_epoch_high = ratio_and_bootstrap(
                between_epoch, within_epoch, repeats=repeats, rng=rng
            )
            ratio_rows.append(
                {
                    "milestone_accuracy": float(milestone),
                    "aligned_epoch": aligned_epoch,
                    "r_same_accuracy": ratio_accuracy,
                    "r_same_accuracy_ci95_low": ratio_accuracy_low,
                    "r_same_accuracy_ci95_high": ratio_accuracy_high,
                    "r_same_epoch": ratio_epoch,
                    "r_same_epoch_ci95_low": ratio_epoch_low,
                    "r_same_epoch_ci95_high": ratio_epoch_high,
                    "mean_between_distance_same_accuracy": float(between_accuracy.mean()),
                    "mean_within_distance_same_accuracy": float(within_accuracy.mean()),
                    "mean_between_distance_same_epoch": float(between_epoch.mean()),
                    "mean_within_distance_same_epoch": float(within_epoch.mean()),
                }
            )

        common_epochs = runs[0].epochs
        for epoch_index, epoch in enumerate(common_epochs):
            epoch_vectors = {
                device: [run_lookup[(device, seed)].probabilities[epoch_index] for seed in seeds]
                for device in devices
            }
            between, within = effect_components(epoch_vectors)
            ratio, low, high = ratio_and_bootstrap(between, within, repeats=repeats, rng=rng)
            same_epoch_rows.append(
                {
                    "epoch": int(epoch),
                    "r_same_epoch": ratio,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_between_distance": float(between.mean()),
                    "mean_within_distance": float(within.mean()),
                }
            )

        for endpoint in ("best", "final"):
            endpoint_vectors: dict[str, list[np.ndarray]] = {device: [] for device in devices}
            endpoint_deltas: dict[str, list[np.ndarray]] = {device: [] for device in devices}
            for device in devices:
                for seed in seeds:
                    run = run_lookup[(device, seed)]
                    index = select_endpoint_index(run.accuracies, endpoint)
                    vector = run.probabilities[index]
                    endpoint_vectors[device].append(vector)
                    endpoint_deltas[device].append(vector - run.probabilities[0])
                    endpoint_run_rows.append(
                        {
                            "endpoint": endpoint,
                            "device": device,
                            "seed": seed,
                            "epoch": int(run.epochs[index]),
                            "validation_accuracy": float(run.accuracies[index]),
                            "validation_loss": float(run.losses[index]),
                        }
                    )

            device_delta = {device: np.mean(np.stack(endpoint_deltas[device]), axis=0) for device in devices}
            pair_similarities: list[float] = []
            for first_index, first in enumerate(devices):
                for second_index, second in enumerate(devices):
                    similarity = cosine_similarity(device_delta[first], device_delta[second])
                    endpoint_similarity_rows.append(
                        {
                            "endpoint": endpoint,
                            "device_i": first,
                            "device_j": second,
                            "cosine_similarity": similarity,
                        }
                    )
                    if second_index > first_index:
                        pair_similarities.append(similarity)

            bootstrap_similarity = np.empty(repeats, dtype=np.float64)
            for repeat in range(repeats):
                sampled_delta = {
                    device: np.mean(
                        np.stack(
                            [
                                endpoint_deltas[device][index]
                                for index in rng.integers(0, len(seeds), len(seeds))
                            ]
                        ),
                        axis=0,
                    )
                    for device in devices
                }
                values = [
                    cosine_similarity(sampled_delta[first], sampled_delta[second])
                    for first, second in itertools.combinations(devices, 2)
                ]
                bootstrap_similarity[repeat] = float(np.nanmean(values))
            similarity_low, similarity_high = np.nanpercentile(bootstrap_similarity, [2.5, 97.5])
            endpoint_similarity_summary_rows.append(
                {
                    "endpoint": endpoint,
                    "mean_cross_device_cosine": float(np.nanmean(pair_similarities)),
                    "ci95_low": float(similarity_low),
                    "ci95_high": float(similarity_high),
                    "device_pair_count": len(pair_similarities),
                    "seed_count_per_device": len(seeds),
                }
            )

            between, within = effect_components(endpoint_vectors)
            ratio, low, high = ratio_and_bootstrap(between, within, repeats=repeats, rng=rng)
            endpoint_effect_rows.append(
                {
                    "endpoint": endpoint,
                    "device_seed_effect_ratio": ratio,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_between_distance": float(between.mean()),
                    "mean_within_distance": float(within.mean()),
                }
            )

        write_csv(analysis_dir / "direction_similarity_matrices.csv", similarity_rows)
        write_csv(
            analysis_dir / "direction_similarity_summary.csv",
            similarity_summary_rows,
        )
        write_csv(analysis_dir / "device_seed_effect_ratio.csv", ratio_rows)
        write_csv(analysis_dir / "same_epoch_effect_curve.csv", same_epoch_rows)
        write_csv(analysis_dir / "endpoint_run_metrics.csv", endpoint_run_rows)
        write_csv(
            analysis_dir / "endpoint_direction_similarity_matrices.csv",
            endpoint_similarity_rows,
        )
        write_csv(
            analysis_dir / "endpoint_direction_similarity_summary.csv",
            endpoint_similarity_summary_rows,
        )
        write_csv(analysis_dir / "endpoint_device_seed_effect.csv", endpoint_effect_rows)

    endpoint_summary = {
        row["endpoint"]: {
            "mean_cross_device_cosine": row["mean_cross_device_cosine"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "device_seed_effect_ratio": next(
                effect["device_seed_effect_ratio"]
                for effect in endpoint_effect_rows
                if effect["endpoint"] == row["endpoint"]
            ),
        }
        for row in endpoint_similarity_summary_rows
    }
    summary = {
        "status": "completed" if not missing else "partial",
        "run_count": len(runs),
        "expected_run_count": len(devices) * len(seeds),
        "missing_runs": missing,
        "observation_count": int(matrix.shape[0]),
        "prediction_feature_count": int(matrix.shape[1]),
        "top3_explained_variance": top3,
        "k90": k90,
        "accuracy_milestones": milestones.tolist(),
        "endpoint_summary": endpoint_summary,
        "distance_normalization": "L2 / sqrt(1000 * 10)",
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "seed resampling for cosine; pair-distance resampling for R",
    }
    atomic_json(analysis_dir / "summary.json", summary)
    methods = f"""# Prediction-trajectory methods

Five device-specific VGG8 models were trained on CIFAR-10 with seeds {list(seeds)}.
All runs used the same optimizer recipe and a fixed stratified 1,000-image test probe.
Probabilities were saved at epoch 0 and every {config["training"]["snapshot_interval"]} epochs.

Joint PCA was computed after centering all checkpoint prediction vectors together;
no device-wise standardization was applied. Prediction-direction similarity uses
the cosine of device-mean changes relative to each run's initialization. Device and
seed distances use L2 distance divided by sqrt(1000 x 10). Accuracy-aligned vectors
use linear interpolation at the first milestone crossing. The same-epoch comparison
uses the nearest checkpoint to the median milestone-crossing epoch.

Endpoint analysis is separate from accuracy alignment. For each run, the best
checkpoint is the earliest saved snapshot with maximum validation accuracy; the
final checkpoint is epoch {config["training"]["epochs"]}. Both endpoints report
cross-device direction similarity and the device/seed effect ratio.

Bootstrap intervals use {repeats} deterministic replicates. With only three seeds,
they summarize run-to-run stability and are not population-level confidence bounds.
"""
    (reports_dir / "prediction_trajectory_methods.md").write_text(methods, encoding="utf-8")
    results = f"""# Prediction-trajectory results

- Status: {summary["status"]}
- Completed runs: {len(runs)} / {summary["expected_run_count"]}
- Joint PCA top-3 explained variance: {top3:.6f}
- Components required for 90% variance: {k90}
- Common accuracy milestones: {", ".join(f"{value:.3f}" for value in milestones) if milestones.size else "not available"}
- Best-checkpoint mean cross-device cosine: {endpoint_summary.get("best", {}).get("mean_cross_device_cosine", float("nan")):.6f}
- Best-checkpoint device/seed effect ratio: {endpoint_summary.get("best", {}).get("device_seed_effect_ratio", float("nan")):.6f}
- Final-checkpoint mean cross-device cosine: {endpoint_summary.get("final", {}).get("mean_cross_device_cosine", float("nan")):.6f}
- Final-checkpoint device/seed effect ratio: {endpoint_summary.get("final", {}).get("device_seed_effect_ratio", float("nan")):.6f}

Interpretation is intentionally deferred to the completed numerical tables and does
not assume that device effects are smaller or larger than seed effects.
"""
    (reports_dir / "prediction_trajectory_results.md").write_text(results, encoding="utf-8")
    write_csv(tables_dir / "prediction_trajectory_summary.csv", [summary])
    print("TRAJECTORY_ANALYSIS=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
