"""Publication figures for the cross-device prediction-trajectory experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEVICE_ORDER = ("reram", "pcm", "stt", "fefet", "flash")
DEVICE_LABELS = {
    "reram": "ReRAM",
    "pcm": "PCM",
    "stt": "STT-MRAM",
    "fefet": "FeFET",
    "flash": "Flash",
}
DEVICE_COLORS = {
    "reram": "#4C78A8",
    "pcm": "#E6A157",
    "stt": "#5B9A78",
    "fefet": "#8064A2",
    "flash": "#C85F86",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_publication_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def plot_joint_pca(root: Path, summary: dict) -> mpl.figure.Figure:
    rows = read_csv(root / "analysis" / "pca_coordinates.csv")
    explained_rows = read_csv(root / "analysis" / "pca_explained_variance.csv")
    endpoint_rows = read_csv(root / "analysis" / "endpoint_run_metrics.csv")
    best_epochs = {
        (row["device"], int(row["seed"])): int(row["epoch"])
        for row in endpoint_rows
        if row["endpoint"] == "best"
    }
    fig, ax = plt.subplots(figsize=(3.503937, 2.992126))  # 89 x 76 mm

    for device in DEVICE_ORDER:
        device_rows = [row for row in rows if row["device"] == device]
        if not device_rows:
            continue
        seeds = sorted({int(row["seed"]) for row in device_rows})
        epoch_values = sorted({int(row["epoch"]) for row in device_rows})
        trajectories: list[np.ndarray] = []
        best_coordinates: list[np.ndarray] = []
        for seed in seeds:
            seed_rows = sorted(
                [row for row in device_rows if int(row["seed"]) == seed],
                key=lambda row: int(row["epoch"]),
            )
            coordinates = np.asarray([[float(row["pc1"]), float(row["pc2"])] for row in seed_rows])
            trajectories.append(coordinates)
            best_epoch = best_epochs[(device, seed)]
            best_index = next(index for index, row in enumerate(seed_rows) if int(row["epoch"]) == best_epoch)
            best_coordinates.append(coordinates[best_index])
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=DEVICE_COLORS[device],
                linewidth=0.65,
                alpha=0.42,
            )
        stacked = np.stack(trajectories)
        mean_trajectory = stacked.mean(axis=0)
        ax.plot(
            mean_trajectory[:, 0],
            mean_trajectory[:, 1],
            color=DEVICE_COLORS[device],
            linewidth=1.8,
            label=DEVICE_LABELS[device],
            zorder=4,
        )
        ax.scatter(
            mean_trajectory[0, 0],
            mean_trajectory[0, 1],
            s=18,
            facecolor="white",
            edgecolor=DEVICE_COLORS[device],
            linewidth=0.9,
            zorder=6,
        )
        mean_best = np.mean(np.stack(best_coordinates), axis=0)
        ax.scatter(
            mean_best[0],
            mean_best[1],
            s=34,
            marker="*",
            facecolor=DEVICE_COLORS[device],
            edgecolor="white",
            linewidth=0.45,
            zorder=7,
        )
        ax.scatter(
            mean_trajectory[-1, 0],
            mean_trajectory[-1, 1],
            s=17,
            marker="D",
            facecolor=DEVICE_COLORS[device],
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
        if len(epoch_values) != mean_trajectory.shape[0]:
            raise ValueError(f"inconsistent PCA trajectory length for {device}")

    explained = np.asarray([float(row["explained_variance_ratio"]) for row in explained_rows])
    pc1 = 100.0 * explained[0]
    pc2 = 100.0 * explained[1]
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        handlelength=1.8,
        columnspacing=0.9,
    )
    ax.text(-0.15, 1.08, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    inset = ax.inset_axes([0.61, 0.57, 0.35, 0.35])
    cumulative = np.cumsum(explained)
    k90 = int(summary["k90"])
    limit = min(len(cumulative), max(12, k90 + 3))
    components = np.arange(1, limit + 1)
    inset.plot(components, cumulative[:limit], color="#4C78A8", linewidth=1.1)
    inset.axhline(0.90, color="#888888", linewidth=0.6, linestyle="--")
    inset.axvline(k90, color="#C85F86", linewidth=0.7, linestyle=":")
    inset.scatter(
        [3],
        [float(summary["top3_explained_variance"])],
        color="#8064A2",
        s=9,
        zorder=3,
    )
    inset.set_xlabel("PC", labelpad=1)
    inset.set_ylabel("Cumulative", labelpad=1)
    inset.set_ylim(0, 1.02)
    inset.tick_params(labelsize=5, length=2, pad=1)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    return fig


def plot_quantification(root: Path) -> mpl.figure.Figure:
    similarity_rows = read_csv(root / "analysis" / "direction_similarity_summary.csv")
    ratio_rows = read_csv(root / "analysis" / "device_seed_effect_ratio.csv")
    endpoint_similarity_rows = read_csv(root / "analysis" / "endpoint_direction_similarity_summary.csv")
    endpoint_effect_rows = read_csv(root / "analysis" / "endpoint_device_seed_effect.csv")
    if not similarity_rows or not ratio_rows:
        raise ValueError("completed cross-device analysis tables are required")
    x_similarity = 100.0 * np.asarray([float(row["milestone_accuracy"]) for row in similarity_rows])
    similarity = np.asarray([float(row["mean_cross_device_cosine"]) for row in similarity_rows])
    similarity_low = np.asarray([float(row["ci95_low"]) for row in similarity_rows])
    similarity_high = np.asarray([float(row["ci95_high"]) for row in similarity_rows])

    x_ratio = 100.0 * np.asarray([float(row["milestone_accuracy"]) for row in ratio_rows])
    ratio_accuracy = np.asarray([float(row["r_same_accuracy"]) for row in ratio_rows])
    ratio_accuracy_low = np.asarray([float(row["r_same_accuracy_ci95_low"]) for row in ratio_rows])
    ratio_accuracy_high = np.asarray([float(row["r_same_accuracy_ci95_high"]) for row in ratio_rows])
    ratio_epoch = np.asarray([float(row["r_same_epoch"]) for row in ratio_rows])
    ratio_epoch_low = np.asarray([float(row["r_same_epoch_ci95_low"]) for row in ratio_rows])
    ratio_epoch_high = np.asarray([float(row["r_same_epoch_ci95_high"]) for row in ratio_rows])

    endpoint_order = ("best", "final")
    endpoint_similarity = {row["endpoint"]: row for row in endpoint_similarity_rows}
    endpoint_effect = {row["endpoint"]: row for row in endpoint_effect_rows}
    x_max = float(max(np.max(x_similarity), np.max(x_ratio)))
    x_min = float(min(np.min(x_similarity), np.min(x_ratio)))
    endpoint_gap = max(7.5, 0.16 * (x_max - x_min))
    endpoint_x = np.asarray([x_max + endpoint_gap, x_max + 2.0 * endpoint_gap])
    endpoint_labels = ["Best", "Final"]

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 2.755906))  # 183 x 70 mm
    left, right = axes
    left.fill_between(
        x_similarity,
        similarity_low,
        similarity_high,
        color="#8FAED0",
        alpha=0.30,
        linewidth=0,
    )
    left.plot(
        x_similarity,
        similarity,
        color="#4C78A8",
        marker="o",
        markersize=3.2,
        linewidth=1.4,
    )
    left.axvline(
        x_max + 0.48 * endpoint_gap,
        color="#AAAAAA",
        linewidth=0.6,
        linestyle=":",
    )
    for index, endpoint in enumerate(endpoint_order):
        row = endpoint_similarity[endpoint]
        value = float(row["mean_cross_device_cosine"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        color = "#C85F86" if endpoint == "best" else "#8064A2"
        left.vlines(endpoint_x[index], low, high, color=color, linewidth=0.9)
        left.hlines(
            [low, high],
            endpoint_x[index] - 0.75,
            endpoint_x[index] + 0.75,
            color=color,
            linewidth=0.9,
        )
        left.scatter(
            endpoint_x[index],
            value,
            color=color,
            marker="*" if endpoint == "best" else "D",
            s=24 if endpoint == "best" else 12,
            zorder=5,
        )
    left.set_xticks(
        np.concatenate([x_similarity, endpoint_x]),
        [*[f"{value:.0f}" for value in x_similarity], *endpoint_labels],
    )
    left.set_xlabel("Accuracy milestone (%) / endpoint")
    left.set_ylabel("Cross-device cosine similarity")
    left.set_ylim(min(-0.05, float(np.nanmin(similarity_low)) - 0.03), 1.02)
    style_axis(left)

    right.axhline(1.0, color="#888888", linewidth=0.75, linestyle="--", zorder=1)
    right.fill_between(
        x_ratio,
        ratio_epoch_low,
        ratio_epoch_high,
        color="#A89AC2",
        alpha=0.22,
        linewidth=0,
    )
    right.fill_between(
        x_ratio,
        ratio_accuracy_low,
        ratio_accuracy_high,
        color="#DCA6B7",
        alpha=0.26,
        linewidth=0,
    )
    right.plot(
        x_ratio,
        ratio_epoch,
        color="#8064A2",
        marker="s",
        markersize=3.0,
        linewidth=1.3,
        label="Same epoch",
    )
    right.plot(
        x_ratio,
        ratio_accuracy,
        color="#C85F86",
        marker="o",
        markersize=3.2,
        linewidth=1.4,
        label="Same accuracy",
    )
    right.axvline(
        x_max + 0.48 * endpoint_gap,
        color="#AAAAAA",
        linewidth=0.6,
        linestyle=":",
    )
    for index, endpoint in enumerate(endpoint_order):
        row = endpoint_effect[endpoint]
        value = float(row["device_seed_effect_ratio"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        color = "#C85F86" if endpoint == "best" else "#8064A2"
        right.vlines(endpoint_x[index], low, high, color=color, linewidth=0.9)
        right.hlines(
            [low, high],
            endpoint_x[index] - 0.75,
            endpoint_x[index] + 0.75,
            color=color,
            linewidth=0.9,
        )
        right.scatter(
            endpoint_x[index],
            value,
            color=color,
            marker="*" if endpoint == "best" else "D",
            s=24 if endpoint == "best" else 12,
            label="Best endpoint" if endpoint == "best" else "Final endpoint",
            zorder=5,
        )
    right.set_xticks(
        np.concatenate([x_ratio, endpoint_x]),
        [*[f"{value:.0f}" for value in x_ratio], *endpoint_labels],
    )
    right.set_xlabel("Accuracy milestone (%) / endpoint")
    right.set_ylabel(r"Device/seed effect ratio, $R$")
    right.set_ylim(bottom=0)
    right.legend(loc="best", ncol=2, fontsize=5.5, columnspacing=0.8)
    style_axis(right)
    left.text(-0.16, 1.05, "c", transform=left.transAxes, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.95, wspace=0.34)
    return fig


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    summary = json.loads((root / "analysis" / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError("publication figures require the complete 15-run matrix")
    figures = root / "figures"
    figure_b = plot_joint_pca(root, summary)
    save_publication_figure(figure_b, figures / "figure3b_joint_pca")
    plt.close(figure_b)
    figure_c = plot_quantification(root)
    save_publication_figure(figure_c, figures / "figure3c_prediction_geometry")
    plt.close(figure_c)
    print(f"TRAJECTORY_FIGURES={figures}", flush=True)


if __name__ == "__main__":
    main()
