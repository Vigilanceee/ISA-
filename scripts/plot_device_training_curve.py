"""Plot every recorded epoch of a selected device training run."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename without a suffix")
    args = parser.parse_args()
    with args.source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")
    values = {name: [float(row[name]) for row in rows] for name in columns}
    if not rows or any(not math.isfinite(value) for group in values.values() for value in group):
        raise ValueError("The complete source curve must contain finite observations")
    if len(set(values["epoch"])) != len(rows):
        raise ValueError("Duplicate epoch records")
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6,
        "axes.grid": False, "legend.frameon": False, "legend.fontsize": 6,
    })
    # Figure contract: one selected run, all epochs, no smoothing or seed-level intervals.
    fig, loss_ax = plt.subplots(figsize=(3.503937, 2.992126), facecolor="white")
    fig.subplots_adjust(left=0.16, right=0.83, bottom=0.17, top=0.80)
    acc_ax = loss_ax.twinx()
    colors = {"train_loss": "#9BBEFF", "val_loss": "#507CC8",
              "train_acc": "#E8A0BE", "val_acc": "#AD3D69"}
    handles = []
    for name, label, axis, style in (
        ("train_loss", "Training loss", loss_ax, "-"),
        ("val_loss", "Validation loss", loss_ax, "--"),
        ("train_acc", "Training accuracy", acc_ax, "-"),
        ("val_acc", "Validation accuracy", acc_ax, "--"),
    ):
        line, = axis.plot(values["epoch"], values[name], color=colors[name],
                          linestyle=style, linewidth=1.0, label=label)
        handles.append(line)
    best = max(range(len(rows)), key=lambda index: values["val_acc"][index])
    acc_ax.plot(values["epoch"][best], values["val_acc"][best], "*", color=colors["val_acc"],
                markersize=6, markeredgecolor="white", markeredgewidth=0.35)
    acc_ax.text(0.97, 0.73, f"Best: {values['val_acc'][best]:.2%}", transform=acc_ax.transAxes,
                ha="right", va="center", color=colors["val_acc"], fontsize=7)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss", color=colors["val_loss"])
    acc_ax.set_ylabel("Accuracy", color=colors["val_acc"])
    loss_ax.tick_params(axis="y", colors=colors["val_loss"], width=0.6, length=2.5)
    acc_ax.tick_params(axis="y", colors=colors["val_acc"], width=0.6, length=2.5)
    loss_ax.tick_params(axis="x", width=0.6, length=2.5)
    loss_ax.set_xlim(0, max(values["epoch"]))
    loss_ax.set_ylim(0, 1.06 * max(values["train_loss"] + values["val_loss"]))
    acc_ax.set_ylim(0, 1.05)
    acc_ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    acc_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for axis in (loss_ax, acc_ax):
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
    fig.legend(handles, [line.get_label() for line in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=2, handlelength=2.0, columnspacing=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output) + ".svg", facecolor="white")
    svg = args.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(str(args.output) + ".pdf", facecolor="white")
    fig.savefig(str(args.output) + ".png", dpi=600, facecolor="white")
    fig.savefig(str(args.output) + ".tiff", dpi=600, facecolor="white")
    plt.close(fig)
    metadata = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "epochs": len(rows), "best_epoch": values["epoch"][best],
                "best_validation_accuracy": values["val_acc"][best], "smoothing": False,
                "omitted_rows": 0, "replicates": "one selected training run",
                "split": "official test split used for validation and checkpoint selection",
                "uncertainty": "No between-run uncertainty is estimated from this single curve"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
