"""
Training curve plot: dual-axis loss + validation MAE vs epoch.

Supports single-run (from metrics.csv) and multi-seed (from aggregate.csv
with mean +/- std shaded bands).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    DOUBLE_COL,
    LINE_COLORS,
    nature_style,
    save_figure,
)


def _read_csv(path: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(float("nan"))
    return data


def plot_training_curve(
    run_dir: str,
    out_path: str | None = None,
    show_lr: bool = True,
) -> None:
    """
    Plot training loss and validation MAE vs epoch.

    Auto-detects single-run (metrics.csv) vs multi-seed (aggregate.csv).

    Args:
        run_dir:  Path to run directory or multi-seed parent directory.
        out_path: Output file path. Defaults to ``<run_dir>/training_curve.png``.
        show_lr:  Whether to include a secondary subplot for learning rate.
    """
    run_path = Path(run_dir)
    agg_path = run_path / "aggregate.csv"
    metrics_path = run_path / "metrics.csv"

    is_multiseed = agg_path.exists()
    csv_path = str(agg_path) if is_multiseed else str(metrics_path)

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"No metrics found in {run_dir}")

    data = _read_csv(csv_path)

    if out_path is None:
        out_path = str(run_path / "training_curve.png")

    with nature_style():
        n_rows = 2 if show_lr else 1
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.45 * n_rows),
            gridspec_kw={"height_ratios": [3, 1]} if show_lr else None,
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        ax1 = axes[0]

        if is_multiseed:
            epochs = np.array(data.get("epoch", []))
            # training loss
            tl_mean = np.array(data.get("train_loss_mean", []))
            tl_std = np.array(data.get("train_loss_std", []))
            l1 = ax1.plot(epochs, tl_mean, color=LINE_COLORS[0], label="Train loss")[0]
            ax1.fill_between(
                epochs,
                tl_mean - tl_std,
                tl_mean + tl_std,
                color=LINE_COLORS[0],
                alpha=0.15,
            )
            # val MAE on right axis
            ax2 = ax1.twinx()
            vm_mean = np.array(data.get("val_mae_mean", []))
            vm_std = np.array(data.get("val_mae_std", []))
            l2 = ax2.plot(epochs, vm_mean, color=LINE_COLORS[1], label="Val MAE")[0]
            ax2.fill_between(
                epochs,
                vm_mean - vm_std,
                vm_mean + vm_std,
                color=LINE_COLORS[1],
                alpha=0.15,
            )
        else:
            epochs = np.array(data.get("epoch", []))
            train_loss = np.array(data.get("train_loss", []))
            val_mae = np.array(data.get("val_mae", []))

            l1 = ax1.plot(epochs, train_loss, color=LINE_COLORS[0], label="Train loss")[
                0
            ]
            ax2 = ax1.twinx()
            l2 = ax2.plot(epochs, val_mae, color=LINE_COLORS[1], label="Val MAE")[0]

        ax1.set_ylabel("Train loss")
        ax2.set_ylabel("Val MAE [rad]")
        ax2.spines["right"].set_visible(True)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        # LR subplot
        if show_lr and n_rows > 1:
            ax_lr = axes[1]
            if is_multiseed:
                lr_mean = np.array(data.get("lr_mean", []))
                ax_lr.plot(epochs, lr_mean, color=LINE_COLORS[2], linewidth=0.8)
            else:
                lr = np.array(data.get("lr", []))
                ax_lr.plot(epochs, lr, color=LINE_COLORS[2], linewidth=0.8)
            ax_lr.set_ylabel("Learning rate")
            ax_lr.set_xlabel("Epoch")
            ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        else:
            ax1.set_xlabel("Epoch")

        fig.align_ylabels()
        save_figure(fig, out_path)
        print(f"Saved training curve: {out_path}")
