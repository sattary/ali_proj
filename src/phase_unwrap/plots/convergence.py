"""
Convergence diagnostics: learning rate schedule and gradient norms.

Reads metrics.csv for LR history. Gradient norm tracking requires
the data to have been logged during training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import DOUBLE_COL, LINE_COLORS, nature_style, save_figure


def plot_convergence(
    run_dir: str,
    out_path: str | None = None,
) -> None:
    """
    Plot LR schedule and per-epoch train loss components.
    """
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")

    data: dict[str, list[float]] = {}
    with open(metrics_path, "r") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(float("nan"))

    if out_path is None:
        out_path = str(Path(run_dir) / "convergence.png")

    epochs = np.array(data.get("epoch", []))

    with nature_style():
        fig, axes = plt.subplots(
            2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 0.5), sharex=True
        )

        # panel 1: loss components
        ax1 = axes[0]
        if "train_mae" in data:
            ax1.plot(
                epochs,
                data["train_mae"],
                color=LINE_COLORS[0],
                label="MAE term",
                linewidth=0.8,
            )
        if "train_grad" in data:
            ax1.plot(
                epochs,
                data["train_grad"],
                color=LINE_COLORS[1],
                label="Gradient term",
                linewidth=0.8,
            )
        if "train_loss" in data:
            ax1.plot(
                epochs,
                data["train_loss"],
                color=LINE_COLORS[2],
                label="Total loss",
                linewidth=1.0,
            )
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Components")
        ax1.legend()

        # panel 2: learning rate
        ax2 = axes[1]
        if "lr" in data:
            ax2.plot(epochs, data["lr"], color=LINE_COLORS[3], linewidth=0.8)
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.tight_layout(pad=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved convergence plot: {out_path}")