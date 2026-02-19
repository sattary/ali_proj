"""
Training curve plot with seaborn-enhanced statistical visualization.

Features:
- Confidence intervals from multi-seed runs
- Seaborn line plots with error bands
- Statistical annotations for convergence
- Publication-ready aesthetics
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from .style import (
    DOUBLE_COL,
    NATURE_PALETTE,
    create_nature_palette,
    nature_style,
    save_figure,
)


def _read_csv(path: str) -> dict[str, list[float]]:
    """Read metrics CSV into dictionary."""
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


def _compute_convergence_epoch(
    epochs: np.ndarray,
    values: np.ndarray,
    window: int = 5,
    threshold: float = 0.01,
) -> int | None:
    """Find epoch where metric stabilizes within threshold."""
    if len(values) < window * 2:
        return None

    for i in range(window, len(values) - window):
        current_window = values[i : i + window]
        if np.std(current_window) / np.mean(np.abs(current_window)) < threshold:
            return int(epochs[i])
    return None


def plot_training_curve(
    run_dir: str,
    out_path: str | None = None,
    show_lr: bool = True,
    confidence: float = 0.95,
) -> None:
    """
    Plot training loss and validation MAE vs epoch with seaborn enhancements.

    Features:
    - Seaborn line plots with confidence intervals
    - Convergence epoch annotation
    - Multi-metric visualization

    Args:
        run_dir: Path to run directory or multi-seed parent directory
        out_path: Output file path. Defaults to ``<run_dir>/training_curve``
        show_lr: Whether to include learning rate subplot
        confidence: Confidence level for intervals (default 0.95)
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
        out_path = str(run_path / "training_curve")

    with nature_style():
        palette = create_nature_palette(6)
        n_rows = 2 if show_lr else 1

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.5 * n_rows),
            gridspec_kw={"height_ratios": [3, 1]} if show_lr else None,
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        ax1 = axes[0]

        epochs = np.array(data.get("epoch", []))

        if is_multiseed:
            # Multi-seed: use mean ± std as confidence bands
            tl_mean = np.array(data.get("train_loss_mean", []))
            tl_std = np.array(data.get("train_loss_std", []))
            vm_mean = np.array(data.get("val_mae_mean", []))
            vm_std = np.array(data.get("val_mae_std", []))

            # Training loss with confidence band
            sns.lineplot(
                x=epochs,
                y=tl_mean,
                ax=ax1,
                color=palette[0],
                label="Train loss",
                linewidth=1.5,
            )
            ax1.fill_between(
                epochs,
                tl_mean - tl_std,
                tl_mean + tl_std,
                color=palette[0],
                alpha=0.2,
            )

            # Validation MAE on secondary axis
            ax2 = ax1.twinx()
            sns.lineplot(
                x=epochs,
                y=vm_mean,
                ax=ax2,
                color=palette[1],
                label="Val MAE",
                linewidth=1.5,
            )
            ax2.fill_between(
                epochs,
                vm_mean - vm_std,
                vm_mean + vm_std,
                color=palette[1],
                alpha=0.2,
            )

            # Find convergence
            conv_epoch = _compute_convergence_epoch(epochs, vm_mean)
            if conv_epoch:
                ax2.axvline(
                    conv_epoch,
                    color=palette[4],
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Convergence (epoch {conv_epoch})",
                )
        else:
            # Single run
            train_loss = np.array(data.get("train_loss", []))
            val_mae = np.array(data.get("val_mae", []))

            sns.lineplot(
                x=epochs,
                y=train_loss,
                ax=ax1,
                color=palette[0],
                label="Train loss",
                linewidth=1.5,
            )

            ax2 = ax1.twinx()
            sns.lineplot(
                x=epochs,
                y=val_mae,
                ax=ax2,
                color=palette[1],
                label="Val MAE",
                linewidth=1.5,
            )

        ax1.set_ylabel("Train Loss", fontsize=9)
        ax2.set_ylabel("Val MAE [rad]", fontsize=9)
        ax2.spines["right"].set_visible(True)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            frameon=True,
            fontsize=7,
        )
        ax2.legend().set_visible(False)

        # Learning rate subplot
        if show_lr and n_rows > 1:
            ax_lr = axes[1]
            if is_multiseed:
                lr_mean = np.array(data.get("lr_mean", []))
                sns.lineplot(
                    x=epochs, y=lr_mean, ax=ax_lr, color=palette[2], linewidth=1.2
                )
            else:
                lr = np.array(data.get("lr", []))
                sns.lineplot(x=epochs, y=lr, ax=ax_lr, color=palette[2], linewidth=1.2)

            ax_lr.set_ylabel("Learning Rate", fontsize=9)
            ax_lr.set_xlabel("Epoch", fontsize=9)
            ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax_lr.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel("Epoch", fontsize=9)

        # Title with stats
        final_mae = vm_mean[-1] if is_multiseed else val_mae[-1]
        ax1.set_title(
            f"Training Convergence (Final Val MAE: {final_mae:.4f} rad)",
            fontsize=10,
            pad=10,
        )

        fig.align_ylabels()
        save_figure(fig, out_path)
        print(f"Saved training curve: {out_path}")
