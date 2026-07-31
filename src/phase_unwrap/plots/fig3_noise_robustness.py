"""
Figure 3: Noise Robustness Analysis.

Contains two plots for the paper:
1. plot_f3_noise_sweep: Statistical degradation curve across SNR levels.
2. plot_f3_noise_grid: Qualitative visual grid comparing clean vs noisy predictions.

Stateless implementations.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import (
    DOUBLE_COL,
    SINGLE_COL,
    create_nature_palette,
    label_panels,
    publication_plot,
)
from .utils import draw_error_panel, draw_phase_panel


def _default_level_fmt(v: float) -> str:
    return "Clean" if math.isinf(v) else f"{v:.0f} dB"


@publication_plot
def plot_f3_noise_sweep(
    results: dict[str, list[float]],
    all_maes_by_snr: dict[float, list[float]] | None = None,
    filepath: str | None = None,
    x_label: str = "SNR [dB]",
    level_fmt=None,
) -> plt.Figure:
    """
    Statistical noise degradation curve.

    When ``all_maes_by_snr`` is supplied, draws a violin+point panel (per-sample
    spread) alongside the mean curve. Without it (e.g. loading saved severity
    means from result.json), draws only the mean degradation curve — no violin
    can be reconstructed from aggregated means alone.
    """
    if level_fmt is None:
        level_fmt = _default_level_fmt
    palette = create_nature_palette(6)
    has_violin = bool(all_maes_by_snr)

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    if has_violin:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])

        plot_data_snr, plot_data_mae = [], []
        for snr_db, maes in all_maes_by_snr.items():
            label = level_fmt(snr_db)
            for mae in maes:
                plot_data_snr.append(label)
                plot_data_mae.append(mae)
        df = pd.DataFrame({"SNR": plot_data_snr, "MAE": plot_data_mae})
        sns.violinplot(
            data=df, x="SNR", y="MAE", hue="SNR", ax=ax1, palette="Blues",
            inner="box", linewidth=1, legend=False,
        )
        sns.pointplot(
            data=df, x="SNR", y="MAE", ax=ax1, color="red", markers="D",
            linestyles="", errorbar=None,
        )
        ax1.set_xlabel(x_label, fontsize=9)
        ax1.set_ylabel("MAE [rad]", fontsize=9)
        ax1.set_title("Error Distribution", fontsize=10)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")
        ax2 = fig.add_subplot(gs[1])
        panels = [ax1, ax2]
    else:
        ax2 = fig.add_subplot(1, 1, 1)
        panels = [ax2]

    noisy_mask = [not math.isinf(s) for s in results["snr_db"]]
    snr_noisy = [s for s, m in zip(results["snr_db"], noisy_mask) if m]
    mae_noisy = [m for m, mask in zip(results["mae"], noisy_mask) if mask]
    std_noisy = [s for s, mask in zip(results["std"], noisy_mask) if mask]

    sns.lineplot(
        x=snr_noisy, y=mae_noisy, ax=ax2, color=palette[0],
        marker="o", linewidth=2, markersize=8,
    )
    ax2.fill_between(
        snr_noisy,
        np.array(mae_noisy) - np.array(std_noisy),
        np.array(mae_noisy) + np.array(std_noisy),
        alpha=0.2, color=palette[0],
    )

    if any(not m for m in noisy_mask):
        clean_idx = [i for i, m in enumerate(noisy_mask) if not m][0]
        clean_mae = results["mae"][clean_idx]
        ax2.axhline(
            clean_mae, color=palette[2], linestyle="--", linewidth=2,
            label=f"Clean ({clean_mae:.3f})",
        )

    ax2.set_xlabel(x_label, fontsize=9)
    ax2.set_ylabel("Mean MAE [rad]", fontsize=9)
    ax2.set_title("Degradation Curve", fontsize=10)
    if any(not m for m in noisy_mask):
        ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Noise Robustness Analysis", fontsize=11, y=1.02)
    label_panels(panels)
    return fig


@publication_plot
def plot_f3_noise_grid(
    gt_np: np.ndarray,
    pred_clean_np: np.ndarray,
    pred_noisy_np: np.ndarray,
    noise_level: float = 1.0,
    filepath: str | None = None,
) -> plt.Figure:
    """
    Qualitative noise comparison grid showing clean vs noisy inference errors.
    
    Args:
        gt_np: (N, 1, H, W) ground truth unwrapped phase.
        pred_clean_np: (N, 1, H, W) prediction from clean input.
        pred_noisy_np: (N, 1, H, W) prediction from noisy input.
    """
    n_samples = gt_np.shape[0]

    err_clean = np.abs(pred_clean_np - gt_np)
    err_noisy = np.abs(pred_noisy_np - gt_np)
    err_diff = err_noisy - err_clean

    mae_clean = err_clean.mean()
    mae_noisy = err_noisy.mean()
    mae_increase = mae_noisy - mae_clean
    mae_increase_pct = (mae_increase / mae_clean) * 100 if mae_clean > 0 else 0

    fig, axes = plt.subplots(
        3, n_samples, figsize=(SINGLE_COL * 1.5, SINGLE_COL * 1.2), squeeze=False
    )

    for i in range(n_samples):
        # Row 1: Clean Path Error
        draw_error_panel(
            axes[0, i],
            err_clean[i, 0],
            f"Sample {i}" if i == 0 else "",
            stats={"mae": err_clean[i].mean()},
        )

        # Row 2: Noisy Path Error
        draw_error_panel(
            axes[1, i], err_noisy[i, 0], "", stats={"mae": err_noisy[i].mean()}
        )

        # Row 3: Difference
        vmax_diff = max(abs(err_diff[i].min()), abs(err_diff[i].max()))
        draw_phase_panel(
            axes[2, i],
            err_diff[i, 0],
            "",
            cmap="PuOr",
            vmin=-vmax_diff,
            vmax=vmax_diff,
        )

    # Labels
    axes[0, 0].set_ylabel("Clean\nError", fontsize=8, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("Noisy\nError", fontsize=8, rotation=0, ha="right", va="center")
    axes[2, 0].set_ylabel("Error\nShift", fontsize=8, rotation=0, ha="right", va="center")

    stats_text = (
        f"Clean MAE: {mae_clean:.4f}\n"
        f"Noisy MAE: {mae_noisy:.4f}\n"
        f"MAE Increase: +{mae_increase:.4f} ({mae_increase_pct:+.1f}%)"
    )

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.suptitle(
        f"Noise Robustness Analysis (Noise: {noise_level:.1f})", fontsize=10, y=0.98
    )
    
    # We only label the first column to avoid clutter
    label_panels(axes[:, 0])
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    return fig
