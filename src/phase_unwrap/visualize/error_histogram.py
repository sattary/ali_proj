"""
Error histogram and CDF: per-sample MAE distribution over a dataset.

Annotates mean, median, and 95th percentile.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from ..config import load_train_config
from ..data import build_dataloaders
from ..model import build_model
from ..ops import affine_align
from ..utils import pick_device
from .style import DOUBLE_COL, LINE_COLORS, nature_style, save_figure


@torch.no_grad()
def plot_error_histogram(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/error_histogram.png",
    config_path: str | None = None,
) -> None:
    """
    Histogram + CDF of per-sample MAE across the full dataset.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
    model.eval()

    _, val_loader = build_dataloaders(
        cfg.data, cfg.optim, device, seed=cfg.logging.seed
    )
    loader = (
        val_loader
        or build_dataloaders(cfg.data, cfg.optim, device, seed=cfg.logging.seed)[0]
    )

    sample_maes: list[float] = []

    for I_input, phi_gt, _ in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            phi_abs = phi_raw + k_off

        aligned, _, _ = affine_align(phi_abs, phi_gt)

        # per-sample MAE
        for i in range(I_input.size(0)):
            mae = (aligned[i] - phi_gt[i]).abs().mean().item()
            sample_maes.append(mae)

    maes = np.array(sample_maes)
    mean_mae = np.mean(maes)
    median_mae = np.median(maes)
    p95_mae = np.percentile(maes, 95)

    with nature_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))

        # histogram
        n_bins = min(50, max(10, len(maes) // 5))
        ax1.hist(
            maes,
            bins=n_bins,
            color=LINE_COLORS[0],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
        )
        ax1.axvline(
            mean_mae,
            color=LINE_COLORS[1],
            linestyle="--",
            linewidth=0.8,
            label=f"Mean = {mean_mae:.3f}",
        )
        ax1.axvline(
            median_mae,
            color=LINE_COLORS[2],
            linestyle="-.",
            linewidth=0.8,
            label=f"Median = {median_mae:.3f}",
        )
        ax1.axvline(
            p95_mae,
            color=LINE_COLORS[3],
            linestyle=":",
            linewidth=0.8,
            label=f"95th pctl = {p95_mae:.3f}",
        )
        ax1.set_xlabel("Per-sample MAE [rad]")
        ax1.set_ylabel("Count")
        ax1.set_title("Error Distribution")
        ax1.legend(fontsize=5)

        # CDF
        sorted_maes = np.sort(maes)
        cdf = np.arange(1, len(sorted_maes) + 1) / len(sorted_maes)
        ax2.plot(sorted_maes, cdf, color=LINE_COLORS[0], linewidth=0.8)
        ax2.axhline(0.5, color="gray", linewidth=0.3, linestyle="--")
        ax2.axhline(0.95, color="gray", linewidth=0.3, linestyle="--")
        ax2.set_xlabel("Per-sample MAE [rad]")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("CDF")

        fig.tight_layout(pad=0.8)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved error histogram: {out_path}")
        print(
            f"  Mean MAE: {mean_mae:.4f} | Median: {median_mae:.4f} | 95th: {p95_mae:.4f}"
        )
