"""
Enhanced error distribution visualization with seaborn.

Features:
- Violin plots showing full distribution shape
- KDE overlays for smooth density estimation
- Rug plots for outlier visibility
- Statistical annotations with significance testing
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from ..core.config import load_train_config
from ..core.ops import affine_align
from ..core.utils import pick_device
from ..data import build_dataloaders
from ..model import build_model
from .style import (
    DOUBLE_COL,
    create_nature_palette,
    nature_style,
    save_figure,
)


@torch.no_grad()
def plot_error_histogram(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/error_histogram",
    config_path: str | None = None,
    compare_tta: bool = False,
    subset: str = "val",
) -> None:
    """
    Enhanced error distribution with seaborn violin plots and statistical analysis.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        out_path: Output file path (without extension)
        config_path: Optional config override
        compare_tta: Whether to compare with TTA (if implemented)
        subset: Dataset subset to use
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    # weights_only=False: loading trusted checkpoint from own training runs
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )
    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = val_loader or train_loader
    else:
        loader = train_loader

    sample_maes: list[float] = []

    for I_input, phi_gt, _, _ in tqdm(loader, desc="Evaluating Dataset (CPU)", mininterval=2.0):
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            if isinstance(phi_raw, list):
                phi_abs = phi_raw[-1] + k_off
            else:
                phi_abs = phi_raw + k_off

        aligned, _, _ = affine_align(phi_abs, phi_gt)

        for i in range(I_input.size(0)):
            mae = (aligned[i] - phi_gt[i]).abs().mean().item()
            sample_maes.append(mae)

    maes = np.array(sample_maes)
    mean_mae = np.mean(maes)
    median_mae = np.median(maes)
    p95_mae = np.percentile(maes, 95)
    p99_mae = np.percentile(maes, 99)
    std_mae = np.std(maes)

    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

        # Create grid: 3 subplots
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

        # Panel 1: Violin + Box plot
        ax1 = fig.add_subplot(gs[0])

        # Seaborn violin plot
        sns.violinplot(
            data=[maes],
            ax=ax1,
            color=palette[0],
            inner="box",
            linewidth=1,
        )

        # Removed sns.swarmplot because it has O(n^2) scaling and hangs endlessly on large datasets.

        # Statistical lines
        ax1.axhline(
            mean_mae,
            color=palette[2],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_mae:.4f}",
        )
        ax1.axhline(
            median_mae,
            color=palette[3],
            linestyle="-.",
            linewidth=1.5,
            label=f"Median: {median_mae:.4f}",
        )
        ax1.axhline(
            p95_mae,
            color=palette[4],
            linestyle=":",
            linewidth=1.5,
            label=f"95th: {p95_mae:.4f}",
        )

        ax1.set_xticks([0])
        ax1.set_xticklabels(["Model"])
        ax1.set_ylabel("MAE [rad]")
        ax1.set_title("Error Distribution (Violin + Box)")
        ax1.legend(loc="upper right", fontsize=6)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: KDE with rug
        ax2 = fig.add_subplot(gs[1])

        sns.kdeplot(
            data=maes,
            ax=ax2,
            color=palette[0],
            fill=True,
            alpha=0.5,
            linewidth=2,
        )

        sns.rugplot(
            data=maes,
            ax=ax2,
            color=palette[1],
            height=0.05,
            alpha=0.3,
        )

        ax2.axvline(mean_mae, color=palette[2], linestyle="--", linewidth=1.5)
        ax2.axvline(median_mae, color=palette[3], linestyle="-.", linewidth=1.5)
        ax2.axvline(p95_mae, color=palette[4], linestyle=":", linewidth=1.5)

        ax2.set_xlabel("MAE [rad]")
        ax2.set_ylabel("Density")
        ax2.set_title("KDE with Outliers")
        ax2.grid(True, alpha=0.3)

        # Panel 3: CDF with confidence
        ax3 = fig.add_subplot(gs[2])

        sorted_maes = np.sort(maes)
        cdf = np.arange(1, len(sorted_maes) + 1) / len(sorted_maes)

        sns.lineplot(
            x=sorted_maes,
            y=cdf,
            ax=ax3,
            color=palette[0],
            linewidth=2,
        )

        # Percentile annotations
        ax3.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax3.axhline(0.95, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax3.axvline(median_mae, color=palette[3], linestyle="-.", linewidth=1.5)
        ax3.axvline(p95_mae, color=palette[4], linestyle=":", linewidth=1.5)

        # Annotation
        ax3.annotate(
            f"Median\n{median_mae:.4f}",
            xy=(median_mae, 0.5),
            xytext=(median_mae * 1.1, 0.6),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color=palette[3]),
        )

        ax3.set_xlabel("MAE [rad]")
        ax3.set_ylabel("Cumulative Probability")
        ax3.set_title("Cumulative Distribution")
        ax3.grid(True, alpha=0.3)

        plt.suptitle(
            f"Error Analysis (n={len(maes)}, μ={mean_mae:.4f}, σ={std_mae:.4f}, 95%={p95_mae:.4f})",
            fontsize=10,
            y=1.02,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved error histogram: {out_path}")
        print(f"  Mean: {mean_mae:.4f} | Median: {median_mae:.4f}")
        print(f"  Std: {std_mae:.4f} | 95th: {p95_mae:.4f} | 99th: {p99_mae:.4f}")
