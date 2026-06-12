"""
Test-Time Augmentation (TTA) benefit visualization.

Shows per-sample improvements and statistical significance of TTA.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..core.ops import affine_align
from ..analysis.tta import predict_tta
from .style import (
    DOUBLE_COL,
    annotate_significance,
    compute_statistical_test,
    create_nature_palette,
    nature_style,
    save_figure,
)


@torch.no_grad()
def plot_tta_benefit(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/tta_benefit",
    config_path: str | None = None,
    n_augments: int = 8,
    max_samples: int = 100,
    subset: str = "val",
    all_data: bool = False,
) -> None:
    """
    Visualize TTA benefits with statistical analysis.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        out_path: Output path (without extension)
        config_path: Optional config override
        n_augments: Number of TTA augmentations
        max_samples: Maximum samples to evaluate
    """
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )

    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = (
            val_loader
            if (val_loader is not None and len(val_loader) > 0)
            else train_loader
        )
    else:
        loader = train_loader

    # Collect errors with and without TTA
    errors_no_tta = []
    errors_tta = []
    sample_indices = []

    for batch_idx, (I_input, phi_gt, _, _) in enumerate(loader):
        if batch_idx >= max_samples // I_input.size(0):
            break

        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        # Without TTA
        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            if isinstance(phi_raw, list):
                phi_no_tta = phi_raw[-1] + k_off
            else:
                phi_no_tta = phi_raw + k_off
        aligned_no_tta, _, _ = affine_align(phi_no_tta, phi_gt)

        # With TTA
        phi_tta = predict_tta(
            model,
            I_input,
            device,
            augments=[f"rot{i}" for i in range(min(n_augments, 4))],
        )
        aligned_tta, _, _ = affine_align(phi_tta, phi_gt)

        # Per-sample MAE
        for i in range(I_input.size(0)):
            mae_no_tta = (aligned_no_tta[i] - phi_gt[i]).abs().mean().item()
            mae_tta = (aligned_tta[i] - phi_gt[i]).abs().mean().item()

            errors_no_tta.append(mae_no_tta)
            errors_tta.append(mae_tta)
            sample_indices.append(batch_idx * I_input.size(0) + i)

    errors_no_tta = np.array(errors_no_tta)
    errors_tta = np.array(errors_tta)
    improvements = errors_no_tta - errors_tta

    # Statistical test
    _, pval = compute_statistical_test(errors_no_tta, errors_tta, test="wilcoxon")
    mean_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)

    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        # Panel 1: Paired comparison
        ax1 = fig.add_subplot(gs[0])

        data_for_plot = {
            "No TTA": errors_no_tta,
            f"TTA (n={n_augments})": errors_tta,
        }

        positions = [0, 1]
        bp = ax1.boxplot(
            [errors_no_tta, errors_tta],
            positions=positions,
            patch_artist=True,
            widths=0.6,
        )

        for patch, color in zip(bp["boxes"], palette[:2]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Individual points with jitter
        for i, (label, values) in enumerate(data_for_plot.items()):
            jitter = np.random.normal(i, 0.04, len(values))
            ax1.scatter(jitter, values, alpha=0.4, s=8, color="black")

        # Connect paired samples
        for i in range(min(50, len(errors_no_tta))):  # Limit lines for clarity
            ax1.plot(
                [0, 1],
                [errors_no_tta[i], errors_tta[i]],
                color="gray",
                alpha=0.2,
                linewidth=0.5,
            )

        # Significance annotation
        y_max = max(errors_no_tta.max(), errors_tta.max())
        annotate_significance(ax1, 0, 1, y_max * 1.05, pval)

        ax1.set_xticklabels(data_for_plot.keys(), fontsize=8)
        ax1.set_ylabel("MAE [rad]", fontsize=9)
        ax1.set_title("TTA vs Baseline", fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Improvement distribution
        ax2 = fig.add_subplot(gs[1])

        sns.histplot(improvements, kde=True, ax=ax2, color=palette[0], alpha=0.6)
        ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax2.axvline(
            mean_improvement,
            color=palette[2],
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_improvement:.4f}",
        )
        ax2.axvline(
            median_improvement,
            color=palette[4],
            linestyle="-.",
            linewidth=2,
            label=f"Median: {median_improvement:.4f}",
        )

        pct_improved = 100 * np.sum(improvements > 0) / len(improvements)

        ax2.set_xlabel("MAE Improvement [rad]", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title(
            f"Improvement Distribution\n({pct_improved:.1f}% improved)", fontsize=10
        )
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Per-sample improvement
        ax3 = fig.add_subplot(gs[2])

        sorted_idx = np.argsort(improvements)
        colors_sorted = [
            palette[2] if improvements[i] > 0 else palette[3] for i in sorted_idx
        ]

        ax3.barh(
            range(len(improvements)),
            improvements[sorted_idx],
            color=colors_sorted,
            alpha=0.7,
        )
        ax3.axvline(0, color="black", linestyle="-", linewidth=1)

        ax3.set_xlabel("MAE Improvement [rad]", fontsize=9)
        ax3.set_ylabel("Sample (sorted)", fontsize=9)
        ax3.set_title("Per-Sample Improvement", fontsize=10)
        ax3.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            f"Test-Time Augmentation Analysis (n={len(improvements)}, p={pval:.2e})",
            fontsize=11,
            y=1.02,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved TTA benefit plot: {out_path}")
        print(
            f"  Mean improvement: {mean_improvement:.4f} rad ({mean_improvement / errors_no_tta.mean() * 100:.1f}%)"
        )
        print(f"  {pct_improved:.1f}% of samples improved")
