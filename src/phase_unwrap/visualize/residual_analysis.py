"""
Residual analysis suite for model validation.

Three-panel diagnostic plot:
1. Residuals vs Fitted (check homoscedasticity)
2. Q-Q plot (normality check)
3. Residual histogram with normal overlay
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from torch.amp import autocast

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
def plot_residual_analysis(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/residual_analysis",
    config_path: str | None = None,
    max_samples: int = 500,
) -> None:
    """
    Comprehensive residual diagnostic plots.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        out_path: Output path (without extension)
        config_path: Optional config override
        max_samples: Maximum samples to analyze
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

    _, val_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed
    )
    loader = (
        val_loader
        or build_dataloaders(cfg, device, seed=cfg.logging.seed)[0]
    )

    # Collect predictions
    all_pred = []
    all_gt = []

    for I_input, phi_gt, _ in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            phi_abs = phi_raw + k_off

        aligned, _, _ = affine_align(phi_abs, phi_gt)

        all_pred.extend(aligned.cpu().numpy().flatten())
        all_gt.extend(phi_gt.cpu().numpy().flatten())

        if len(all_pred) >= max_samples * 100:
            break

    pred = np.array(all_pred[: max_samples * 100])
    gt = np.array(all_gt[: max_samples * 100])

    residuals = pred - gt
    fitted = pred

    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for Shapiro-Wilk
    _, ks_p = stats.kstest(residuals, "norm", args=(residuals.mean(), residuals.std()))

    with nature_style():
        palette = create_nature_palette(6)
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))

        # Panel 1: Residuals vs Fitted
        ax1 = axes[0]
        sns.scatterplot(
            x=fitted[::100],
            y=residuals[::100],
            ax=ax1,
            alpha=0.4,
            s=10,
            color=palette[0],
        )
        ax1.axhline(0, color="red", linestyle="--", linewidth=1)

        # Add LOESS smoothing
        try:
            sns.regplot(
                x=fitted,
                y=residuals,
                scatter=False,
                lowess=True,
                ax=ax1,
                color=palette[1],
                line_kws={"linewidth": 2},
            )
        except Exception:
            pass

        ax1.set_xlabel("Fitted Values [rad]", fontsize=9)
        ax1.set_ylabel("Residuals [rad]", fontsize=9)
        ax1.set_title("Residuals vs Fitted", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Q-Q plot
        ax2 = axes[1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor(palette[0])
        ax2.get_lines()[0].set_markersize(3)
        ax2.get_lines()[0].set_alpha(0.5)
        ax2.get_lines()[1].set_color("red")
        ax2.get_lines()[1].set_linewidth(1.5)
        ax2.set_title("Normal Q-Q Plot", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Histogram with normal overlay
        ax3 = axes[2]

        sns.histplot(
            residuals, kde=True, ax=ax3, color=palette[0], alpha=0.6, stat="density"
        )

        # Overlay normal distribution
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        ax3.plot(x_norm, y_norm, color="red", linewidth=2, label="Normal fit")

        ax3.axvline(0, color="black", linestyle="--", linewidth=1)
        ax3.set_xlabel("Residuals [rad]", fontsize=9)
        ax3.set_ylabel("Density", fontsize=9)
        ax3.set_title("Residual Distribution", fontsize=10)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Shapiro-Wilk: p={shapiro_p:.2e}\n"
            f"Kolmogorov-Smirnov: p={ks_p:.2e}\n"
            f"Mean: {residuals.mean():.4f}\n"
            f"Std: {residuals.std():.4f}"
        )
        ax3.text(
            0.98,
            0.98,
            stats_text,
            transform=ax3.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(
            f"Residual Analysis (n={len(residuals):,}, "
            f"{'Normal' if shapiro_p > 0.05 else 'Non-normal'} residuals)",
            fontsize=11,
            y=1.02,
        )

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved residual analysis: {out_path}")