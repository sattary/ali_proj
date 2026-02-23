"""
Noise Comparison Grid - Dedicated figure for clean vs noisy analysis.

Shows:
- Clean path: Clean Input → Clean Prediction
- Noisy path: Noisy Input → Noisy Prediction
- Analysis: Error comparison, difference map, statistics

Publication-ready figure for papers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..core.config import load_train_config
from ..core.ops import affine_align
from ..core.utils import pick_device
from ..data import build_dataloaders
from ..model import build_model
from .style import (
    CMAP_ERROR_ABS,
    CMAP_INTENSITY,
    DOUBLE_COL,
    SINGLE_COL,
    add_colorbar,
    create_nature_palette,
    nature_style,
    save_figure,
)


@torch.no_grad()
def plot_noise_comparison_grid(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/noise_comparison",
    n_samples: int = 4,
    noise_level: float = 1.0,
    config_path: str | None = None,
) -> None:
    """
    Noise comparison grid showing clean vs noisy inference.

    Figure layout:
        Row 1 (Clean Path):  Clean Input | Prediction | Error
        Row 2 (Noisy Path): Noisy Input | Prediction | Error
        Row 3 (Analysis):   Error Diff | Statistics

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        out_path: Output file path
        n_samples: Number of samples to visualize
        noise_level: Noise level to apply (0.0 = clean, 1.0 = max)
        config_path: Optional config file path
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    # Build clean dataloader (no noise)
    cfg_clean = cfg.__class__()
    cfg_clean.data = cfg.data
    cfg_clean.model = cfg.model
    cfg_clean.aug = cfg.aug
    cfg_clean.aug.enable = False

    _, clean_loader = build_dataloaders(cfg_clean, device, seed=cfg.logging.seed)
    clean_iter = iter(clean_loader)
    I_clean, phi_gt, I_raw_clean = next(clean_iter)
    I_clean = I_clean[:n_samples].to(device)
    phi_gt = phi_gt[:n_samples].to(device)
    I_raw_clean = I_raw_clean[:n_samples]

    with autocast(device_type=device.type, enabled=False):
        phi_raw_clean, k_off_clean = model(I_clean)
        phi_abs_clean = phi_raw_clean + k_off_clean

    phi_aligned_clean, _, _ = affine_align(phi_abs_clean, phi_gt)

    # Build noisy dataloader
    cfg_noisy = cfg.__class__()
    cfg_noisy.data = cfg.data
    cfg_noisy.model = cfg.model
    cfg_noisy.aug = cfg.aug
    cfg_noisy.data.augment = True

    _, noisy_loader = build_dataloaders(cfg_noisy, device, seed=cfg.logging.seed)
    noisy_iter = iter(noisy_loader)
    I_noisy, _, I_raw_noisy = next(noisy_iter)
    I_noisy = I_noisy[:n_samples].to(device)
    I_raw_noisy = I_raw_noisy[:n_samples]

    # Apply noise at specified level
    if (
        hasattr(noisy_loader.dataset, "noise_aug")
        and noisy_loader.dataset.noise_aug is not None
    ):
        noisy_loader.dataset.noise_aug.set_level(noise_level)

    with autocast(device_type=device.type, enabled=False):
        phi_raw_noisy, k_off_noisy = model(I_noisy)
        phi_abs_noisy = phi_raw_noisy + k_off_noisy

    phi_aligned_noisy, _, _ = affine_align(phi_abs_noisy, phi_gt)

    # Convert to numpy
    pred_clean_np = phi_aligned_clean.cpu().numpy()
    pred_noisy_np = phi_aligned_noisy.cpu().numpy()
    gt_np = phi_gt.cpu().numpy()
    raw_clean_np = I_raw_clean.numpy()
    raw_noisy_np = I_raw_noisy.numpy()

    err_clean = np.abs(pred_clean_np - gt_np)
    err_noisy = np.abs(pred_noisy_np - gt_np)
    err_diff = err_noisy - err_clean

    # Calculate statistics
    mae_clean = err_clean.mean()
    mae_noisy = err_noisy.mean()
    mae_increase = mae_noisy - mae_clean
    mae_increase_pct = (mae_increase / mae_clean) * 100

    rmse_clean = np.sqrt((err_clean**2).mean())
    rmse_noisy = np.sqrt((err_noisy**2).mean())

    # Per-sample stats
    sample_maes_clean = [err_clean[i].mean() for i in range(n_samples)]
    sample_maes_noisy = [err_noisy[i].mean() for i in range(n_samples)]

    # Create figure with 3 rows
    with nature_style():
        palette = create_nature_palette(6)

        fig, axes = plt.subplots(
            3,
            n_samples,
            figsize=(SINGLE_COL * 1.5, SINGLE_COL * 1.2),
        )

        phase_cmap = sns.color_palette("viridis", as_cmap=True)
        error_cmap = sns.color_palette("rocket", as_cmap=True)

        # ========== ROW 1: Clean Path ==========
        for i in range(n_samples):
            ax = axes[0, i]
            im = ax.imshow(raw_clean_np[i, 0], cmap=CMAP_INTENSITY, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_ylabel("Clean", fontsize=8, rotation=0, ha="right", va="center")

            if i == n_samples // 2:
                ax.set_title("Clean Input → Prediction → Error", fontsize=9, pad=10)

        # Clean predictions
        for i in range(n_samples):
            ax = axes[0, i]
            im = ax.imshow(pred_clean_np[i, 0], cmap=phase_cmap, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # Clean errors
        for i in range(n_samples):
            ax = axes[0, i]
            im = ax.imshow(err_clean[i, 0], cmap=error_cmap, aspect="equal")
            ax.text(
                0.98,
                0.98,
                f"MAE: {sample_maes_clean[i]:.4f}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor=palette[0], alpha=0.8),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # ========== ROW 2: Noisy Path ==========
        for i in range(n_samples):
            ax = axes[1, i]
            im = ax.imshow(raw_noisy_np[i, 0], cmap=CMAP_INTENSITY, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_ylabel("Noisy", fontsize=8, rotation=0, ha="right", va="center")

        # Noisy predictions
        for i in range(n_samples):
            ax = axes[1, i]
            im = ax.imshow(pred_noisy_np[i, 0], cmap=phase_cmap, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # Noisy errors
        for i in range(n_samples):
            ax = axes[1, i]
            im = ax.imshow(err_noisy[i, 0], cmap=error_cmap, aspect="equal")
            ax.text(
                0.98,
                0.98,
                f"MAE: {sample_maes_noisy[i]:.4f}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor=palette[0], alpha=0.8),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # ========== ROW 3: Analysis ==========
        # Error difference
        for i in range(n_samples):
            ax = axes[2, i]
            vmax_diff = max(abs(err_diff[i].min()), abs(err_diff[i].max()))
            im = ax.imshow(
                err_diff[i, 0],
                cmap="PuOr",
                aspect="equal",
                vmin=-vmax_diff,
                vmax=vmax_diff,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_ylabel(
                    "Δ Error", fontsize=8, rotation=0, ha="right", va="center"
                )

        # Add statistics text box
        stats_text = (
            f"Clean MAE: {mae_clean:.4f}\n"
            f"Noisy MAE: {mae_noisy:.4f}\n"
            f"Increase: +{mae_increase:.4f} ({mae_increase_pct:+.1f}%)\n"
            f"Clean RMSE: {rmse_clean:.4f}\n"
            f"Noisy RMSE: {rmse_noisy:.4f}"
        )

        # Create a centered statistics annotation
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
            f"Noise Robustness Analysis (Noise Level: {noise_level:.1f})",
            fontsize=10,
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)

        print(f"Saved noise comparison grid: {out_path}")
        print(
            f"  Clean MAE: {mae_clean:.4f} | Noisy MAE: {mae_noisy:.4f} | Increase: {mae_increase_pct:+.1f}%"
        )
        print(f"  Per-sample clean MAEs: {[f'{m:.4f}' for m in sample_maes_clean]}")
        print(f"  Per-sample noisy MAEs: {[f'{m:.4f}' for m in sample_maes_noisy]}")
