"""
Noise comparison grid showing clean vs noisy inference.
Ensures consistency and robustness metrics are visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from ..core.ops import affine_align
from .style import (
    SINGLE_COL,
    nature_style,
    save_figure,
)
from ..data import build_dataloaders
from ..data.augmentation import prepare_batch
from .utils import (
    to_numpy,
    draw_error_panel,
    draw_phase_panel,
)


@torch.no_grad()
def plot_noise_comparison_grid(
    checkpoint_path: str,
    data_dir: str | None = None,
    out_path: str = "results/figs/noise_comparison",
    n_samples: int = 4,
    noise_level: float = 1.0,
    config_path: str | None = None,
) -> None:
    """
    Noise comparison grid showing clean vs noisy inference.

    Rows:
        1. Clean Path Errors
        2. Noisy Path Errors
        3. Difference (Noisy - Clean)
    """
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir=data_dir, config_path=config_path)

    # Build clean dataloader (no noise)
    cfg_clean = cfg.__class__()
    for k, v in cfg.__dict__.items():
        setattr(cfg_clean, k, v)
    cfg_clean.aug.enable = False

    train_loader_c, val_loader_c, test_loader_c = build_dataloaders(
        cfg_clean, device, seed=cfg.logging.seed
    )
    if subset == "test":
        if test_loader_c is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        clean_loader = test_loader_c
    elif subset == "val":
        clean_loader = (
            val_loader_c
            if (val_loader_c is not None and len(val_loader_c) > 0)
            else train_loader_c
        )
    else:
        clean_loader = train_loader_c

    clean_iter = iter(clean_loader)
    I_clean, phi_gt, I_raw_n, I_raw_c = next(clean_iter)
    I_clean = I_clean[:n_samples].to(device)
    phi_gt = phi_gt[:n_samples].to(device)

    with autocast(device_type=device.type, enabled=False):
        phi_raw_clean, k_off_clean = model(I_clean)
        phi_abs_clean = phi_raw_clean + k_off_clean

    phi_aligned_clean, _, _ = affine_align(phi_abs_clean, phi_gt)

    # Build noisy dataloader
    cfg_noisy = cfg.__class__()
    for k, v in cfg.__dict__.items():
        setattr(cfg_noisy, k, v)
    cfg_noisy.aug.enable = True

    train_loader_n, val_loader_n, test_loader_n = build_dataloaders(
        cfg_noisy, device, seed=cfg.logging.seed
    )
    if subset == "test":
        noisy_loader = test_loader_n
    elif subset == "val":
        noisy_loader = (
            val_loader_n
            if (val_loader_n is not None and len(val_loader_n) > 0)
            else train_loader_n
        )
    else:
        noisy_loader = train_loader_n

    # Apply noise level
    if (
        hasattr(noisy_loader.dataset, "noise_aug")
        and noisy_loader.dataset.noise_aug is not None
    ):
        noisy_loader.dataset.noise_aug.set_level(noise_level)

    noisy_iter = iter(noisy_loader)
    I_noisy, _, _, _ = next(noisy_iter)
    I_noisy = I_noisy[:n_samples].to(device)

    with autocast(device_type=device.type, enabled=False):
        phi_raw_noisy, k_off_noisy = model(I_noisy)
        phi_abs_noisy = phi_raw_noisy + k_off_noisy

    phi_aligned_noisy, _, _ = affine_align(phi_abs_noisy, phi_gt)

    # Convert to numpy
    pred_clean_np = to_numpy(phi_aligned_clean)
    pred_noisy_np = to_numpy(phi_aligned_noisy)
    gt_np = to_numpy(phi_gt)

    err_clean = np.abs(pred_clean_np - gt_np)
    err_noisy = np.abs(pred_noisy_np - gt_np)
    err_diff = err_noisy - err_clean

    # Calculate statistics
    mae_clean = err_clean.mean()
    mae_noisy = err_noisy.mean()
    mae_increase = mae_noisy - mae_clean
    mae_increase_pct = (mae_increase / mae_clean) * 100 if mae_clean > 0 else 0

    # Create figure with 3 rows
    with nature_style():
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
        axes[0, 0].set_ylabel(
            "Clean\nError", fontsize=8, rotation=0, ha="right", va="center"
        )
        axes[1, 0].set_ylabel(
            "Noisy\nError", fontsize=8, rotation=0, ha="right", va="center"
        )
        axes[2, 0].set_ylabel(
            "Error\nShift", fontsize=8, rotation=0, ha="right", va="center"
        )

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
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])

        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, str(save_path))
        print(f"Saved noise comparison: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate noise comparison grid.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/figs/noise_comp")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--noise_level", type=float, default=1.0)
    args = parser.parse_args()

    plot_noise_comparison_grid(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        out_path=args.out,
        n_samples=args.n_samples,
        noise_level=args.noise_level,
    )
