"""
Per-epoch training visuals (quick sanity-check PNGs).

This module is the internal helper used by train.py during training.
For publication-quality figures, use the dedicated plot modules.

Enhancements (Tasks 1-7):
1. Show both clean + noisy input side by side
2. Noise level indicator overlay
3. Visualize wrapped phase (classical view)
4. Error heatmap with histogram + stats overlay
5. Multi-row comparison (4 samples per file)
6. Phase profile (1D cross-section)
7. Phase quality map
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
_HEADLESS = not os.environ.get("DISPLAY")
if _HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ..data.augmentation import prepare_batch

from ..core.utils import ensure_dir
from .utils import (
    to_numpy,
    wrap_phase,
    draw_intensity_panel,
    draw_phase_panel,
    draw_error_panel,
)


def save_epoch_visuals(
    I_input: torch.Tensor,
    phi_pred_abs_aligned: torch.Tensor,
    phi_gt: torch.Tensor,
    out_dir: str,
    epoch: int,
    max_items: int = 8,
    I_raw_clean: Optional[torch.Tensor] = None,
    I_raw_noisy: Optional[torch.Tensor] = None,
    noise_level: float = 0.0,
    samples_per_file: int = 4,
) -> None:
    """Save quick per-sample visualizations during training.

    Refactored to a 1-row layout per sample:
    [Noisy Input] | [GT Phase] | [Wrapped Phase] | [Pred Phase] | [Error Map]
    """
    ensure_dir(out_dir)

    B = min(I_input.shape[0], max_items)
    n_files = (B + samples_per_file - 1) // samples_per_file

    # Get model input
    # Channel 0 of I_input is ALWAYS the normalized (Z-scored) input.
    I_noisy = (
        to_numpy(I_raw_noisy) if I_raw_noisy is not None else to_numpy(I_input[:, 0:1])
    )
    I_clean = to_numpy(I_raw_clean) if I_raw_clean is not None else None

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad = phi_gt.float()

    # Global error scale
    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel() > 0 else 1.0

    for file_idx in range(n_files):
        start_idx = file_idx * samples_per_file
        end_idx = min(start_idx + samples_per_file, B)
        current_batch_size = end_idx - start_idx

        fig, axes = plt.subplots(
            current_batch_size,
            6,
            figsize=(18, 3.5 * current_batch_size),
            squeeze=False,
        )

        for local_idx, i in enumerate(range(start_idx, end_idx)):
            # Panel 1: Clean Input (Pristine physics)
            ax = axes[local_idx, 0]
            if I_clean is not None:
                draw_intensity_panel(ax, I_clean[i, 0], "Clean Input", colorbar=False)
            else:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, "No Clean\nData", ha="center", va="center", fontsize=8
                )

            # Panel 2: Noisy Input (What model sees)
            ax = axes[local_idx, 1]
            noise_text = (
                f"Noisy Input (L:{noise_level:.2f})"
                if noise_level > 0
                else "Input (Clean)"
            )
            draw_intensity_panel(ax, I_noisy[i, 0], noise_text, colorbar=False)

            # Panel 3: GT Phase
            ax = axes[local_idx, 2]
            gt_np = to_numpy(gt_rad[i, 0])
            draw_phase_panel(ax, gt_np, "GT Phase")

            # Panel 4: Wrapped Phase
            ax = axes[local_idx, 3]
            draw_phase_panel(
                ax,
                wrap_phase(gt_np),
                "Wrapped GT",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )

            # Panel 5: Predicted Phase
            ax = axes[local_idx, 4]
            pred_np = to_numpy(pred_rad[i, 0])
            draw_phase_panel(ax, pred_np, "Prediction")

            # Panel 6: Error Map
            ax = axes[local_idx, 5]
            err_raw = pred_rad[i] - gt_rad[i]
            mae = float(err_raw.abs().mean().item())
            rmse = float(torch.sqrt((err_raw**2).mean()).item())

            draw_error_panel(
                ax,
                to_numpy(err_raw[0].abs()),
                f"Error\nMAE:{mae:.3f} RMSE:{rmse:.3f}",
                vmax=emax,
            )

        plt.tight_layout()
        filename = f"epoch{epoch:03d}_batch{file_idx}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_epoch_visuals_simple(
    I_input: torch.Tensor,
    phi_pred_abs_aligned: torch.Tensor,
    phi_gt: torch.Tensor,
    out_dir: str,
    epoch: int,
    max_items: int = 8,
) -> None:
    """Legacy function for backward compatibility."""
    ensure_dir(out_dir)
    B = min(I_input.shape[0], max_items)

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad = phi_gt.float()

    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel() > 0 else 1.0

    for i in range(B):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        draw_intensity_panel(axes[0, 0], to_numpy(I_input[i, 0]), "Interferogram")
        draw_phase_panel(axes[0, 1], to_numpy(gt_rad[i, 0]), "GT phase [rad]")
        draw_phase_panel(axes[1, 0], to_numpy(pred_rad[i, 0]), "Predicted phase [rad]")

        err_raw = pred_rad[i] - gt_rad[i]
        draw_error_panel(
            axes[1, 1], to_numpy(err_raw[0].abs()), "Error [rad]", vmax=emax
        )

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}.png"), dpi=150)
        plt.close(fig)
