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
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..core.utils import ensure_dir
from .utils import (
    to_numpy,
    wrap_phase,
    compute_quality_map,
    draw_intensity_panel,
    draw_phase_panel,
    draw_error_panel,
    draw_profile_panel,
)

_HEADLESS = not os.environ.get("DISPLAY")
if _HEADLESS:
    matplotlib.use("Agg")


def save_epoch_visuals(
    I_input: torch.Tensor,
    phi_pred_abs_aligned: torch.Tensor,
    phi_gt: torch.Tensor,
    out_dir: str,
    epoch: int,
    max_items: int = 8,
    I_raw: Optional[torch.Tensor] = None,
    noise_level: float = 0.0,
    samples_per_file: int = 4,
) -> None:
    """Save quick per-sample visualizations during training.

    Enhanced layout with all 7 tasks:
    - Row 1: Clean Input | Noisy Input (with noise level) | GT Phase | Wrapped Phase
    - Row 2: Pred Phase | Error Map (with histogram) | Quality Map | Phase Profile
    """
    ensure_dir(out_dir)

    B = min(I_input.shape[0], max_items)
    n_files = (B + samples_per_file - 1) // samples_per_file

    # Get model input
    I_noisy = to_numpy(I_input[:, 0:1])
    I_clean = to_numpy(I_raw) if I_raw is not None else None

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
            2,
            4 * current_batch_size,
            figsize=(4 * current_batch_size * 2.5, 5),
            squeeze=False,
        )

        for local_idx, i in enumerate(range(start_idx, end_idx)):
            col_offset = local_idx * 4

            # Panel 1: Clean Input
            ax = axes[0, col_offset]
            if I_clean is not None:
                draw_intensity_panel(
                    ax, I_clean[i, 0], f"Sample {i}\nClean Input", colorbar=False
                )
            else:
                ax.imshow(np.zeros((128, 128)), cmap="gray")
                ax.text(
                    64, 64, "N/A", ha="center", va="center", fontsize=10, color="white"
                )
                ax.set_title(f"Sample {i}\n(No Clean)", fontsize=8)
                ax.axis("off")

            # Panel 2: Noisy Input
            ax = axes[0, col_offset + 1]
            noise_text = f"Noise: {noise_level:.2f}" if noise_level > 0 else "No Noise"
            draw_intensity_panel(ax, I_noisy[i, 0], noise_text, colorbar=False)

            # Panel 3: GT Phase
            ax = axes[0, col_offset + 2]
            gt_np = to_numpy(gt_rad[i, 0])
            draw_phase_panel(ax, gt_np, "GT Phase [rad]")

            # Panel 4: Wrapped Phase
            ax = axes[0, col_offset + 3]
            draw_phase_panel(
                ax,
                wrap_phase(gt_np),
                "Wrapped",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )

            # Panel 5: Predicted Phase
            ax = axes[1, col_offset]
            pred_np = to_numpy(pred_rad[i, 0])
            draw_phase_panel(ax, pred_np, "Predicted [rad]")

            # Panel 6: Error Map
            ax = axes[1, col_offset + 1]
            err_raw = pred_rad[i] - gt_rad[i]
            mae = float(err_raw.abs().mean().item())
            rmse = float(torch.sqrt((err_raw**2).mean()).item())

            draw_error_panel(
                ax,
                to_numpy(err_raw[0].abs()),
                f"Error Map\nMAE:{mae:.3f} RMSE:{rmse:.3f}",
                vmax=emax,
            )

            # Add histogram inset
            inset_ax = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
            inset_ax.hist(
                to_numpy(err_raw[0]).flatten(),
                bins=30,
                density=True,
                alpha=0.7,
                color="steelblue",
            )
            inset_ax.set_xlim(-emax * 2, emax * 2)
            inset_ax.set_yticks([])
            inset_ax.axis("off")

            # Panel 7: Quality Map
            ax = axes[1, col_offset + 2]
            draw_phase_panel(ax, compute_quality_map(gt_np), "Quality Map", cmap="hot")

            # Panel 8: Profile
            ax = axes[1, col_offset + 3]
            draw_profile_panel(ax, gt_np, pred_np)

        plt.tight_layout()
        filename = f"epoch{epoch:03d}_{'sample' if samples_per_file == 1 else 'grid'}{start_idx if samples_per_file == 1 else file_idx}.png"
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
