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
from scipy import ndimage

from ..core.utils import ensure_dir

_HEADLESS = not os.environ.get("DISPLAY")
if _HEADLESS:
    matplotlib.use("Agg")


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] range (simulating what classical methods see)."""
    return np.angle(np.exp(1j * phi))


def _compute_local_contrast(x: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute local contrast/quality map using rolling std."""
    return (
        ndimage.uniform_filter(x**2, window_size)
        - ndimage.uniform_filter(x, window_size) ** 2
    )


def _compute_quality_map(phase: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute phase quality map based on local contrast."""
    wrapped = _wrap_phase(phase)
    contrast = _compute_local_contrast(wrapped, window_size)
    return np.sqrt(np.clip(contrast, 0, None))


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

    Args:
        I_input: Noisy/normalized input [B, 2, H, W] - channel 0 is what model sees
        phi_pred_abs_aligned: Predicted phase (affine-aligned) [B, 1, H, W]
        phi_gt: Ground truth phase [B, 1, H, W]
        out_dir: Output directory
        epoch: Current epoch number
        max_items: Maximum samples to visualize
        I_raw: Clean/raw interferogram (if available) [B, 1, H, W] - Task 1
        noise_level: Current noise curriculum level (0-1) - Task 2
        samples_per_file: Number of samples per output file - Task 5
    """
    ensure_dir(out_dir)

    B = min(I_input.shape[0], max_items)
    n_files = (B + samples_per_file - 1) // samples_per_file

    # Get the actual model input (noisy normalized interferogram)
    I_noisy = _to_np(I_input[:, 0:1])  # Channel 0 = normalized interferogram

    # Get clean input if available (Task 1)
    if I_raw is not None:
        I_clean = _to_np(I_raw)
    else:
        I_clean = None

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad = phi_gt.float()

    # Compute global error scale
    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel() > 0 else 1.0

    # Process in batches of samples_per_file
    for file_idx in range(n_files):
        start_idx = file_idx * samples_per_file
        end_idx = min(start_idx + samples_per_file, B)
        current_batch_size = end_idx - start_idx

        # Create figure: 2 rows x (4 columns x current_batch_size) = 8 panels per row
        # Actually let's do: 2 rows, 4 cols per sample = 8 subplots per sample
        # But easier: 2 rows x 4 cols with each cell containing current_batch_size panels side by side

        fig, axes = plt.subplots(
            2 * current_batch_size,
            4,
            figsize=(10, 5 * current_batch_size),
            squeeze=False,
        )

        for local_idx, i in enumerate(range(start_idx, end_idx)):
            row_offset = local_idx * 2

            # ========== ROW 1 ==========

            # Panel 1: Clean Input (Task 1)
            ax = axes[row_offset, 0]
            if I_clean is not None:
                ax.imshow(I_clean[i, 0], cmap="gray")
            else:
                ax.imshow(np.zeros((128, 128)), cmap="gray")
                ax.text(64, 64, "N/A", ha="center", va="center", fontsize=10)
            sample_label = f"Sample {i}"
            ax.set_title(
                f"{sample_label}\nClean Input"
                if I_clean is not None
                else f"{sample_label}\n(No Clean)",
                fontsize=8,
            )
            ax.axis("off")

            # Panel 2: Noisy Input with noise level (Task 1, 2)
            ax = axes[row_offset, 1]
            ax.imshow(I_noisy[i, 0], cmap="gray")
            noise_text = f"Noise: {noise_level:.2f}" if noise_level > 0 else "No Noise"
            ax.set_title(noise_text, fontsize=8)
            ax.axis("off")

            # Panel 3: GT Phase
            ax = axes[row_offset, 2]
            gt_np = _to_np(gt_rad[i, 0])
            im_gt = ax.imshow(gt_np, cmap="viridis")
            ax.set_title("GT Phase [rad]", fontsize=8)
            ax.axis("off")
            plt.colorbar(im_gt, ax=ax, fraction=0.046, pad=0.04)

            # Panel 4: Wrapped Phase (Task 3)
            ax = axes[row_offset, 3]
            wrapped = _wrap_phase(gt_np)
            im_wrap = ax.imshow(wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.set_title("Wrapped (classical)", fontsize=8)
            ax.axis("off")
            plt.colorbar(im_wrap, ax=ax, fraction=0.046, pad=0.04)

            # ========== ROW 2 ==========

            # Panel 5: Predicted Phase
            ax = axes[row_offset + 1, 0]
            pred_np = _to_np(pred_rad[i, 0])
            im_pred = ax.imshow(pred_np, cmap="viridis")
            ax.set_title("Predicted [rad]", fontsize=8)
            ax.axis("off")
            plt.colorbar(im_pred, ax=ax, fraction=0.046, pad=0.04)

            # Panel 6: Error Map with Stats + Histogram (Task 4)
            ax = axes[row_offset + 1, 1]
            err_raw = pred_rad[i] - gt_rad[i]
            err_np = _to_np(err_raw[0])
            bias = float(err_raw.mean().item())
            std = float(err_raw.std(unbiased=False).item())
            mae = float(err_raw.abs().mean().item())
            rmse = float(torch.sqrt((err_raw**2).mean()).item())

            im_err = ax.imshow(err_np, cmap="PuOr", vmin=-emax, vmax=emax)
            ax.set_title(f"Error Map\nMAE:{mae:.3f} RMSE:{rmse:.3f}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)

            # Add histogram inset (Task 4)
            inset_ax = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
            inset_ax.hist(
                err_np.flatten(), bins=30, density=True, alpha=0.7, color="steelblue"
            )
            inset_ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
            inset_ax.set_xlim(-emax * 2, emax * 2)
            inset_ax.set_yticks([])
            inset_ax.set_title(f"μ={bias:+.3f} σ={std:.3f}", fontsize=5)

            # Panel 7: Phase Quality Map (Task 7)
            ax = axes[row_offset + 1, 2]
            quality = _compute_quality_map(gt_np)
            im_qual = ax.imshow(quality, cmap="hot")
            ax.set_title("Quality Map", fontsize=8)
            ax.axis("off")
            plt.colorbar(im_qual, ax=ax, fraction=0.046, pad=0.04)

            # Panel 8: Phase Profile (Task 6)
            ax = axes[row_offset + 1, 3]
            H, W = gt_np.shape
            center_row = H // 2
            ax.plot(gt_np[center_row, :], label="GT", linewidth=1.5, alpha=0.8)
            ax.plot(pred_np[center_row, :], label="Pred", linewidth=1.5, alpha=0.8)
            ax.set_title("Phase Profile (row)", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xlabel("x", fontsize=7)
            ax.set_ylabel("phase [rad]", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with naming convention
        if samples_per_file == 1:
            filename = f"epoch{epoch:03d}_sample{start_idx}.png"
        else:
            filename = f"epoch{epoch:03d}_grid{file_idx}.png"

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
    """Legacy function for backward compatibility.

    Saves 2x2 grid per sample: Interferogram | GT | Pred | Error
    """
    ensure_dir(out_dir)
    B = min(I_input.shape[0], max_items)

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad = phi_gt.float()

    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel() > 0 else 1.0
    vmin, vmax = -emax, emax

    for i in range(B):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].imshow(_to_np(I_input[i, 0]), cmap="gray")
        axs[0, 0].set_title("Interferogram")
        axs[0, 0].axis("off")

        im_gt = axs[0, 1].imshow(_to_np(gt_rad[i, 0]), cmap="viridis")
        axs[0, 1].set_title("GT phase [rad]")
        axs[0, 1].axis("off")
        plt.colorbar(im_gt, ax=axs[0, 1], fraction=0.046)

        im_pred = axs[1, 0].imshow(_to_np(pred_rad[i, 0]), cmap="viridis")
        axs[1, 0].set_title("Predicted phase [rad]")
        axs[1, 0].axis("off")
        plt.colorbar(im_pred, ax=axs[1, 0], fraction=0.046)

        err_raw = pred_rad[i] - gt_rad[i]
        bias = float(err_raw.mean().item())
        std = float(err_raw.std(unbiased=False).item())

        im_err = axs[1, 1].imshow(_to_np(err_raw[0]), cmap="PuOr", vmin=vmin, vmax=vmax)
        axs[1, 1].set_title("Error [rad]")
        axs[1, 1].axis("off")
        plt.colorbar(im_err, ax=axs[1, 1], fraction=0.046)

        axs[1, 1].text(
            0.02,
            0.98,
            f"mean={bias:+.3f}\nstd={std:.3f}",
            transform=axs[1, 1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}.png"), dpi=150)
        plt.close(fig)
