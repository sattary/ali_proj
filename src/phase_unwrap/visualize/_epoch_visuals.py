"""
Per-epoch training visuals (quick sanity-check PNGs).

This module is the internal helper used by train.py during training.
For publication-quality figures, use the dedicated plot modules.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..core.utils import ensure_dir

# Handle headless plotting (no GUI)
_HEADLESS = not os.environ.get("DISPLAY")
if _HEADLESS:
    matplotlib.use("Agg")


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def save_epoch_visuals(
    I_input: torch.Tensor,
    phi_pred_abs_aligned: torch.Tensor,
    phi_gt: torch.Tensor,
    out_dir: str,
    epoch: int,
    max_items: int = 8,
) -> None:
    """Save quick per-sample visualizations during training."""
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
