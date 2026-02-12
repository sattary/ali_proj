from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import ensure_dir


# Handle headless plotting (no GUI)
HEADLESS = not os.environ.get("DISPLAY")
if HEADLESS:
    matplotlib.use("Agg")  # type: ignore[func-returns-value]


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
    """
    Save visualizations of interferograms, GT phase, predictions, and error maps.
    """
    ensure_dir(out_dir)
    B = min(I_input.shape[0], max_items)

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad = phi_gt.float()

    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel() > 0 else 1.0
    vmin, vmax = -emax, emax
    cmap_div = "PuOr"

    for i in range(B):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # channel 0 is normalized interferogram
        axs[0, 0].imshow(_to_np(I_input[i, 0]), cmap="gray")
        axs[0, 0].set_title("I_norm[0]")
        axs[0, 0].axis("off")

        # GT absolute phase
        im_gt = axs[0, 1].imshow(_to_np(gt_rad[i, 0]), cmap="viridis")
        axs[0, 1].set_title("GT φ (rad)")
        axs[0, 1].axis("off")
        cb_gt = plt.colorbar(im_gt, ax=axs[0, 1], fraction=0.046)
        cb_gt.set_label("φ (rad)")

        # predicted absolute phase (aligned)
        im_pred = axs[1, 0].imshow(_to_np(pred_rad[i, 0]), cmap="viridis")
        axs[1, 0].set_title("Pred φ_abs_aligned (rad)")
        axs[1, 0].axis("off")
        cb_pred = plt.colorbar(im_pred, ax=axs[1, 0], fraction=0.046)
        cb_pred.set_label("φ (rad)")

        # error map
        err_raw = pred_rad[i] - gt_rad[i]
        bias = float(err_raw.mean().item())
        std = float(err_raw.std(unbiased=False).item())

        im_err = axs[1, 1].imshow(_to_np(err_raw[0]), cmap=cmap_div, vmin=vmin, vmax=vmax)
        axs[1, 1].set_title("Δφ = Pred_aligned − GT (rad)")
        axs[1, 1].axis("off")
        cb_err = plt.colorbar(im_err, ax=axs[1, 1], fraction=0.046)
        cb_err.set_label("Δφ (rad)")

        # annotate bias/std
        axs[1, 1].text(
            0.02,
            0.98,
            f"mean={bias:+.3f} rad\nstd={std:.3f} rad",
            transform=axs[1, 1].transAxes,
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )

        plt.tight_layout()

        fig_path = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


def save_training_curve(
    epochs: Sequence[int],
    train_losses: Sequence[float],
    val_maes: Sequence[float],
    out_dir: str,
    filename: str = "training_curve.png",
) -> None:
    """
    Save a training curve plotting train loss and validation MAE over epochs.
    """
    ensure_dir(out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.grid(True)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_maes, label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


