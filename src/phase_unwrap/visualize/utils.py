"""
Centralized visualization utilities and reusable components.
Ensures consistency between training monitoring and paper figures.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from .style import add_colorbar, CMAP_INTENSITY


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert torch tensor or any array to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] range."""
    return np.angle(np.exp(1j * phi))


def compute_quality_map(phase: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute phase quality map based on local contrast."""
    wrapped = wrap_phase(phase)
    # Compute local variance
    mean = ndimage.uniform_filter(wrapped, window_size)
    mean_sq = ndimage.uniform_filter(wrapped**2, window_size)
    contrast = mean_sq - mean**2
    return np.sqrt(np.clip(contrast, 0, None))


def draw_intensity_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    cmap: str = CMAP_INTENSITY,
    colorbar: bool = True,
) -> None:
    """Draw an intensity/interferogram panel."""
    im = ax.imshow(data, cmap=cmap, aspect="equal")
    if colorbar:
        add_colorbar(ax, im)
    ax.set_title(title, fontsize=8, pad=5)
    ax.axis("off")


def draw_phase_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    cmap: str = "viridis",
    label: str = "[rad]",
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Draw a phase panel (wrapped or absolute)."""
    im = ax.imshow(data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    if colorbar:
        add_colorbar(ax, im, label=label)
    ax.set_title(title, fontsize=8, pad=5)
    ax.axis("off")


def draw_error_panel(
    ax: plt.Axes,
    error_data: np.ndarray,
    title: str,
    cmap: str = "rocket",
    label: str = "[rad]",
    colorbar: bool = True,
    vmax: Optional[float] = None,
    stats: Optional[dict] = None,
) -> None:
    """Draw an error map panel with optional statistics overlay."""
    im = ax.imshow(error_data, cmap=cmap, aspect="equal", vmin=0, vmax=vmax)
    if colorbar:
        add_colorbar(ax, im, label=label)

    if stats:
        mae = stats.get("mae", np.mean(np.abs(error_data)))
        ax.text(
            0.98,
            0.98,
            f"MAE: {mae:.4f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

    ax.set_title(title, fontsize=8, pad=5)
    ax.axis("off")


def draw_profile_panel(
    ax: plt.Axes,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str = "Phase Profile (row)",
    label_gt: str = "GT",
    label_pred: str = "Pred",
) -> None:
    """Draw 1D cross-section comparison."""
    H, W = gt.shape
    center_row = H // 2
    ax.plot(gt[center_row, :], label=label_gt, linewidth=1.5, alpha=0.8)
    ax.plot(pred[center_row, :], label=label_pred, linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("phase [rad]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
