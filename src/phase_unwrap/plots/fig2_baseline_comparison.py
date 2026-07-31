"""
Figure 2: Visual Baseline Superiority Matrix comparing classical vs Deep Learning models.

Stateless implementation.
Columns: Interferogram | Ground Truth | Itoh 1D | Least-Squares | PCLCN (Ours)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import DOUBLE_COL, create_nature_palette, label_panels, publication_plot
from .utils import draw_intensity_panel, draw_phase_panel


@publication_plot
def plot_f2_baseline_comparison(
    raw_i_np: np.ndarray,
    gt_np: np.ndarray,
    itoh_np: np.ndarray,
    lsq_np: np.ndarray,
    unet_np: np.ndarray,
    filepath: str | None = None,
) -> plt.Figure:
    """
    Nature-style multiclass visual comparison.
    
    Args:
        raw_i_np: (N, 1, H, W) interferograms.
        gt_np: (N, 1, H, W) ground truth unwrapped phase.
        itoh_np: (N, 1, H, W) Itoh 1D unwrapped phase.
        lsq_np: (N, 1, H, W) Least-Squares unwrapped phase.
        unet_np: (N, 1, H, W) PCLCN model unwrapped phase.
    """
    n_samples = raw_i_np.shape[0]

    palette = create_nature_palette()

    fig, axes = plt.subplots(
        n_samples,
        5,
        figsize=(DOUBLE_COL * 1.25, DOUBLE_COL * 0.26 * n_samples),
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Interferogram",
        r"Ground Truth $\varphi_{\mathrm{GT}}$",
        "Itoh 1D",
        "Least-Squares",
        "PCLCN (Ours)",
    ]

    for row in range(n_samples):
        I_img = raw_i_np[row, 0]
        gt_img = gt_np[row, 0]
        unet_img = unet_np[row, 0]
        itoh_img = itoh_np[row, 0]
        lsq_img = lsq_np[row, 0]

        vmin_phase = gt_img.min()
        vmax_phase = gt_img.max()

        draw_intensity_panel(axes[row, 0], I_img, "")
        draw_phase_panel(axes[row, 1], gt_img, "", vmin=vmin_phase, vmax=vmax_phase)
        draw_phase_panel(axes[row, 2], itoh_img, "", vmin=vmin_phase, vmax=vmax_phase)
        draw_phase_panel(axes[row, 3], lsq_img, "", vmin=vmin_phase, vmax=vmax_phase)
        draw_phase_panel(axes[row, 4], unet_img, "", vmin=vmin_phase, vmax=vmax_phase)

        def ann_mae(ax, pred):
            mae = np.abs(pred - gt_img).mean()
            ax.text(
                0.98,
                0.98,
                f"MAE: {mae:.4f}",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
                color="black",
                bbox=dict(boxstyle="round", facecolor=palette[0], alpha=0.8),
            )

        ann_mae(axes[row, 2], itoh_img)
        ann_mae(axes[row, 3], lsq_img)
        ann_mae(axes[row, 4], unet_img)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, pad=5)

    label_panels(axes.flat)
    fig.tight_layout(pad=0.5)

    return fig
