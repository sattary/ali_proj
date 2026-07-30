"""
Figure 1: Qualitative architecture baseline / inference grid.

Publication-ready N-row grid showing:
- Clean Input | Noisy Input | Ground Truth | Wrapped Phase | Prediction | Error

Stateless implementation: expects pure numpy arrays as input.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import DOUBLE_COL, publication_plot, label_panels
from .utils import wrap_phase, draw_intensity_panel, draw_phase_panel, draw_error_panel


@publication_plot
def plot_f1_architecture(
    raw_clean_np: np.ndarray,
    raw_noisy_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    show_noise: bool = True,
    filepath: str | None = None,
) -> plt.Figure:
    """
    Enhanced N-row qualitative results grid with seaborn aesthetics.

    Args:
        raw_clean_np: (N, 1, H, W) clean interferograms.
        raw_noisy_np: (N, 1, H, W) noisy interferograms (can be same as clean if not show_noise).
        gt_np: (N, 1, H, W) ground truth unwrapped phase.
        pred_np: (N, 1, H, W) predicted unwrapped phase.
        show_noise: Whether to include the noisy input column.
        filepath: Save destination (handled by @publication_plot).
    """
    n_samples = raw_clean_np.shape[0]
    err_np = np.abs(pred_np - gt_np)
    sample_maes = [err_np[i].mean() for i in range(n_samples)]

    n_cols = 6 if show_noise else 4

    fig, axes = plt.subplots(
        n_samples,
        n_cols,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.22 * n_samples),
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        r"Clean $\mathbf{I}$",
        r"Noisy $\mathbf{I}$",
        r"Ground Truth $\varphi_{\mathrm{GT}}$",
        r"Wrapped $\angle\varphi$",
        r"Prediction $\hat{\varphi}$",
        r"$|\hat{\varphi} - \varphi_{\mathrm{GT}}|$",
    ]

    if not show_noise:
        col_titles = [
            r"Interferogram $\mathbf{I}$",
            r"Ground Truth $\varphi_{\mathrm{GT}}$",
            r"Prediction $\hat{\varphi}$",
            r"$|\hat{\varphi} - \varphi_{\mathrm{GT}}|$",
        ]

    for row in range(n_samples):
        # Column 0: Clean Input
        draw_intensity_panel(
            axes[row, 0], raw_clean_np[row, 0], "" if row > 0 else col_titles[0]
        )

        col_idx = 1
        if show_noise:
            # Column 1: Noisy Input
            draw_intensity_panel(
                axes[row, 1], raw_noisy_np[row, 0], "" if row > 0 else col_titles[1]
            )
            col_idx = 2

        # Column 2: GT Phase
        gt_img = gt_np[row, 0]
        vmin, vmax = gt_img.min(), gt_img.max()
        draw_phase_panel(
            axes[row, col_idx],
            gt_img,
            "" if row > 0 else col_titles[col_idx],
            vmin=vmin,
            vmax=vmax,
        )

        # Column 3: Wrapped Phase
        col_idx += 1
        draw_phase_panel(
            axes[row, col_idx],
            wrap_phase(gt_img),
            "" if row > 0 else col_titles[col_idx],
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )

        # Column 4: Prediction
        col_idx += 1
        draw_phase_panel(
            axes[row, col_idx],
            pred_np[row, 0],
            "" if row > 0 else col_titles[col_idx],
            vmin=vmin,
            vmax=vmax,
        )

        # Column 5: Error
        col_idx += 1
        draw_error_panel(
            axes[row, col_idx],
            err_np[row, 0],
            "" if row > 0 else col_titles[col_idx],
            stats={"mae": sample_maes[row]},
        )

    label_panels(axes.flat)
    fig.tight_layout(pad=0.5)
    
    print(f"  Per-sample MAEs: {[f'{m:.4f}' for m in sample_maes]}")
    return fig
