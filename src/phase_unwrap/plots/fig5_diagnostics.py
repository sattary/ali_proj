"""
Figure 5: Model Fidelity & Residual Error Diagnostics.

Publication-ready 4-panel figure:
- Panel A: Scatter plot (GT vs Predicted Phase with R^2 regression line)
- Panel B: Spatial Residual Map r(x, y) = phi_pred - phi_gt
- Panel C: Error Residual Histogram with Gaussian distribution fit (mu, sigma)
- Panel D: 1D Center Line Cut Profile (GT vs Prediction across row H/2)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

from .style import (
    DOUBLE_COL,
    add_colorbar,
    create_nature_palette,
    publication_plot,
    label_panels,
)


@publication_plot
def plot_f5_diagnostics(
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    filepath: str | None = None,
) -> plt.Figure:
    """
    Stateless 4-panel model fidelity and residual diagnostic renderer.

    Args:
        gt_np: (1, 1, H, W) or (H, W) ground truth phase.
        pred_np: (1, 1, H, W) or (H, W) predicted unwrapped phase.
    """
    if gt_np.ndim == 4:
        gt_2d = gt_np[0, 0]
        pred_2d = pred_np[0, 0]
    else:
        gt_2d = gt_np
        pred_2d = pred_np

    residual = pred_2d - gt_2d
    H, W = gt_2d.shape
    center_row = H // 2

    # R^2 correlation calculation
    gt_flat = gt_2d.flatten()
    pred_flat = pred_2d.flatten()
    corr_matrix = np.corrcoef(gt_flat, pred_flat)
    r_squared = corr_matrix[0, 1] ** 2

    palette = create_nature_palette(6)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.8))

    # Panel A: Scatter plot with regression line
    ax_scatter = axes[0, 0]
    sample_sub = np.random.choice(len(gt_flat), size=min(2000, len(gt_flat)), replace=False)
    ax_scatter.scatter(gt_flat[sample_sub], pred_flat[sample_sub], alpha=0.3, s=5, color=palette[0])
    
    lims = [min(gt_flat.min(), pred_flat.min()), max(gt_flat.max(), pred_flat.max())]
    ax_scatter.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = x)")
    ax_scatter.set_xlabel(r"Ground Truth $\varphi_{\mathrm{GT}}$ [rad]", fontsize=8)
    ax_scatter.set_ylabel(r"Prediction $\hat{\varphi}$ [rad]", fontsize=8)
    ax_scatter.set_title(f"Phase Correlation ($R^2 = {r_squared:.4f}$)", fontsize=9)
    ax_scatter.legend(fontsize=7)
    ax_scatter.grid(True, alpha=0.3)

    # Panel B: Spatial Residual Map
    ax_res = axes[0, 1]
    vmax_res = max(abs(residual.min()), abs(residual.max()))
    im_res = ax_res.imshow(residual, cmap="seismic", aspect="equal", vmin=-vmax_res, vmax=vmax_res)
    add_colorbar(ax_res, im_res, label="[rad]")
    ax_res.set_title(r"Spatial Residual $\hat{\varphi} - \varphi_{\mathrm{GT}}$", fontsize=9)
    ax_res.axis("off")

    # Panel C: Error Residual Histogram
    ax_hist = axes[1, 0]
    res_flat = residual.flatten()
    mu, std = norm.fit(res_flat)
    
    sns.histplot(res_flat, kde=True, ax=ax_hist, color=palette[0], stat="density", bins=40)
    x_axis = np.linspace(res_flat.min(), res_flat.max(), 100)
    ax_hist.plot(x_axis, norm.pdf(x_axis, mu, std), "r-", linewidth=1.5, label=f"Fit (μ={mu:.3f}, σ={std:.3f})")
    ax_hist.set_xlabel("Residual Error [rad]", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_title("Residual Error Distribution", fontsize=9)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, alpha=0.3)

    # Panel D: 1D Line Cut Profile
    ax_cut = axes[1, 1]
    x_range = np.arange(W)
    ax_cut.plot(x_range, gt_2d[center_row, :], "b-", linewidth=1.5, label=r"GT $\varphi$")
    ax_cut.plot(x_range, pred_2d[center_row, :], "r--", linewidth=1.5, label=r"Pred $\hat{\varphi}$")
    ax_cut.set_xlabel("Pixel Index (x)", fontsize=8)
    ax_cut.set_ylabel("Phase [rad]", fontsize=8)
    ax_cut.set_title(f"1D Center Line Cut (Row {center_row})", fontsize=9)
    ax_cut.legend(fontsize=7)
    ax_cut.grid(True, alpha=0.3)

    label_panels(axes.flat)
    fig.tight_layout(pad=0.5)
    return fig
