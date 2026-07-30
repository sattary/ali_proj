"""
Figure 6: Physics Vector Curl & Zernike Spectral Modal Decomposition.

Publication-ready physics diagnostic figure:
- Panel A: Wrapped Curl Density Map |nabla x g| (singularity detection)
- Panel B: Corrected Curl Density Map |nabla x g_tilde| (PCLCN integrability restoration)
- Panel C: Zernike Modal Spectrum (c_1 ... c_15 GT vs PCLCN predicted coefficients)
- Panel D: Pupil Radial Error Profile epsilon(r) showing disk-masked boundary performance
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    DOUBLE_COL,
    add_colorbar,
    create_nature_palette,
    publication_plot,
    label_panels,
)


@publication_plot
def plot_f6_physics_zernike(
    curl_raw_np: np.ndarray,
    curl_corr_np: np.ndarray,
    c_gt_np: np.ndarray,
    c_pred_np: np.ndarray,
    gt_2d_np: np.ndarray,
    pred_2d_np: np.ndarray,
    filepath: str | None = None,
) -> plt.Figure:
    """
    Stateless 4-panel physics vector curl and Zernike modal spectrum renderer.

    Args:
        curl_raw_np: (H, W) or (1, 1, H, W) raw wrapped curl magnitude.
        curl_corr_np: (H, W) or (1, 1, H, W) PCLCN corrected curl magnitude.
        c_gt_np: (15,) ground-truth Zernike coefficients.
        c_pred_np: (15,) predicted Zernike coefficients.
        gt_2d_np: (H, W) ground-truth phase map.
        pred_2d_np: (H, W) predicted phase map.
    """
    if curl_raw_np.ndim == 4:
        curl_raw = curl_raw_np[0, 0]
        curl_corr = curl_corr_np[0, 0]
        gt_2d = gt_2d_np[0, 0]
        pred_2d = pred_2d_np[0, 0]
    else:
        curl_raw = curl_raw_np
        curl_corr = curl_corr_np
        gt_2d = gt_2d_np
        pred_2d = pred_2d_np

    if c_gt_np.ndim == 2:
        c_gt = c_gt_np[0]
        c_pred = c_pred_np[0]
    else:
        c_gt = c_gt_np
        c_pred = c_pred_np

    palette = create_nature_palette(6)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.8))

    # Panel A: Wrapped Curl Map
    ax_curl1 = axes[0, 0]
    im1 = ax_curl1.imshow(curl_raw, cmap="magma", aspect="equal")
    add_colorbar(ax_curl1, im1, label="Density")
    ax_curl1.set_title(r"Wrapped Curl Density $|\nabla \times \mathbf{g}|$", fontsize=9)
    ax_curl1.axis("off")

    # Panel B: Corrected Curl Map
    ax_curl2 = axes[0, 1]
    im2 = ax_curl2.imshow(curl_corr, cmap="magma", aspect="equal", vmin=0, vmax=max(curl_raw.max(), 1e-3))
    add_colorbar(ax_curl2, im2, label="Density")
    ax_curl2.set_title(r"PCLCN Corrected Curl $|\nabla \times \tilde{\mathbf{g}}|$", fontsize=9)
    ax_curl2.axis("off")

    # Panel C: Zernike Spectrum Comparison
    ax_zern = axes[1, 0]
    modes = np.arange(1, len(c_gt) + 1)
    width = 0.35
    ax_zern.bar(modes - width / 2, c_gt, width, label="GT", color=palette[0], alpha=0.8)
    ax_zern.bar(modes + width / 2, c_pred, width, label="PCLCN", color=palette[1], alpha=0.8)
    ax_zern.set_xlabel("Zernike Mode Index $j$", fontsize=8)
    ax_zern.set_ylabel(r"Amplitude $c_j$", fontsize=8)
    ax_zern.set_title("Zernike Modal Spectrum ($j=1..15$)", fontsize=9)
    ax_zern.set_xticks(modes)
    ax_zern.legend(fontsize=7)
    ax_zern.grid(True, alpha=0.3, axis="y")

    # Panel D: Radial Error Profile epsilon(r)
    ax_rad = axes[1, 1]
    H, W = gt_2d.shape
    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.0, 1.0, W)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r_grid = np.sqrt(xx**2 + yy**2)
    err_2d = np.abs(pred_2d - gt_2d)

    # Bin error by radial distance
    r_bins = np.linspace(0, 1.0, 20)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    mean_err_r = []
    std_err_r = []

    for i in range(len(r_bins) - 1):
        mask_bin = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
        if np.any(mask_bin):
            mean_err_r.append(err_2d[mask_bin].mean())
            std_err_r.append(err_2d[mask_bin].std())
        else:
            mean_err_r.append(0.0)
            std_err_r.append(0.0)

    mean_err_r = np.array(mean_err_r)
    std_err_r = np.array(std_err_r)

    ax_rad.plot(r_centers, mean_err_r, "o-", color=palette[0], linewidth=1.5, label=r"Radial MAE $\epsilon(r)$")
    ax_rad.fill_between(r_centers, mean_err_r - std_err_r, mean_err_r + std_err_r, alpha=0.2, color=palette[0])
    ax_rad.axvline(1.0, color="red", linestyle="--", linewidth=1.0, label="Pupil Boundary ($r=1.0$)")
    ax_rad.set_xlabel(r"Normalized Pupil Radius $r = \rho / R$", fontsize=8)
    ax_rad.set_ylabel("MAE [rad]", fontsize=8)
    ax_rad.set_title("Pupil Radial Error Distribution", fontsize=9)
    ax_rad.legend(fontsize=7)
    ax_rad.grid(True, alpha=0.3)

    label_panels(axes.flat)
    fig.tight_layout(pad=0.5)
    return fig
