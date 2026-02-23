"""
Enhanced qualitative results grid with seaborn styling.

Publication-ready 6-column grid showing:
- Clean Input | Noisy Input | Ground Truth | Wrapped Phase | Prediction | Error

Includes noise comparison and wrapped phase visualization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..core.config import load_train_config
from ..core.ops import affine_align
from ..core.utils import pick_device
from ..data import build_dataloaders
from ..model import build_model
from .style import (
    CMAP_INTENSITY,
    DOUBLE_COL,
    add_colorbar,
    create_nature_palette,
    nature_style,
    save_figure,
)


def _wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] range."""
    return np.angle(np.exp(1j * phi))


@torch.no_grad()
def plot_qualitative_grid(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/qualitative_grid",
    n_samples: int = 4,
    config_path: str | None = None,
    show_noise: bool = True,
    noise_level: float = 1.0,
) -> None:
    """
    Enhanced N-row qualitative results grid with seaborn aesthetics.

    Columns (6 columns):
        1. Clean Input - Original interferogram without noise
        2. Noisy Input - Interferogram with curriculum noise applied
        3. Ground Truth - Unwrapped phase (target)
        4. Wrapped Phase - Phase wrapped to [-pi, pi] (what classical methods see)
        5. Prediction - Model output
        6. Absolute Error - |prediction - GT|

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        out_path: Output file path
        n_samples: Number of samples to visualize
        config_path: Optional config file path
        show_noise: Whether to show noisy input column
        noise_level: Noise level to apply (0.0 = clean, 1.0 = max)
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    # Apply curriculum noise config
    if cfg.aug.enable:
        cfg.data.augment = True

    # Build dataloaders
    _, val_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = (
        val_loader
        if val_loader is not None
        else build_dataloaders(cfg, device, seed=cfg.logging.seed)[0]
    )

    # Get clean data (no noise)
    cfg_no_noise = cfg.__class__()
    cfg_no_noise.data = cfg.data
    cfg_no_noise.aug = cfg.aug
    cfg_no_noise.aug.enable = False
    _, clean_loader = build_dataloaders(cfg_no_noise, device, seed=cfg.logging.seed)
    clean_data_iter = iter(clean_loader)
    I_clean, phi_gt_clean, I_raw_clean = next(clean_data_iter)
    I_clean = I_clean[:n_samples].to(device)
    phi_gt_clean = phi_gt_clean[:n_samples].to(device)
    I_raw_clean = I_raw_clean[:n_samples]

    with autocast(device_type=device.type, enabled=False):
        phi_raw_clean, k_off_clean = model(I_clean)
        phi_abs_clean = phi_raw_clean + k_off_clean

    phi_aligned_clean, _, _ = affine_align(phi_abs_clean, phi_gt_clean)

    # Get noisy data if requested
    if show_noise and cfg.aug.enable:
        if (
            hasattr(loader.dataset, "noise_aug")
            and loader.dataset.noise_aug is not None
        ):
            loader.dataset.noise_aug.set_level(noise_level)

        I_noisy, phi_gt_noisy, I_raw_noisy = next(iter(loader))
        I_noisy = I_noisy[:n_samples].to(device)
        phi_gt_noisy = phi_gt_noisy[:n_samples].to(device)
        I_raw_noisy = I_raw_noisy[:n_samples]

        with autocast(device_type=device.type, enabled=False):
            phi_raw_noisy, k_off_noisy = model(I_noisy)
            phi_abs_noisy = phi_raw_noisy + k_off_noisy

        phi_aligned_noisy, _, _ = affine_align(phi_abs_noisy, phi_gt_noisy)
    else:
        I_noisy = I_clean
        phi_aligned_noisy = phi_aligned_clean
        I_raw_noisy = I_raw_clean
        phi_gt_noisy = phi_gt_clean

    # Convert to numpy
    pred_np = phi_aligned_clean.cpu().numpy()
    gt_np = phi_gt_clean.cpu().numpy()
    raw_clean_np = I_raw_clean.numpy()
    raw_noisy_np = I_raw_noisy.numpy()
    err_np = np.abs(pred_np - gt_np)

    # Calculate per-sample MAE
    sample_maes = [err_np[i].mean() for i in range(n_samples)]

    # Determine number of columns
    n_cols = 6 if show_noise else 4

    with nature_style():
        palette = create_nature_palette(6)
        fig, axes = plt.subplots(
            n_samples,
            n_cols,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.22 * n_samples),
        )
        if n_samples == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

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
            ax = axes[row, 0]
            im0 = ax.imshow(raw_clean_np[row, 0], cmap=CMAP_INTENSITY, aspect="equal")
            add_colorbar(ax, im0)

            # Column 1: Noisy Input (if enabled)
            if show_noise:
                ax = axes[row, 1]
                im1 = ax.imshow(
                    raw_noisy_np[row, 0], cmap=CMAP_INTENSITY, aspect="equal"
                )
                add_colorbar(ax, im1)
                col_idx = 2
            else:
                col_idx = 1

            # Column 2: GT Phase
            gt_img = gt_np[row, 0]
            ax = axes[row, col_idx]
            vmin_phase = gt_img.min()
            vmax_phase = gt_img.max()
            phase_cmap = sns.color_palette("viridis", as_cmap=True)
            im2 = ax.imshow(
                gt_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(ax, im2, label="[rad]")

            # Column 3: Wrapped Phase
            col_idx += 1
            ax = axes[row, col_idx]
            wrapped = _wrap_phase(gt_img)
            im_wrap = ax.imshow(
                wrapped, cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi
            )
            add_colorbar(ax, im_wrap, label="[rad]")

            # Column 4: Prediction
            col_idx += 1
            pred_img = pred_np[row, 0]
            ax = axes[row, col_idx]
            im4 = ax.imshow(
                pred_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(ax, im4, label="[rad]")

            # Column 5: Error
            col_idx += 1
            err_img = err_np[row, 0]
            ax = axes[row, col_idx]
            error_cmap = sns.color_palette("rocket", as_cmap=True)
            im5 = ax.imshow(err_img, cmap=error_cmap, aspect="equal")
            add_colorbar(ax, im5, label="[rad]")

            # Add MAE annotation on error column
            ax.text(
                0.98,
                0.98,
                f"MAE: {sample_maes[row]:.4f}",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
                color="white" if sample_maes[row] > err_img.mean() else "black",
                bbox=dict(boxstyle="round", facecolor=palette[0], alpha=0.8),
            )

            # Clean up axes
            for col in range(n_cols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].spines[:].set_visible(False)

        # Add column titles
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=8, pad=5)

        plt.suptitle(
            f"Qualitative Results (Mean MAE: {np.mean(sample_maes):.4f} rad)"
            + (f" | Noise Level: {noise_level:.1f}" if show_noise else ""),
            fontsize=10,
            y=0.98,
        )

        fig.tight_layout(pad=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved qualitative grid: {out_path}")
        print(f"  Per-sample MAEs: {[f'{m:.4f}' for m in sample_maes]}")
