"""
Noise Degradation Grid evaluating structural robustness and failure limits.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..core.ops import affine_align
from .style import (
    CMAP_ERROR_ABS,
    CMAP_INTENSITY,
    DOUBLE_COL,
    add_colorbar,
    create_nature_palette,
    nature_style,
    save_figure,
)


def add_awgn(signal: torch.Tensor, snr_db: float, seed: int = 42) -> torch.Tensor:
    """Add Additive White Gaussian Noise to a tensor at target SNR."""
    if math.isinf(snr_db):
        return signal.clone()

    # Calculate signal power
    sig_power = torch.var(signal)

    # Calculate required noise power
    noise_power = sig_power / (10 ** (snr_db / 10))

    generator = torch.Generator(device=signal.device)
    generator.manual_seed(seed)

    noise = torch.randn(
        signal.shape, generator=generator, device=signal.device
    ) * math.sqrt(noise_power)
    return signal + noise


@torch.no_grad()
def plot_noise_degradation(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/noise_degradation",
    sample_idx: int = 0,
    config_path: str | None = None,
    subset: str = "val",
) -> None:
    """
    Nature-style iterative noise degradation evaluation.
    Rows: SNR levels (inf, 40dB, 20dB, 10dB, 5dB).
    Columns: Degraded Interferogram | Predicted Phase | Absolute Error Map.
    """
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )

    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = (
            val_loader
            if (val_loader is not None and len(val_loader) > 0)
            else train_loader
        )
    else:
        loader = train_loader

    dataset = loader.dataset

    # Get single target sample
    I_input, phi_gt, I_raw = dataset[sample_idx]

    # Move to batch format
    phi_gt = phi_gt.unsqueeze(0).to(device)
    I_raw = I_raw.unsqueeze(0).to(device)

    snr_levels = [float("inf"), 40.0, 20.0, 10.0, 5.0]
    n_rows = len(snr_levels)

    with nature_style():
        palette = create_nature_palette()
        phase_cmap = sns.color_palette("viridis", as_cmap=True)

        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.35 * n_rows),
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            "Degraded Interferogram",
            r"Prediction $\hat{\varphi}$",
            r"$|\hat{\varphi} - \varphi_{\mathrm{GT}}|$",
        ]

        vmin_phase = phi_gt.cpu().numpy().min()
        vmax_phase = phi_gt.cpu().numpy().max()

        for idx, snr in enumerate(snr_levels):
            # Degrade raw interferogram
            noisy_raw = add_awgn(I_raw, snr, seed=42 + idx)

            # Reconstruct model input
            mean = noisy_raw.mean(dim=(1, 2, 3), keepdim=True)
            std = noisy_raw.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            noisy_norm = (noisy_raw - mean) / std

            # Extract central hint value from GT
            _, _, H, W = phi_gt.shape
            cy, cx = H // 2, W // 2
            ref_val = float(phi_gt[0, 0, cy, cx].item())
            phi_hint = torch.full_like(noisy_norm, ref_val)

            noisy_input = torch.cat([noisy_norm, phi_hint], dim=1)

            with autocast(device_type=device.type, enabled=False):
                phi_raw_pred, k_off_pred = model(noisy_input)
                if isinstance(phi_raw_pred, list):
                    phi_abs_pred = phi_raw_pred[-1] + k_off_pred
                else:
                    phi_abs_pred = phi_raw_pred + k_off_pred

            phi_aligned, _, _ = affine_align(phi_abs_pred, phi_gt)

            pred_np = phi_aligned.cpu().numpy()[0, 0]
            err_np = np.abs(pred_np - phi_gt.cpu().numpy()[0, 0])
            I_img = noisy_raw.cpu().numpy()[0, 0]

            im0 = axes[idx, 0].imshow(I_img, cmap=CMAP_INTENSITY, aspect="equal")
            add_colorbar(axes[idx, 0], im0)

            label_snr = f"SNR: {snr}dB" if not math.isinf(snr) else r"SNR: $\infty$"
            axes[idx, 0].text(
                0.02,
                0.98,
                label_snr,
                transform=axes[idx, 0].transAxes,
                fontsize=8,
                ha="left",
                va="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )

            im1 = axes[idx, 1].imshow(
                pred_np,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[idx, 1], im1, label="[rad]")

            im2 = axes[idx, 2].imshow(err_np, cmap=CMAP_ERROR_ABS, aspect="equal")
            add_colorbar(axes[idx, 2], im2, label="[rad]")

            # Annotate MAEs
            mae = err_np.mean()
            axes[idx, 2].text(
                0.98,
                0.98,
                f"MAE: {mae:.4f}",
                transform=axes[idx, 2].transAxes,
                fontsize=7,
                ha="right",
                va="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor=palette[1], alpha=0.8),
            )

            for col in range(3):
                axes[idx, col].set_xticks([])
                axes[idx, col].set_yticks([])
                axes[idx, col].spines[:].set_visible(False)

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=9, pad=8)

        fig.tight_layout(pad=1.0)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved noise degradation grid: {out_path}")
