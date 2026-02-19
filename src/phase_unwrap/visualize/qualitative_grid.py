"""
Qualitative results grid: Interferogram | GT Phase | Predicted Phase | Error.

Loads a trained checkpoint, runs inference on a data batch, and produces
a publication-ready figure with proper colorbars and LaTeX labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from ..config import TrainConfig, load_train_config
from ..data import build_dataloaders
from ..model import build_model
from ..ops import affine_align
from ..utils import pick_device
from .style import (
    CMAP_ERROR_ABS,
    CMAP_INTENSITY,
    CMAP_CONTINUOUS,
    DOUBLE_COL,
    add_colorbar,
    nature_style,
    save_figure,
)


@torch.no_grad()
def plot_qualitative_grid(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/qualitative_grid.png",
    n_samples: int = 4,
    config_path: str | None = None,
) -> None:
    """
    Generate N-row qualitative results grid.

    Columns: Interferogram | Ground Truth | Prediction | Absolute Error
    """
    # load config from run dir or defaults
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
    model.eval()

    _, val_loader = build_dataloaders(
        cfg.data, cfg.optim, device, seed=cfg.logging.seed
    )
    loader = (
        val_loader
        if val_loader is not None
        else build_dataloaders(cfg.data, cfg.optim, device, seed=cfg.logging.seed)[0]
    )

    I_input, phi_gt, I_raw = next(iter(loader))
    I_input = I_input[:n_samples].to(device)
    phi_gt = phi_gt[:n_samples].to(device)
    I_raw = I_raw[:n_samples]

    with autocast(device_type=device.type, enabled=False):
        phi_raw, k_off = model(I_input)
        phi_abs = phi_raw + k_off

    phi_aligned, _, _ = affine_align(phi_abs, phi_gt)

    pred_np = phi_aligned.cpu().numpy()
    gt_np = phi_gt.cpu().numpy()
    raw_np = I_raw.numpy()
    err_np = np.abs(pred_np - gt_np)

    with nature_style():
        fig, axes = plt.subplots(
            n_samples,
            4,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.25 * n_samples),
        )
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            "Interferogram",
            r"Ground Truth $\varphi_{\mathrm{GT}}$",
            r"Prediction $\hat{\varphi}$",
            r"$|\hat{\varphi} - \varphi_{\mathrm{GT}}|$",
        ]

        for row in range(n_samples):
            I_img = raw_np[row, 0]
            gt_img = gt_np[row, 0]
            pred_img = pred_np[row, 0]
            err_img = err_np[row, 0]

            # shared scale for GT and pred
            vmin_phase = min(gt_img.min(), pred_img.min())
            vmax_phase = max(gt_img.max(), pred_img.max())

            im0 = axes[row, 0].imshow(I_img, cmap=CMAP_INTENSITY, aspect="equal")
            add_colorbar(axes[row, 0], im0)

            im1 = axes[row, 1].imshow(
                gt_img,
                cmap=CMAP_CONTINUOUS,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 1], im1, label="[rad]")

            im2 = axes[row, 2].imshow(
                pred_img,
                cmap=CMAP_CONTINUOUS,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 2], im2, label="[rad]")

            im3 = axes[row, 3].imshow(err_img, cmap=CMAP_ERROR_ABS, aspect="equal")
            add_colorbar(axes[row, 3], im3, label="[rad]")

            for col in range(4):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].spines[:].set_visible(False)

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=7, pad=4)

        fig.tight_layout(pad=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved qualitative grid: {out_path}")
