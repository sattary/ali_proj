"""
Enhanced phase profile with seaborn regression and confidence bands.

1D cross-section through center with regression analysis and residual diagnostics.
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
    DOUBLE_COL,
    create_nature_palette,
    nature_style,
    save_figure,
)


@torch.no_grad()
def plot_phase_profile(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/phase_profile",
    sample_idx: int = 0,
    config_path: str | None = None,
    subset: str = "val",
) -> None:
    """
    Enhanced 1D cross-section with seaborn regression and confidence intervals.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        out_path: Output path (without extension)
        sample_idx: Which sample to plot
        config_path: Optional config override
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    # weights_only=False: loading trusted checkpoint from own training runs
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )
    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = val_loader or train_loader
    else:
        loader = train_loader

    I_input, phi_gt, _, _ = next(iter(loader))
    I_in = I_input[sample_idx : sample_idx + 1].to(device)
    gt = phi_gt[sample_idx : sample_idx + 1].to(device)

    with autocast(device_type=device.type, enabled=False):
        phi_raw, k_off = model(I_in)
        phi_abs = phi_raw + k_off

    aligned, _, _ = affine_align(phi_abs, gt)

    pred_2d = aligned[0, 0].cpu().numpy()
    gt_2d = gt[0, 0].cpu().numpy()
    H, W = pred_2d.shape
    cy, cx = H // 2, W // 2

    # Calculate statistics
    rmse_h = np.sqrt(np.mean((pred_2d[cy, :] - gt_2d[cy, :]) ** 2))
    rmse_v = np.sqrt(np.mean((pred_2d[:, cx] - gt_2d[:, cx]) ** 2))

    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

        x_pix = np.arange(W)
        y_pix = np.arange(H)

        # Horizontal cut - main plot with seaborn
        ax1_main = fig.add_subplot(gs[0, 0])

        # Prepare data for seaborn
        import pandas as pd

        df_h = pd.DataFrame(
            {
                "x": np.tile(x_pix, 2),
                "phase": np.concatenate([gt_2d[cy, :], pred_2d[cy, :]]),
                "type": ["GT"] * W + ["Prediction"] * W,
            }
        )

        # Line plot with confidence
        sns.lineplot(
            data=df_h,
            x="x",
            y="phase",
            hue="type",
            ax=ax1_main,
            palette=[palette[0], palette[1]],
            linewidth=2,
        )

        # Add regression confidence band for prediction
        sns.regplot(
            x=x_pix,
            y=pred_2d[cy, :],
            ax=ax1_main,
            scatter=False,
            color=palette[1],
            line_kws={"linewidth": 0},
            ci=95,
            truncate=True,
        )

        ax1_main.set_ylabel("Phase [rad]", fontsize=9)
        ax1_main.set_title(f"Horizontal Cut (row {cy}, RMSE={rmse_h:.4f})", fontsize=9)
        ax1_main.set_xlabel("")
        ax1_main.legend(fontsize=7)
        ax1_main.grid(True, alpha=0.3)

        # Horizontal residual
        ax1_res = fig.add_subplot(gs[1, 0])
        residual_h = pred_2d[cy, :] - gt_2d[cy, :]

        sns.lineplot(x=x_pix, y=residual_h, ax=ax1_res, color=palette[3], linewidth=1.5)
        ax1_res.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax1_res.fill_between(x_pix, residual_h, 0, alpha=0.3, color=palette[3])

        # Add statistics
        res_mean = np.mean(residual_h)
        res_std = np.std(residual_h)
        ax1_res.text(
            0.02,
            0.95,
            f"μ={res_mean:.3f}, σ={res_std:.3f}",
            transform=ax1_res.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax1_res.set_xlabel("Pixel", fontsize=9)
        ax1_res.set_ylabel("Residual [rad]", fontsize=9)
        ax1_res.grid(True, alpha=0.3)

        # Vertical cut - main plot
        ax2_main = fig.add_subplot(gs[0, 1])

        df_v = pd.DataFrame(
            {
                "y": np.tile(y_pix, 2),
                "phase": np.concatenate([gt_2d[:, cx], pred_2d[:, cx]]),
                "type": ["GT"] * H + ["Prediction"] * H,
            }
        )

        sns.lineplot(
            data=df_v,
            x="y",
            y="phase",
            hue="type",
            ax=ax2_main,
            palette=[palette[0], palette[1]],
            linewidth=2,
        )

        ax2_main.set_ylabel("Phase [rad]", fontsize=9)
        ax2_main.set_title(f"Vertical Cut (col {cx}, RMSE={rmse_v:.4f})", fontsize=9)
        ax2_main.set_xlabel("")
        ax2_main.legend(fontsize=7)
        ax2_main.grid(True, alpha=0.3)

        # Vertical residual
        ax2_res = fig.add_subplot(gs[1, 1])
        residual_v = pred_2d[:, cx] - gt_2d[:, cx]

        sns.lineplot(x=y_pix, y=residual_v, ax=ax2_res, color=palette[3], linewidth=1.5)
        ax2_res.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2_res.fill_between(y_pix, residual_v, 0, alpha=0.3, color=palette[3])

        res_mean_v = np.mean(residual_v)
        res_std_v = np.std(residual_v)
        ax2_res.text(
            0.02,
            0.95,
            f"μ={res_mean_v:.3f}, σ={res_std_v:.3f}",
            transform=ax2_res.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax2_res.set_xlabel("Pixel", fontsize=9)
        ax2_res.set_ylabel("Residual [rad]", fontsize=9)
        ax2_res.grid(True, alpha=0.3)

        plt.suptitle(
            f"Phase Profile Analysis (Sample {sample_idx})", fontsize=11, y=1.02
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved phase profile: {out_path}")
