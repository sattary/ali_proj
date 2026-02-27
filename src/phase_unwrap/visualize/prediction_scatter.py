"""
Prediction vs Ground Truth scatter plot with regression analysis.

Features:
- Hexbin density plot for large datasets
- Regression line with confidence intervals
- Bland-Altman agreement analysis
- Statistical metrics annotation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
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
from .utils import to_numpy


@torch.no_grad()
def plot_prediction_scatter(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/prediction_scatter",
    config_path: str | None = None,
    max_samples: int = 1000,
    subset: str = "val",
    noise_level: float | None = None,
) -> None:
    """
    Create prediction vs GT scatter with hexbin and Bland-Altman.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    if noise_level is not None and noise_level > 0.0:
        cfg.data.augment = True
    else:
        cfg.data.augment = False

    # Standard overrides for CLI/Standalone
    if __name__ == "__main__":
        cfg.data.workers = 0
        cfg.optim.batch_size = 10

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    # Datalaoder
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

    if noise_level is not None and noise_level > 0.0:
        if (
            hasattr(loader.dataset, "noise_aug")
            and loader.dataset.noise_aug is not None
        ):
            loader.dataset.noise_aug.set_level(noise_level)

    # Collect predictions and GT
    all_pred = []
    all_gt = []

    for I_input, phi_gt, *_ in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            if isinstance(phi_raw, list):
                phi_abs = phi_raw[-1] + k_off
            else:
                phi_abs = phi_raw + k_off

        aligned, _, _ = affine_align(phi_abs, phi_gt)

        all_pred.extend(to_numpy(aligned).flatten())
        all_gt.extend(to_numpy(phi_gt).flatten())

        if len(all_pred) >= max_samples * 100:  # Approximate pixels per sample
            break

    pred = np.array(all_pred[: max_samples * 100])
    gt = np.array(all_gt[: max_samples * 100])

    # Compute statistics
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    r2 = stats.pearsonr(pred, gt)[0] ** 2

    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

        # Create grid: 1 row, 3 columns
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        # Panel 1: Hexbin scatter
        ax1 = fig.add_subplot(gs[0])

        ax1.hexbin(gt, pred, gridsize=50, cmap="Blues", mincnt=1, alpha=0.8)

        # Perfect prediction line
        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.5,
            label="Perfect prediction",
        )

        # Regression line
        z = np.polyfit(gt, pred, 1)
        p = np.poly1d(z)
        ax1.plot(
            gt,
            p(gt),
            color=palette[1],
            linewidth=1.5,
            label=f"Regression (R²={r2:.3f})",
        )

        ax1.set_xlabel("Ground Truth [rad]", fontsize=9)
        ax1.set_ylabel("Prediction [rad]", fontsize=9)
        ax1.set_title("Prediction vs GT (Hexbin)", fontsize=10)
        ax1.legend(loc="upper left", fontsize=6)
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        ax1.text(
            0.05,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Panel 2: Joint plot style scatter with regression
        ax2 = fig.add_subplot(gs[1])

        sns.scatterplot(
            x=gt[::100],
            y=pred[::100],  # Subsample for visibility
            ax=ax2,
            alpha=0.3,
            s=10,
            color=palette[0],
        )

        sns.regplot(
            x=gt,
            y=pred,
            ax=ax2,
            scatter=False,
            color=palette[1],
            line_kws={"linewidth": 2},
        )

        ax2.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.7
        )

        ax2.set_xlabel("Ground Truth [rad]", fontsize=9)
        ax2.set_ylabel("Prediction [rad]", fontsize=9)
        ax2.set_title("Regression with CI", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Bland-Altman
        ax3 = fig.add_subplot(gs[2])

        mean_vals = (gt + pred) / 2
        diff = pred - gt

        md = np.mean(diff)
        sd = np.std(diff)

        sns.scatterplot(
            x=mean_vals[::50],
            y=diff[::50],  # Subsample
            ax=ax3,
            alpha=0.4,
            s=8,
            color=palette[0],
        )

        ax3.axhline(
            md,
            color=palette[2],
            linestyle="-",
            linewidth=2,
            label=f"Mean diff: {md:.4f}",
        )
        ax3.axhline(
            md + 1.96 * sd,
            color=palette[3],
            linestyle="--",
            linewidth=1.5,
            label=f"+1.96 SD: {md + 1.96 * sd:.4f}",
        )
        ax3.axhline(
            md - 1.96 * sd,
            color=palette[3],
            linestyle="--",
            linewidth=1.5,
            label=f"-1.96 SD: {md - 1.96 * sd:.4f}",
        )

        ax3.set_xlabel("Mean of GT and Prediction [rad]", fontsize=9)
        ax3.set_ylabel("Difference (Pred - GT) [rad]", fontsize=9)
        ax3.set_title("Bland-Altman Plot", fontsize=10)
        ax3.legend(loc="upper right", fontsize=6)
        ax3.grid(True, alpha=0.3)

        plt.suptitle(
            f"Prediction Accuracy Analysis (n={len(pred):,} pixels)",
            fontsize=11,
            y=1.02,
        )

        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, str(save_path))
        print(f"Saved prediction scatter: {out_path}")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate prediction scatter plot.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/figs/prediction_scatter")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    plot_prediction_scatter(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        out_path=args.out,
        max_samples=args.max_samples,
    )
