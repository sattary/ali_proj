"""
Visual Baseline Superiority Matrix comparing classical vs Deep Learning models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..analysis.baselines import _unwrap_itoh, _unwrap_skimage
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


@torch.no_grad()
def plot_baseline_comparison(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/baseline_comparison",
    n_samples: int = 4,
    config_path: str | None = None,
) -> None:
    """
    Nature-style multiclass visual comparison.
    Columns: Interferogram | Ground Truth | Itoh 1D | Least-Squares | UNetRes2
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False
    cfg.data.workers = 0
    cfg.optim.batch_size = n_samples

    device = pick_device("cpu")  # Strict CPU extraction logic
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    else:
        model.load_state_dict(ckpt.state_dict())
    model.eval()

    train_loader, val_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = (
        val_loader if (val_loader is not None and len(val_loader) > 0) else train_loader
    )

    I_input, phi_gt, I_raw_n, I_raw_c = next(iter(loader))
    I_input = I_input[:n_samples].to(device)
    phi_gt = phi_gt[:n_samples].to(device)
    I_raw = I_raw_c[:n_samples].numpy()
    gt_np = phi_gt.cpu().numpy()

    with autocast(device_type=device.type, enabled=False):
        phi_raw, k_off = model(I_input)
        phi_abs = phi_raw + k_off

    phi_aligned, _, _ = affine_align(phi_abs, phi_gt)
    unet_np = phi_aligned.cpu().numpy()

    with nature_style():
        palette = create_nature_palette()
        phase_cmap = sns.color_palette("viridis", as_cmap=True)

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
            "UNetRes2 (Ours)",
        ]

        for row in range(n_samples):
            I_img = I_raw[row, 0]
            gt_img = gt_np[row, 0]
            unet_img = unet_np[row, 0]

            # Classical Baselines
            try:
                itoh_pred = _unwrap_itoh(I_img)
                itoh_t = torch.from_numpy(itoh_pred).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)
                itoh_aligned, _, _ = affine_align(itoh_t, gt_t)
                itoh_img = itoh_aligned.numpy()[0, 0]
            except Exception:
                itoh_img = np.zeros_like(gt_img)

            try:
                lsq_pred = _unwrap_skimage(I_img)
                lsq_t = torch.from_numpy(lsq_pred).unsqueeze(0).unsqueeze(0)
                lsq_aligned, _, _ = affine_align(lsq_t, gt_t)
                lsq_img = lsq_aligned.numpy()[0, 0]
            except Exception:
                lsq_img = np.zeros_like(gt_img)

            vmin_phase = gt_img.min()
            vmax_phase = gt_img.max()

            im0 = axes[row, 0].imshow(I_img, cmap=CMAP_INTENSITY, aspect="equal")
            add_colorbar(axes[row, 0], im0)

            im1 = axes[row, 1].imshow(
                gt_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 1], im1, label="[rad]")

            im2 = axes[row, 2].imshow(
                itoh_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 2], im2, label="[rad]")

            im3 = axes[row, 3].imshow(
                lsq_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 3], im3, label="[rad]")

            im4 = axes[row, 4].imshow(
                unet_img,
                cmap=phase_cmap,
                aspect="equal",
                vmin=vmin_phase,
                vmax=vmax_phase,
            )
            add_colorbar(axes[row, 4], im4, label="[rad]")

            for col in range(5):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].spines[:].set_visible(False)

            # Annotate MAEs for the models
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

        fig.tight_layout(pad=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved baseline comparison grid: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline comparison grid.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/figs/baseline_comparison")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    plot_baseline_comparison(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        out_path=args.out,
        n_samples=args.n_samples,
    )
