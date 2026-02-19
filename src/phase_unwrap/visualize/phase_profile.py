"""
Phase profile line-cut plot: 1D cross-section through GT vs prediction.

Shows local accuracy along horizontal and vertical center lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from ..config import load_train_config
from ..data import build_dataloaders
from ..model import build_model
from ..ops import affine_align
from ..utils import pick_device
from .style import DOUBLE_COL, LINE_COLORS, nature_style, save_figure


@torch.no_grad()
def plot_phase_profile(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/phase_profile.png",
    sample_idx: int = 0,
    config_path: str | None = None,
) -> None:
    """
    1D cross-section plot through center row and center column.

    Shows GT and predicted phase overlaid with residual subplot.
    """
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
        or build_dataloaders(cfg.data, cfg.optim, device, seed=cfg.logging.seed)[0]
    )

    I_input, phi_gt, _ = next(iter(loader))
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

    with nature_style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.55),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        x_pix = np.arange(W)
        y_pix = np.arange(H)

        # horizontal cut (center row)
        axes[0, 0].plot(
            x_pix, gt_2d[cy, :], color=LINE_COLORS[0], label=r"$\varphi_{\mathrm{GT}}$"
        )
        axes[0, 0].plot(
            x_pix,
            pred_2d[cy, :],
            color=LINE_COLORS[1],
            linestyle="--",
            label=r"$\hat{\varphi}$",
        )
        axes[0, 0].set_ylabel(r"Phase [rad]")
        axes[0, 0].set_title(f"Horizontal cut (row {cy})", fontsize=7)
        axes[0, 0].legend()

        residual_h = pred_2d[cy, :] - gt_2d[cy, :]
        axes[1, 0].fill_between(x_pix, residual_h, 0, alpha=0.3, color=LINE_COLORS[1])
        axes[1, 0].plot(x_pix, residual_h, color=LINE_COLORS[1], linewidth=0.7)
        axes[1, 0].axhline(0, color="gray", linewidth=0.3)
        axes[1, 0].set_xlabel("Pixel")
        axes[1, 0].set_ylabel("Residual [rad]")

        # vertical cut (center column)
        axes[0, 1].plot(
            y_pix, gt_2d[:, cx], color=LINE_COLORS[0], label=r"$\varphi_{\mathrm{GT}}$"
        )
        axes[0, 1].plot(
            y_pix,
            pred_2d[:, cx],
            color=LINE_COLORS[1],
            linestyle="--",
            label=r"$\hat{\varphi}$",
        )
        axes[0, 1].set_title(f"Vertical cut (col {cx})", fontsize=7)
        axes[0, 1].legend()

        residual_v = pred_2d[:, cx] - gt_2d[:, cx]
        axes[1, 1].fill_between(y_pix, residual_v, 0, alpha=0.3, color=LINE_COLORS[1])
        axes[1, 1].plot(y_pix, residual_v, color=LINE_COLORS[1], linewidth=0.7)
        axes[1, 1].axhline(0, color="gray", linewidth=0.3)
        axes[1, 1].set_xlabel("Pixel")

        fig.tight_layout(pad=0.8)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved phase profile: {out_path}")
