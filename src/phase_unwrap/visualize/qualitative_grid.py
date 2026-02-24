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
import torch
from torch.amp import autocast

from ..core.config import load_train_config
from ..core.ops import affine_align
from ..core.utils import pick_device
from ..data import build_dataloaders
from ..model import build_model
from .style import (
    DOUBLE_COL,
    nature_style,
    save_figure,
)
from .utils import (
    to_numpy,
    wrap_phase,
    draw_intensity_panel,
    draw_phase_panel,
    draw_error_panel,
)


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
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.workers = 0
    cfg.optim.batch_size = n_samples

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    else:
        # Probable TorchScript module
        print(f"[load] Checkpoint is {type(ckpt)}. Attempting state_dict extraction...")
        model.load_state_dict(ckpt.state_dict())
    model.eval()

    # Apply curriculum noise config
    if cfg.aug.enable:
        cfg.data.augment = True

    # Build dataloaders
    train_loader, _ = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = train_loader

    # Get clean data (no noise)
    cfg_no_noise = cfg.__class__()
    for k, v in cfg.__dict__.items():
        setattr(cfg_no_noise, k, v)
    cfg_no_noise.aug.enable = False
    clean_loader, _ = build_dataloaders(cfg_no_noise, device, seed=cfg.logging.seed)
    clean_data_iter = iter(clean_loader)
    I_clean, phi_gt_clean, I_raw_n_clean, I_raw_c_clean = next(clean_data_iter)
    I_clean = I_clean[:n_samples].to(device)
    phi_gt_clean = phi_gt_clean[:n_samples].to(device)
    I_raw_clean = I_raw_c_clean[:n_samples]

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

        I_noisy, phi_gt_noisy, I_raw_n_noisy, I_raw_c_noisy = next(iter(loader))

        with autocast(device_type=device.type, enabled=False):
            phi_raw_noisy, k_off_noisy = model(I_noisy)
            phi_abs_noisy = phi_raw_noisy + k_off_noisy

        phi_aligned_noisy, _, _ = affine_align(phi_abs_noisy, phi_gt_noisy)
    else:
        I_noisy = I_clean
        I_raw_noisy = I_raw_clean
        phi_gt_noisy = phi_gt_clean

    # Convert to numpy
    pred_np = to_numpy(phi_aligned_clean)
    gt_np = to_numpy(phi_gt_clean)
    raw_clean_np = I_raw_clean.numpy()
    raw_noisy_np = I_raw_noisy.numpy()
    err_np = np.abs(pred_np - gt_np)

    # Calculate per-sample MAE
    sample_maes = [err_np[i].mean() for i in range(n_samples)]

    # Determine number of columns
    n_cols = 6 if show_noise else 4

    with nature_style():
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

        fig.tight_layout(pad=0.5)
        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, str(save_path))
        print(f"Saved qualitative grid: {out_path}")
        print(f"  Per-sample MAEs: {[f'{m:.4f}' for m in sample_maes]}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate qualitative grid.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/figs/qual_grid")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--noise_level", type=float, default=1.0)
    parser.add_argument("--show-noise", action="store_true", default=True)
    parser.add_argument("--no-show-noise", action="store_false", dest="show_noise")
    args = parser.parse_args()

    plot_qualitative_grid(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        out_path=args.out,
        n_samples=args.n_samples,
        noise_level=args.noise_level,
        show_noise=args.show_noise,
    )
