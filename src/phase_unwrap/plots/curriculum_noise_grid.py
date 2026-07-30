"""
Visualize the curriculum noise progression across epochs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from ..core.config import load_train_config
from ..data.augmentation import NoiseAug, NoiseScheduler
from ..data.dataset import H5ShardDataset, discover_h5_shards
from .style import nature_style, save_figure


def plot_curriculum_noise(
    data_dir: str | None = None,
    out_path: str = "results/figs/curriculum_noise_grid.png",
    sample_idx: int = 0,
    config_path: str | None = None,
) -> None:
    cfg = load_train_config(config_path)
    if data_dir is not None:
        cfg.data.data_dir = data_dir

    paths = discover_h5_shards(cfg.data)
    if not paths:
        raise ValueError(f"No HDF5 data found in {cfg.data.data_dir}")

    with H5ShardDataset(
        paths, I_key=cfg.data.I_key, phi_key=cfg.data.phi_key
    ) as ds:
        I_raw_t, phi_gt_t = ds[sample_idx]

    I_raw = I_raw_t.unsqueeze(0)  # [1, 1, H, W]

    epochs_to_plot = [0, 5, 10, 20, 35]
    n_cols = len(epochs_to_plot)

    aug_cfg = cfg.aug
    noise_aug = NoiseAug(
        gauss_std=aug_cfg.gauss_std,
        speckle_std=aug_cfg.speckle_std,
        poisson_scale=aug_cfg.poisson_scale,
        lowfreq_amp=aug_cfg.lowfreq_amp,
        blur_prob=aug_cfg.blur_prob,
        blur_sigma=(aug_cfg.blur_min, aug_cfg.blur_max),
        dropout_prob=aug_cfg.dropout_prob,
        s_and_p_prob=aug_cfg.sap_prob,
        gain_jitter=(aug_cfg.gain_min, aug_cfg.gain_max),
        offset_jitter=(aug_cfg.off_min, aug_cfg.off_max),
        hint_offset_std=aug_cfg.hint_std,
        enable=True,
    )
    noise_sched = NoiseScheduler(
        warmup_ratio=aug_cfg.warmup_ratio,
        full_ratio=aug_cfg.full_ratio,
        profile=aug_cfg.profile,
    )

    with nature_style():
        fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3.0))
        if n_cols == 1:
            axes = [axes]  # type: ignore

        # Set a fixed seed to visualize how the noise changes purely due to level scaling
        torch.manual_seed(1337)

        for i, epoch in enumerate(epochs_to_plot):
            level = noise_sched.level(epoch)
            noise_aug.set_level(level)

            # Pass the 4D tensor [1, 1, H, W] into __call__ as expected
            img_raw_noisy, img_norm_noisy, _, _ = noise_aug(I_raw, phi_hint=None)

            img_np = img_raw_noisy[0, 0].cpu().numpy()

            ax = axes[i]
            ax.imshow(img_np, cmap="gray", origin="lower", aspect="equal")
            ax.set_title(f"Epoch {epoch}\n(Noise Lv: {level:.2f})", fontsize=10, pad=10)
            ax.axis("off")

        plt.tight_layout()
        save_figure(fig, out_path)
        print(f"Curriculum noise progression saved to {out_path}")
