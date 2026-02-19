"""
Noise robustness sweep: evaluate a trained model across SNR levels.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from ..core.config import load_train_config
from ..core.ops import affine_align
from ..core.utils import pick_device
from ..data.generate import _build_grid, generate_sample
from ..model import build_model
from ..visualize.style import LINE_COLORS, SINGLE_COL, nature_style, save_figure


def _add_gaussian_noise(I: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to interferogram at a given SNR (dB)."""
    signal_power = np.mean(I**2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*I.shape).astype(np.float32) * np.sqrt(noise_power)
    return (I + noise).astype(np.float32)


@torch.no_grad()
def noise_robustness_sweep(
    checkpoint_path: str,
    out_path: str = "results/figs/noise_robustness.png",
    snr_range: tuple[float, float] = (5.0, 40.0),
    n_snr_steps: int = 8,
    n_samples: int = 100,
    seed: int = 42,
    config_path: str | None = None,
    device_str: str = "auto",
) -> dict[str, list[float]]:
    """Sweep SNR levels and record MAE at each, then plot."""
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)

    device = pick_device(device_str)
    model = build_model(cfg.model).to(device)
    # weights_only=False: loading trusted checkpoint from own training runs
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
    model.eval()

    x, y, r2 = _build_grid()

    snr_levels = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    results: dict[str, list[float]] = {"snr_db": [], "mae": [], "std": []}
    snr_eval = list(snr_levels) + [float("inf")]

    for snr_db in snr_eval:
        maes: list[float] = []
        test_rng = np.random.default_rng(seed)

        for _ in range(n_samples):
            I_clean, dphi = generate_sample(x, y, r2, test_rng)

            I_noisy = (
                I_clean if math.isinf(snr_db) else _add_gaussian_noise(I_clean, snr_db)
            )

            I_mean = I_noisy.mean()
            I_std = I_noisy.std() + 1e-6
            I_norm = (I_noisy - I_mean) / I_std
            phi_hint = np.angle(np.exp(1j * I_noisy))

            inp_t = (
                torch.from_numpy(np.stack([I_norm, phi_hint], 0))
                .unsqueeze(0)
                .to(device)
            )
            gt_t = torch.from_numpy(dphi).unsqueeze(0).unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=False):
                phi_raw, k_off = model(inp_t)
                phi_abs = phi_raw + k_off

            aligned, _, _ = affine_align(phi_abs, gt_t)
            maes.append(float((aligned - gt_t).abs().mean().item()))

        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        label = "clean" if math.isinf(snr_db) else f"{snr_db:.0f}"
        print(f"  SNR={label:>5s} dB | MAE={mean_mae:.4f} +/- {std_mae:.4f}")

        results["snr_db"].append(snr_db)
        results["mae"].append(mean_mae)
        results["std"].append(std_mae)

    with nature_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

        noisy_mask = [not math.isinf(s) for s in results["snr_db"]]
        clean_mask = [math.isinf(s) for s in results["snr_db"]]

        snr_noisy = [s for s, m in zip(results["snr_db"], noisy_mask) if m]
        mae_noisy = [m for m, mask in zip(results["mae"], noisy_mask) if mask]
        std_noisy = [s for s, mask in zip(results["std"], noisy_mask) if mask]

        ax.errorbar(
            snr_noisy,
            mae_noisy,
            yerr=std_noisy,
            color=LINE_COLORS[0],
            marker="o",
            capsize=3,
            linewidth=1.0,
            markersize=4,
            label="Noisy",
        )

        if any(clean_mask):
            clean_mae = [m for m, mask in zip(results["mae"], clean_mask) if mask][0]
            ax.axhline(
                clean_mae,
                color=LINE_COLORS[2],
                linestyle="--",
                linewidth=0.8,
                label=f"Clean (MAE={clean_mae:.3f})",
            )

        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("MAE [rad]")
        ax.set_title("Noise Robustness")
        ax.legend(fontsize=6)
        fig.tight_layout(pad=0.5)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved: {out_path}")

    return results
