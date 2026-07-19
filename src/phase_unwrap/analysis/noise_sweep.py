"""
Enhanced noise robustness sweep with seaborn violin plots.

Shows full error distribution at each SNR level, not just mean±std.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

from ..core.ops import piston_align
from ..visualize.style import (
    DOUBLE_COL,
    create_nature_palette,
    nature_style,
    save_figure,
)


def _add_gaussian_noise(interferogram: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to interferogram at a given SNR (dB)."""
    signal_power = np.mean(interferogram**2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*interferogram.shape).astype(np.float32) * np.sqrt(
        noise_power
    )
    return (interferogram + noise).astype(np.float32)


@torch.no_grad()
def noise_robustness_sweep(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/noise_robustness",
    snr_range: tuple[float, float] = (5.0, 40.0),
    n_snr_steps: int = 8,
    n_samples: int = 100,
    seed: int = 42,
    config_path: str | None = None,
    device_str: str = "auto",
) -> dict[str, list[float]]:
    """
    Sweep SNR levels with enhanced seaborn visualization.

    Features:
    - Violin plots at each SNR level showing full distribution
    - Individual sample tracking
    - Statistical trend analysis
    """
    from ..core.inference import load_inference_state

    model, cfg, loader, device = load_inference_state(
        checkpoint_path, data_dir=data_dir, subset="test", config_path=config_path
    )

    # Collect n_samples from the held-out test split (once, reused across SNR levels)
    test_samples: list[tuple[np.ndarray, np.ndarray]] = []
    for I_raw_batch, phi_gt_batch in loader:
        for i in range(I_raw_batch.size(0)):
            if len(test_samples) >= n_samples:
                break
            I_clean = I_raw_batch[i, 0].numpy()
            dphi = phi_gt_batch[i, 0].numpy()
            test_samples.append((I_clean, dphi))
        if len(test_samples) >= n_samples:
            break

    if not test_samples:
        raise ValueError(
            f"Test split is empty. Check data_dir={data_dir!r} and test_frac in config."
        )

    snr_levels = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    snr_eval = list(snr_levels) + [float("inf")]

    # Store all individual MAEs, not just statistics
    all_maes_by_snr: dict[float, list[float]] = {}
    results: dict[str, list[float]] = {"snr_db": [], "mae": [], "std": []}

    for snr_db in snr_eval:
        maes: list[float] = []

        for I_clean, dphi in test_samples:

            I_noisy = (
                I_clean if math.isinf(snr_db) else _add_gaussian_noise(I_clean, snr_db)
            )

            I_mean = I_noisy.mean()
            I_std = I_noisy.std() + 1e-6
            I_norm = (I_noisy - I_mean) / I_std
            # Option B: zero hint matches train/deploy
            phi_hint = np.zeros_like(I_norm, dtype=np.float32)

            inp_t = (
                torch.from_numpy(np.stack([I_norm, phi_hint], 0))
                .unsqueeze(0)
                .to(device)
            )
            gt_t = torch.from_numpy(dphi).unsqueeze(0).unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=False):
                phi_raw, k_off = model(inp_t)
                phi_abs = phi_raw + k_off

            aligned, _ = piston_align(phi_abs, gt_t)
            maes.append(float((aligned - gt_t).abs().mean().item()))

        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        label = "clean" if math.isinf(snr_db) else f"{snr_db:.0f}"
        print(f"  SNR={label:>5s} dB | MAE={mean_mae:.4f} +/- {std_mae:.4f}")

        all_maes_by_snr[snr_db] = maes
        results["snr_db"].append(snr_db)
        results["mae"].append(mean_mae)
        results["std"].append(std_mae)

    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)

        # Panel 1: Violin plot grid
        ax1 = fig.add_subplot(gs[0])

        # Prepare data for seaborn
        plot_data_snr = []
        plot_data_mae = []

        for snr_db, maes in all_maes_by_snr.items():
            label = "Clean" if math.isinf(snr_db) else f"{snr_db:.0f} dB"
            for mae in maes:
                plot_data_snr.append(label)
                plot_data_mae.append(mae)

        import pandas as pd

        df = pd.DataFrame({"SNR": plot_data_snr, "MAE": plot_data_mae})

        # Create violin plot
        sns.violinplot(
            data=df, x="SNR", y="MAE", ax=ax1, palette="Blues", inner="box", linewidth=1
        )

        # Overlay mean points
        sns.pointplot(
            data=df,
            x="SNR",
            y="MAE",
            ax=ax1,
            color="red",
            markers="D",
            scale=0.8,
            linestyles="",
            ci=None,
        )

        ax1.set_xlabel("SNR Level", fontsize=9)
        ax1.set_ylabel("MAE [rad]", fontsize=9)
        ax1.set_title("Error Distribution by SNR", fontsize=10)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Mean trend with confidence
        ax2 = fig.add_subplot(gs[1])

        noisy_mask = [not math.isinf(s) for s in results["snr_db"]]
        snr_noisy = [s for s, m in zip(results["snr_db"], noisy_mask) if m]
        mae_noisy = [m for m, mask in zip(results["mae"], noisy_mask) if mask]
        std_noisy = [s for s, mask in zip(results["std"], noisy_mask) if mask]

        sns.lineplot(
            x=snr_noisy,
            y=mae_noisy,
            ax=ax2,
            color=palette[0],
            marker="o",
            linewidth=2,
            markersize=8,
        )
        ax2.fill_between(
            snr_noisy,
            np.array(mae_noisy) - np.array(std_noisy),
            np.array(mae_noisy) + np.array(std_noisy),
            alpha=0.2,
            color=palette[0],
        )

        # Clean baseline
        if any(not m for m in noisy_mask):
            clean_idx = [i for i, m in enumerate(noisy_mask) if not m][0]
            clean_mae = results["mae"][clean_idx]
            ax2.axhline(
                clean_mae,
                color=palette[2],
                linestyle="--",
                linewidth=2,
                label=f"Clean ({clean_mae:.3f})",
            )

        ax2.set_xlabel("SNR [dB]", fontsize=9)
        ax2.set_ylabel("Mean MAE [rad]", fontsize=9)
        ax2.set_title("Degradation Curve", fontsize=10)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Noise Robustness Analysis", fontsize=11, y=1.02)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved: {out_path}")

    return results