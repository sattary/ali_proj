"""
Noise robustness visualization with seaborn.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .style import DOUBLE_COL, create_nature_palette, nature_style, save_figure


def plot_noise_sweep(
    results: dict[str, list[float]],
    all_maes_by_snr: dict[float, list[float]],
    out_path: str = "results/figs/noise_robustness.png",
) -> None:
    """
    Render SNR levels with enhanced seaborn visualization.
    """
    with nature_style():
        palette = create_nature_palette(6)
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)

        ax1 = fig.add_subplot(gs[0])

        plot_data_snr = []
        plot_data_mae = []

        for snr_db, maes in all_maes_by_snr.items():
            label = "Clean" if math.isinf(snr_db) else f"{snr_db:.0f} dB"
            for mae in maes:
                plot_data_snr.append(label)
                plot_data_mae.append(mae)

        df = pd.DataFrame({"SNR": plot_data_snr, "MAE": plot_data_mae})

        sns.violinplot(
            data=df, x="SNR", y="MAE", ax=ax1, palette="Blues", inner="box", linewidth=1
        )

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

