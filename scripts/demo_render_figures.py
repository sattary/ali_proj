"""
Demonstration script to render all 6 Optics Express paper figures using synthetic inputs.
Exports high-resolution figures to results/paper_figures/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase_unwrap.plots import (
    plot_f1_architecture,
    plot_f2_baseline_comparison,
    plot_f3_noise_grid,
    plot_f3_noise_sweep,
    plot_f4_multiseed,
    plot_f4_radar,
    plot_f5_diagnostics,
    plot_f6_physics_zernike,
)


def render_all_figures(out_dir: str = "results/paper_figures") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    H, W = 128, 128
    y = np.linspace(-1, 1, H).reshape(-1, 1)
    x = np.linspace(-1, 1, W).reshape(1, -1)
    
    # Synthetic ground-truth phase (spherical + astigmatism)
    gt_phase = 5.0 * (x**2 + y**2) + 2.0 * (x**2 - y**2)
    gt_np = gt_phase[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
    
    # Synthetic interferogram
    clean_i = np.cos(gt_phase + 0.125 * np.pi * (x + y))
    clean_i_np = clean_i[np.newaxis, np.newaxis, ...]
    
    # Synthetic noisy interferogram
    noisy_i_np = clean_i_np + 0.05 * np.random.randn(*clean_i_np.shape)

    # Synthetic PCLCN prediction with minor residual error
    pred_np = gt_np + 0.02 * np.random.randn(*gt_np.shape)

    # 1. Figure 1: Architecture & Inference Grid
    print("Rendering Figure 1 (Architecture Grid)...")
    plot_f1_architecture(
        raw_clean_np=clean_i_np,
        raw_noisy_np=noisy_i_np,
        gt_np=gt_np,
        pred_np=pred_np,
        show_noise=True,
        filepath=str(out_path / "fig1_architecture.png"),
    )

    # 2. Figure 2: Baseline Comparison Matrix
    print("Rendering Figure 2 (Baseline Comparison Matrix)...")
    itoh_np = gt_np + 0.15 * np.random.randn(*gt_np.shape)
    lsq_np = gt_np + 0.08 * np.random.randn(*gt_np.shape)
    plot_f2_baseline_comparison(
        raw_i_np=clean_i_np,
        gt_np=gt_np,
        itoh_np=itoh_np,
        lsq_np=lsq_np,
        unet_np=pred_np,
        filepath=str(out_path / "fig2_baseline_comparison.png"),
    )

    # 3. Figure 3: Noise Robustness Analysis
    print("Rendering Figure 3 (Noise Robustness Analysis)...")
    results = {
        "snr_db": [float("inf"), 30.0, 20.0, 10.0, 5.0],
        "mae": [0.015, 0.022, 0.035, 0.068, 0.125],
        "std": [0.002, 0.004, 0.008, 0.015, 0.030],
    }
    all_maes = {
        float("inf"): [0.015 + np.random.normal(0, 0.002) for _ in range(50)],
        30.0: [0.022 + np.random.normal(0, 0.004) for _ in range(50)],
        20.0: [0.035 + np.random.normal(0, 0.008) for _ in range(50)],
        10.0: [0.068 + np.random.normal(0, 0.015) for _ in range(50)],
        5.0: [0.125 + np.random.normal(0, 0.030) for _ in range(50)],
    }
    plot_f3_noise_sweep(results, all_maes, filepath=str(out_path / "fig3_noise_robustness.png"))

    # 4. Figure 4: Ablation Study Boxen & Radar Plots
    print("Rendering Figure 4 (PCLCN Ablation Matrix)...")
    ablation_results = {
        "PCLCN (Full)": {"val_mae": 0.020, "val_rmse": 0.028, "w_curl": 0.005},
        "no_ref_prior": {"val_mae": 0.052, "val_rmse": 0.071, "w_curl": 0.018},
        "no_curl_loss": {"val_mae": 0.045, "val_rmse": 0.062, "w_curl": 0.042},
        "unmasked_zern": {"val_mae": 0.068, "val_rmse": 0.089, "w_curl": 0.025},
    }
    plot_f4_radar(ablation_results, baseline_name="PCLCN (Full)", filepath=str(out_path / "fig4_ablation.png"))

    # 5. Figure 5: Model Fidelity & Residual Error Diagnostics
    print("Rendering Figure 5 (Model Fidelity & Residual Diagnostics)...")
    plot_f5_diagnostics(gt_np=gt_np, pred_np=pred_np, filepath=str(out_path / "fig5_diagnostics.png"))

    # 6. Figure 6: Physics Vector Curl & Zernike Spectrum
    print("Rendering Figure 6 (Physics Vector Curl & Zernike Spectrum)...")
    curl_raw = np.abs(np.random.randn(H, W)) * 0.5
    curl_corr = np.abs(np.random.randn(H, W)) * 0.02
    c_gt = np.random.randn(15)
    c_pred = c_gt + 0.05 * np.random.randn(15)
    plot_f6_physics_zernike(
        curl_raw_np=curl_raw,
        curl_corr_np=curl_corr,
        c_gt_np=c_gt,
        c_pred_np=c_pred,
        gt_2d_np=gt_np[0, 0],
        pred_2d_np=pred_np[0, 0],
        filepath=str(out_path / "fig6_physics_zernike.png"),
    )

    print(f"\nAll 6 publication paper figures generated successfully in: {out_path}")


if __name__ == "__main__":
    render_all_figures()
