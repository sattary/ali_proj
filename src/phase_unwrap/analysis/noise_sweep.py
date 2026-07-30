"""
Core computation for noise robustness sweep.
Extracts SNR vs MAE performance metrics.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch.amp import autocast

from ..core.ops import piston_align

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
def compute_noise_robustness(
    checkpoint_path: str,
    data_dir: str,
    snr_range: tuple[float, float] = (5.0, 40.0),
    n_snr_steps: int = 8,
    n_samples: int = 100,
    seed: int = 42,
    config_path: str | None = None,
    device_str: str = "auto",
) -> tuple[dict[str, list[float]], dict[float, list[float]]]:
    """
    Sweep SNR levels and evaluate model robustness.
    Returns:
        results: Dictionary with 'snr_db', 'mae', 'std' arrays.
        all_maes_by_snr: Dictionary mapping SNR to list of individual sample MAEs.
    """
    from ..core.inference import load_inference_state

    model, cfg, loader, device = load_inference_state(
        checkpoint_path, data_dir=data_dir, subset="test", config_path=config_path
    )

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

    return results, all_maes_by_snr