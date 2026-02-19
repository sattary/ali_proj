"""
Classical phase unwrapping baselines for comparison.

Wraps scikit-image 2D unwrapping and custom Goldstein branch-cut
as thin adapters to run on the same test data and produce identical
metric comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .generate import _build_grid, generate_sample
from .ops import affine_align


def _unwrap_skimage(I: np.ndarray) -> np.ndarray:
    """
    Least-squares 2D phase unwrapping via skimage.

    Input:  I (H, W) interferogram intensity.
    Output: unwrapped phase (H, W).
    """
    from skimage.restoration import unwrap_phase

    # extract wrapped phase from interferogram
    # I = |E1 + E2|^2 = 2 + 2*cos(dphi)  =>  dphi = arccos((I/2 - 1).clip(-1,1))
    # but more robust: use Hilbert/angle approach
    wrapped = np.angle(np.exp(1j * I))
    return unwrap_phase(wrapped).astype(np.float32)


def _unwrap_itoh(I: np.ndarray) -> np.ndarray:
    """
    Itoh's method: 1D unwrapping along rows then columns.

    Fastest classical method but fails on discontinuities.
    """
    wrapped = np.angle(np.exp(1j * I))
    unwrapped = np.copy(wrapped)

    # unwrap along rows
    for row in range(wrapped.shape[0]):
        diff = np.diff(wrapped[row])
        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi
        unwrapped[row, 1:] = wrapped[row, 0] + np.cumsum(diff)

    # unwrap along columns (refine)
    for col in range(unwrapped.shape[1]):
        diff = np.diff(unwrapped[:, col])
        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi
        unwrapped[1:, col] = unwrapped[0, col] + np.cumsum(diff)

    return unwrapped.astype(np.float32)


def evaluate_baselines(
    n_samples: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Evaluate classical methods on synthetic test data.

    Returns:
        Dict of {method_name: {mae: float, rmse: float}}.
    """
    x, y, r2 = _build_grid()
    rng = np.random.default_rng(seed)

    methods = {
        "Least-Squares (skimage)": _unwrap_skimage,
        "Itoh 1D": _unwrap_itoh,
    }

    results: dict[str, dict[str, list[float]]] = {
        name: {"mae": [], "rmse": []} for name in methods
    }

    for i in range(n_samples):
        I_clean, dphi_gt = generate_sample(x, y, r2, rng)

        for name, method in methods.items():
            try:
                pred = method(I_clean)

                # affine align (same as DL pipeline)
                pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(dphi_gt).unsqueeze(0).unsqueeze(0)
                aligned, _, _ = affine_align(pred_t, gt_t)

                mae = float((aligned - gt_t).abs().mean().item())
                rmse = float(((aligned - gt_t) ** 2).mean().sqrt().item())

                results[name]["mae"].append(mae)
                results[name]["rmse"].append(rmse)
            except Exception:
                results[name]["mae"].append(float("nan"))
                results[name]["rmse"].append(float("nan"))

    summary: dict[str, dict[str, float]] = {}
    for name in methods:
        maes = np.array(results[name]["mae"])
        rmses = np.array(results[name]["rmse"])
        maes = maes[~np.isnan(maes)]
        rmses = rmses[~np.isnan(rmses)]

        summary[name] = {
            "mae_mean": float(np.mean(maes)) if len(maes) > 0 else float("nan"),
            "mae_std": float(np.std(maes)) if len(maes) > 0 else float("nan"),
            "rmse_mean": float(np.mean(rmses)) if len(rmses) > 0 else float("nan"),
            "rmse_std": float(np.std(rmses)) if len(rmses) > 0 else float("nan"),
        }

        print(
            f"  {name:30s}  MAE={summary[name]['mae_mean']:.4f} +/- "
            f"{summary[name]['mae_std']:.4f}  "
            f"RMSE={summary[name]['rmse_mean']:.4f}"
        )

    return summary


@torch.no_grad()
def evaluate_dl_baseline(
    checkpoint_path: str,
    n_samples: int = 200,
    seed: int = 42,
    device_str: str = "cpu",
    config_path: str | None = None,
) -> dict[str, float]:
    """
    Evaluate the DL model on the same samples used for classical baselines.

    Returns:
        Dict with mae_mean, mae_std, rmse_mean, rmse_std.
    """
    from torch.amp import autocast

    from .config import load_train_config
    from .model import build_model
    from .utils import pick_device

    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)

    device = pick_device(device_str)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
    model.eval()

    x, y, r2 = _build_grid()
    rng = np.random.default_rng(seed)

    maes: list[float] = []
    rmses: list[float] = []

    for _ in range(n_samples):
        I_clean, dphi_gt = generate_sample(x, y, r2, rng)

        I_mean = I_clean.mean()
        I_std = I_clean.std() + 1e-6
        I_norm = (I_clean - I_mean) / I_std
        phi_hint = np.angle(np.exp(1j * I_clean))

        inp = np.stack([I_norm, phi_hint], axis=0)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(dphi_gt).unsqueeze(0).unsqueeze(0).to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(inp_t)
            phi_abs = phi_raw + k_off

        aligned, _, _ = affine_align(phi_abs, gt_t)
        maes.append(float((aligned - gt_t).abs().mean().item()))
        rmses.append(float(((aligned - gt_t) ** 2).mean().sqrt().item()))

    maes_arr = np.array(maes)
    rmses_arr = np.array(rmses)

    result = {
        "mae_mean": float(np.mean(maes_arr)),
        "mae_std": float(np.std(maes_arr)),
        "rmse_mean": float(np.mean(rmses_arr)),
        "rmse_std": float(np.std(rmses_arr)),
    }

    print(
        f"  UNetRes2 (Ours)                MAE={result['mae_mean']:.4f} +/- "
        f"{result['mae_std']:.4f}  RMSE={result['rmse_mean']:.4f}"
    )
    return result
