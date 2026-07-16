"""
Classical phase unwrapping baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..core.ops import piston_align
from ..data.generate import _build_grid, generate_sample


def _unwrap_skimage(interferogram: np.ndarray) -> np.ndarray:
    """Least-squares 2D phase unwrapping via skimage."""
    from skimage.restoration import unwrap_phase

    wrapped = np.angle(np.exp(1j * interferogram))
    return unwrap_phase(wrapped).astype(np.float32)


def _unwrap_itoh(interferogram: np.ndarray) -> np.ndarray:
    """Itoh's 1D method: unwrap along rows, then columns."""
    wrapped = np.angle(np.exp(1j * interferogram))
    unwrapped = np.copy(wrapped)

    for row in range(wrapped.shape[0]):
        diff = np.diff(wrapped[row])
        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi
        unwrapped[row, 1:] = wrapped[row, 0] + np.cumsum(diff)

    for col in range(unwrapped.shape[1]):
        diff = np.diff(unwrapped[:, col])
        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi
        unwrapped[1:, col] = unwrapped[0, col] + np.cumsum(diff)

    return unwrapped.astype(np.float32)


def evaluate_baselines(
    data_dir: str,
    n_samples: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate classical methods on synthetic test data."""
    from ..core.config import TrainConfig
    from ..data.dataset import discover_h5_shards, smart_split, H5ShardDataset
    cfg = TrainConfig()
    cfg.data.data_dir = data_dir
    paths = discover_h5_shards(cfg.data)
    _, _, test_paths = smart_split(paths, seed=seed)
    
    if not test_paths:
        raise ValueError(f"No test data found in {data_dir}")
        
    test_ds = H5ShardDataset(test_paths, augment=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_ds), size=min(n_samples, len(test_ds)), replace=False)

    methods = {
        "Least-Squares (skimage)": _unwrap_skimage,
        "Itoh 1D": _unwrap_itoh,
    }

    results: dict[str, dict[str, list[float]]] = {
        name: {"mae": [], "rmse": []} for name in methods
    }

    for idx in indices:
        I_t, gt_t = test_ds[int(idx)]
        I_clean = I_t.squeeze().numpy()
        dphi_gt = gt_t.squeeze().numpy()

        for name, method in methods.items():
            try:
                pred = method(I_clean)
                pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(dphi_gt).unsqueeze(0).unsqueeze(0)
                aligned, _ = piston_align(pred_t, gt_t)

                results[name]["mae"].append(float((aligned - gt_t).abs().mean().item()))
                results[name]["rmse"].append(
                    float(((aligned - gt_t) ** 2).mean().sqrt().item())
                )
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
            f"  {name:30s}  MAE={summary[name]['mae_mean']:.4f} "
            f"+/- {summary[name]['mae_std']:.4f}  "
            f"RMSE={summary[name]['rmse_mean']:.4f}"
        )

    return summary


@torch.no_grad()
def evaluate_dl_baseline(
    checkpoint_path: str,
    data_dir: str,
    n_samples: int = 200,
    seed: int = 42,
    device_str: str = "cpu",
    config_path: str | None = None,
) -> dict[str, float]:
    """Evaluate the DL model on the same samples as the classical baselines."""
    from torch.amp import autocast
            
    from ..core.inference import load_inference_state
    from ..data.dataset import discover_h5_shards, smart_split, H5ShardDataset
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)
    cfg.data.data_dir = data_dir
    paths = discover_h5_shards(cfg.data)
    _, _, test_paths = smart_split(paths, seed=seed)
    
    if not test_paths:
        raise ValueError(f"No test data found in {data_dir}")
        
    test_ds = H5ShardDataset(test_paths, augment=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_ds), size=min(n_samples, len(test_ds)), replace=False)

    maes: list[float] = []
    rmses: list[float] = []

    for idx in indices:
        I_t, gt_t = test_ds[int(idx)]
        I_clean = I_t.squeeze().numpy()
        dphi_gt = gt_t.squeeze().numpy()

        I_mean = I_clean.mean()
        I_std = I_clean.std() + 1e-6
        I_norm = (I_clean - I_mean) / I_std
        # Option B: zero hint — same as train/deploy (no GT absolute channel)
        phi_hint = np.zeros_like(I_norm, dtype=np.float32)

        inp_t = (
            torch.from_numpy(np.stack([I_norm, phi_hint], 0)).unsqueeze(0).to(device)
        )
        gt_t = torch.from_numpy(dphi_gt).unsqueeze(0).unsqueeze(0).to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(inp_t)
            phi_abs = phi_raw + k_off

        aligned, _ = piston_align(phi_abs, gt_t)
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
        f"  UNetRes2 (Ours)                MAE={result['mae_mean']:.4f} "
        f"+/- {result['mae_std']:.4f}  RMSE={result['rmse_mean']:.4f}"
    )
    return result