"""
Test-time augmentation (TTA) for phase prediction.

Predicts on 4 rotations x 2 flips = 8 augmented views,
averages the predictions for improved accuracy at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.amp import autocast

from .config import load_train_config
from .model import build_model
from .ops import affine_align
from .utils import pick_device


def _rotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate tensor by k*90 degrees (counterclockwise)."""
    return torch.rot90(x, k, dims=(2, 3))


def _unrotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of _rotate90."""
    return torch.rot90(x, -k, dims=(2, 3))


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(3,))


def _unhflip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(3,))


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    I_input: torch.Tensor,
    device: torch.device,
    augments: Sequence[str] = (
        "rot0",
        "rot1",
        "rot2",
        "rot3",
        "hflip_rot0",
        "hflip_rot1",
        "hflip_rot2",
        "hflip_rot3",
    ),
) -> torch.Tensor:
    """
    TTA prediction: average over geometric augmentations.

    Args:
        model:    Trained model in eval mode.
        I_input:  [B, 2, H, W] input tensor.
        device:   Compute device.
        augments: Which augmentations to use. Subset of all 8.

    Returns:
        phi_abs: [B, 1, H, W] averaged absolute phase prediction.
    """
    model.eval()
    I_input = I_input.to(device)
    preds: list[torch.Tensor] = []

    for aug in augments:
        flip = aug.startswith("hflip")
        k = int(aug[-1])  # rotation count

        x = _hflip(I_input) if flip else I_input
        x = _rotate90(x, k)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(x)
            phi_abs = phi_raw + k_off

        # undo augmentation
        phi_abs = _unrotate90(phi_abs, k)
        if flip:
            phi_abs = _unhflip(phi_abs)

        preds.append(phi_abs)

    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_tta(
    checkpoint_path: str,
    data_dir: str,
    n_augments: int = 8,
    config_path: str | None = None,
) -> dict[str, float]:
    """
    Evaluate model with and without TTA for comparison.

    Returns:
        Dict with 'mae_noaug' and 'mae_tta'.
    """
    from .data import build_dataloaders

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

    ALL_AUGS = (
        "rot0",
        "rot1",
        "rot2",
        "rot3",
        "hflip_rot0",
        "hflip_rot1",
        "hflip_rot2",
        "hflip_rot3",
    )
    augs = ALL_AUGS[:n_augments]

    total_noaug = 0.0
    total_tta = 0.0
    n = 0

    for I_input, phi_gt, _ in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        # no augmentation
        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            phi_noaug = phi_raw + k_off
        aligned_noaug, _, _ = affine_align(phi_noaug, phi_gt)
        total_noaug += float((aligned_noaug - phi_gt).abs().mean()) * I_input.size(0)

        # TTA
        phi_tta = predict_tta(model, I_input, device, augments=augs)
        aligned_tta, _, _ = affine_align(phi_tta, phi_gt)
        total_tta += float((aligned_tta - phi_gt).abs().mean()) * I_input.size(0)

        n += I_input.size(0)

    mae_noaug = total_noaug / max(1, n)
    mae_tta = total_tta / max(1, n)
    improvement = (mae_noaug - mae_tta) / mae_noaug * 100

    print(f"MAE without TTA: {mae_noaug:.4f}")
    print(f"MAE with TTA ({n_augments} augs): {mae_tta:.4f}")
    print(f"Improvement: {improvement:.1f}%")

    return {"mae_noaug": mae_noaug, "mae_tta": mae_tta, "improvement_pct": improvement}
