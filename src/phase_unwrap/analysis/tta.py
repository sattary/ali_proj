"""
Test-time augmentation (TTA) for phase prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.amp import autocast

from ..core.ops import affine_align


def _rotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(2, 3))


def _unrotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, -k, dims=(2, 3))


def _hflip(x: torch.Tensor) -> torch.Tensor:
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
    TTA: average predictions over 4 rotations x 2 flips (D4 group).

    Returns phi_abs: [B, 1, H, W].
    """
    model.eval()
    I_input = I_input.to(device)
    preds: list[torch.Tensor] = []

    for aug in augments:
        flip = aug.startswith("hflip")
        k = int(aug[-1])

        x = _hflip(I_input) if flip else I_input
        x = _rotate90(x, k)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(x)
            if isinstance(phi_raw, list):
                phi_abs = phi_raw[-1] + k_off
            else:
                phi_abs = phi_raw + k_off

        phi_abs = _unrotate90(phi_abs, k)
        if flip:
            phi_abs = _hflip(phi_abs)

        preds.append(phi_abs)

    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_tta(
    checkpoint_path: str,
    data_dir: str,
    n_augments: int = 8,
    config_path: str | None = None,
    subset: str = "val",
    all_data: bool = False,
) -> dict[str, float]:
    """
    Compare model MAE with and without TTA on the validation set.

    Returns dict with 'mae_noaug', 'mae_tta', 'improvement_pct'.
    """
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )

    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = (
            val_loader
            if (val_loader is not None and len(val_loader) > 0)
            else train_loader
        )
    else:
        loader = train_loader

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

    for I_input, phi_gt, _, _ in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            if isinstance(phi_raw, list):
                phi_noaug = phi_raw[-1] + k_off
            else:
                phi_noaug = phi_raw + k_off
        aligned_noaug, _, _ = affine_align(phi_noaug, phi_gt)
        total_noaug += float((aligned_noaug - phi_gt).abs().mean()) * I_input.size(0)

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
