"""
Test-time augmentation (TTA) for PCLCN phase prediction.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..core.ops import piston_align


def _rotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(2, 3))


def _unrotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, -k, dims=(2, 3))


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(3,))


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    I_raw: torch.Tensor,
    grad_phi2: torch.Tensor,
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
    TTA: average PCLCN predictions over D4 symmetry group.
    """
    model.eval()
    I_raw = I_raw.to(device)
    grad_phi2 = grad_phi2.to(device)
    preds: list[torch.Tensor] = []

    for aug in augments:
        flip = aug.startswith("hflip")
        k = int(aug[-1])

        x_i = _hflip(I_raw) if flip else I_raw
        x_i = _rotate90(x_i, k)

        x_g = _hflip(grad_phi2) if flip else grad_phi2
        x_g = _rotate90(x_g, k)

        phi_abs, _, _, _, _ = model(x_i, x_g)

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
    Compare PCLCN model MAE with and without TTA on validation set.
    """
    from ..core.inference import load_inference_state

    model, cfg, loader, device = load_inference_state(
        checkpoint_path,
        data_dir=data_dir,
        config_path=config_path,
        subset=subset,
        all_data=all_data,
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

    for batch in loader:
        I_raw, phi_gt, grad_phi2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        phi_noaug, _, _, _, _ = model(I_raw, grad_phi2)
        aligned_noaug, _ = piston_align(phi_noaug, phi_gt)
        total_noaug += float((aligned_noaug - phi_gt).abs().mean()) * I_raw.size(0)

        phi_tta = predict_tta(model, I_raw, grad_phi2, device, augments=augs)
        aligned_tta, _ = piston_align(phi_tta, phi_gt)
        total_tta += float((aligned_tta - phi_gt).abs().mean()) * I_raw.size(0)

        n += I_raw.size(0)

    mae_noaug = total_noaug / max(1, n)
    mae_tta = total_tta / max(1, n)
    improvement = (mae_noaug - mae_tta) / max(mae_noaug, 1e-8) * 100

    print(f"MAE without TTA: {mae_noaug:.4f}")
    print(f"MAE with TTA ({n_augments} augs): {mae_tta:.4f}")
    print(f"Improvement: {improvement:.1f}%")

    return {"mae_noaug": mae_noaug, "mae_tta": mae_tta, "improvement_pct": improvement}
