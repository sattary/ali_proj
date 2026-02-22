"""
Loss functions for absolute phase supervision.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ops import FixedSobel


def compute_metrics(
    phi_pred: torch.Tensor, phi_gt: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    MAE and RMSE between absolute phase predictions and ground truth.

    Both inputs: [B, 1, H, W]. No phase wrapping.
    """
    diff = phi_pred - phi_gt
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff**2).mean().clamp_min(1e-12))
    return {"MAE": mae.detach(), "RMSE": rmse.detach()}


class MAEGradLoss(nn.Module):
    """
    Supervised loss on absolute phase.

    ``L = w_mae * |pred - gt| + w_grad * |grad(pred) - grad(gt)|``

    The gradient term can optionally be weighted by ``sqrt(I_raw)``
    (enabled via ``intensity_weighted=True`` / ``--int-wgrad``).
    """

    def __init__(
        self,
        w_mae: float = 1.0,
        w_grad: float = 0.1,
        intensity_weighted: bool = False,
    ) -> None:
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_grad = float(w_grad)
        self.intensity_weighted = bool(intensity_weighted)
        self.sobel = FixedSobel()

    def forward(
        self,
        phi_pred: torch.Tensor,
        phi_gt: torch.Tensor,
        I_raw: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        abs_err = (phi_pred - phi_gt).abs()

        pgx, pgy = self.sobel(phi_pred)
        tgx, tgy = self.sobel(phi_gt)
        grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()

        if self.intensity_weighted and (I_raw is not None):
            wI = I_raw.clamp_min(1e-6).sqrt()
            grad_err = grad_err * wI

        abs_term = abs_err.mean()
        grad_term = grad_err.mean()

        total = self.w_mae * abs_term + self.w_grad * grad_term
        return total, {"mae": abs_term.detach(), "grad": grad_term.detach()}