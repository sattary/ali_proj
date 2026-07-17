"""
Loss functions for absolute phase supervision.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ops import FixedSobel, curvature_loss


import torch.nn.functional as F


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

    Supports Deep Supervision via multi-scale predictions:
    If `phi_pred` is a list, returns weighted sum across scales.
    """

    def __init__(
        self,
        w_mae: float = 1.0,
        w_grad: float = 0.1,
        w_curv: float = 0.01,
        intensity_weighted: bool = False,
        ds_weights: Tuple[float, ...] = (0.25, 0.5, 1.0),
    ) -> None:
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_grad = float(w_grad)
        self.w_curv = float(w_curv)
        self.intensity_weighted = bool(intensity_weighted)
        self.ds_weights = ds_weights
        self.sobel = FixedSobel()

    def _single_scale_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        I_raw: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_err = (pred - gt).abs()

        pgx, pgy = self.sobel(pred)
        tgx, tgy = self.sobel(gt)
        grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()

        if self.intensity_weighted and (I_raw is not None):
            # Scale intensity for potentially downsampled resolutions
            if I_raw.shape[-2:] != pred.shape[-2:]:
                I_raw_scale = F.interpolate(
                    I_raw, size=pred.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                I_raw_scale = I_raw
            wI = I_raw_scale.clamp_min(1e-6).sqrt()
            grad_err = grad_err * wI

        abs_term = abs_err.mean()
        grad_term = grad_err.mean()
        curv_term = curvature_loss(pred)

        loss = self.w_mae * abs_term + self.w_grad * grad_term + self.w_curv * curv_term
        return loss, abs_term, grad_term, curv_term

    def forward(
        self,
        phi_pred: torch.Tensor | list[torch.Tensor],
        phi_gt: torch.Tensor,
        I_raw: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(phi_pred, list):
            # Standard single-scale forward
            loss, mae, grad, curv = self._single_scale_loss(phi_pred, phi_gt, I_raw)
            return loss, {"mae": mae.detach(), "grad": grad.detach(), "curv": curv.detach()}

        # Multi-scale Deep Supervision forward
        total_loss = 0.0
        final_mae = 0.0
        final_grad = 0.0
        final_curv = 0.0

        # Ensure we don't have more weights than predictions
        n_scales = min(len(self.ds_weights), len(phi_pred))

        for i in range(n_scales):
            pred_scale = phi_pred[-(i + 1)]  # Go backwards from finest to coarsest
            weight = self.ds_weights[-(i + 1)]

            # Downsample GT to match current prediction scale
            if pred_scale.shape[-2:] != phi_gt.shape[-2:]:
                gt_scale = F.interpolate(
                    phi_gt,
                    size=pred_scale.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                gt_scale = phi_gt

            loss, mae, grad, curv = self._single_scale_loss(pred_scale, gt_scale, I_raw)
            total_loss += weight * loss

            # Only track the finest resolution (last index) metrics for logging
            if i == 0:
                final_mae = mae
                final_grad = grad
                final_curv = curv

        # Must divide by sum of weights to keep learning rate magnitude stable
        weight_sum = sum(self.ds_weights[-n_scales:])
        total_loss = total_loss / weight_sum

        return total_loss, {
            "mae": torch.as_tensor(final_mae).detach(),
            "grad": torch.as_tensor(final_grad).detach(),
            "curv": torch.as_tensor(final_curv).detach(),
        }
