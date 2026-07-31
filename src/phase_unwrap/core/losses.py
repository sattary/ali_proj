"""
Loss functions for absolute phase supervision.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import FixedSobel, curvature_loss


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





class ComplexDomainLoss(nn.Module):
    """
    Complex-domain sine/cosine consistency loss.
    L = |sin(pred) - sin(gt)| + |cos(pred) - cos(gt)|
    Prevents 2pi jump artifacts and atan2 singularities at V -> 0.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        sin_err = (torch.sin(pred) - torch.sin(gt)).abs()
        cos_err = (torch.cos(pred) - torch.cos(gt)).abs()
        return (sin_err + cos_err).mean()


class GradientCurlLoss(nn.Module):
    """
    Integrability loss penalizing non-zero curl in corrected gradient fields.
    L_curl = |d(gy)/dx - d(gx)/dy|
    Forces CNN 1 to output curl-free gradients for DCT-Poisson integration.
    """

    def forward(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        # Finite differences for curl
        dgy_dx = torch.diff(gy, dim=-1, prepend=gy[..., :1])
        dgx_dy = torch.diff(gx, dim=-2, prepend=gx[..., :1, :])
        curl = (dgy_dx - dgx_dy).abs()
        return curl.mean()


class CurvatureWeightedGradLoss(nn.Module):
    """
    Grad loss weighted by inverse local fringe density (1 + |grad phi_gt|^2 / pi^2)^(-1).
    Avoids over-penalizing steep fringe boundaries near spherical beam edges.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sobel = FixedSobel()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pgx, pgy = self.sobel(pred)
        tgx, tgy = self.sobel(gt)
        grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()
        
        # Local fringe density weighting
        fringe_density = 1.0 + (tgx**2 + tgy**2) / (math.pi**2)
        weighted_err = grad_err / fringe_density
        return weighted_err.mean()


class PCLCNLoss(nn.Module):
    """
    Combined PCLCN loss module for Option A (supervised) and Option B (unsupervised).
    """

    def __init__(
        self,
        w_mae: float = 1.0,
        w_phase: float = 1.0,
        w_curl: float = 0.1,
        w_grad_curv: float = 0.1,
        option_b: bool = False,
    ) -> None:
        super().__init__()
        self.w_mae = w_mae
        self.w_phase = w_phase
        self.w_curl = w_curl
        self.w_grad_curv = w_grad_curv
        self.option_b = option_b

        self.complex_loss = ComplexDomainLoss()
        self.curl_loss = GradientCurlLoss()
        self.curv_grad_loss = CurvatureWeightedGradLoss()

    def forward(
        self,
        phi_pred: torch.Tensor,
        phi_gt: torch.Tensor,
        gx_tilde: torch.Tensor,
        gy_tilde: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_phase = self.complex_loss(phi_pred, phi_gt)
        l_curl = self.curl_loss(gx_tilde, gy_tilde)
        l_grad = self.curv_grad_loss(phi_pred, phi_gt)

        if self.option_b:
            # Unsupervised: no direct MAE on phi_gt
            total_loss = self.w_phase * l_phase + self.w_curl * l_curl
            l_mae = torch.tensor(0.0, device=phi_pred.device)
        else:
            # Supervised Option A
            l_mae = (phi_pred - phi_gt).abs().mean()
            total_loss = (
                self.w_mae * l_mae
                + self.w_phase * l_phase
                + self.w_curl * l_curl
                + self.w_grad_curv * l_grad
            )

        metrics = {
            "mae": l_mae.detach(),
            "phase": l_phase.detach(),
            "curl": l_curl.detach(),
            "grad_curv": l_grad.detach(),
        }
        return total_loss, metrics

