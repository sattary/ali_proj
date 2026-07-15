"""
Signal-processing operators for phase supervision.

Contains:
    - FixedSobel: depthwise Sobel gradient extraction.
    - laplacian: discrete 2D Laplacian with reflect padding.
    - curvature_loss: L1 Laplacian smoothness penalty.
    - affine_align: per-image least-squares affine fit (eval-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSobel(nn.Module):
    """Depthwise Sobel operator producing x and y gradients per channel."""

    def __init__(self) -> None:
        super().__init__()
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        gy = gx.t().contiguous()
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = self.gx.to(x.device, x.dtype)
        gy = self.gy.to(x.device, x.dtype)
        C = x.shape[1]
        kx, ky = gx.repeat(C, 1, 1, 1), gy.repeat(C, 1, 1, 1)
        px = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(px, kx, groups=C), F.conv2d(px, ky, groups=C)


def laplacian(u: torch.Tensor) -> torch.Tensor:
    """Discrete 2D Laplacian with reflect padding. Input: [B, 1, H, W]."""
    u_pad = F.pad(u, (1, 1, 1, 1), mode="reflect")
    return (
        -4 * u_pad[..., 1:-1, 1:-1]
        + u_pad[..., 1:-1, 2:]
        + u_pad[..., 1:-1, :-2]
        + u_pad[..., 2:, 1:-1]
        + u_pad[..., :-2, 1:-1]
    )


def curvature_loss(phi_pred: torch.Tensor) -> torch.Tensor:
    """
    L1 Laplacian smoothness penalty.

    Args:
        phi_pred: [B, 1, H, W] predicted phase.
    """
    return laplacian(phi_pred).abs().mean()


def affine_align(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-image affine fit: ``a * pred + c ~ gt`` (least squares).

    Diagnostic / figure tool only. Do **not** use for headline metrics: the scale
    ``a`` is fit to GT and is not available at deployment. Prefer ``piston_align``.

    Args:
        pred: [B, 1, H, W] predictions.
        gt:   [B, 1, H, W] ground truth.

    Returns:
        pred_aligned [B,1,H,W], a [B,1], c [B,1].
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    gt_flat = gt.reshape(B, -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    gt_mean = gt_flat.mean(dim=1, keepdim=True)

    pred_centered = pred_flat - pred_mean
    gt_centered = gt_flat - gt_mean

    var_pred = (pred_centered**2).mean(dim=1, keepdim=True) + 1e-6
    cov_pg = (pred_centered * gt_centered).mean(dim=1, keepdim=True)

    a = cov_pg / var_pred
    c = gt_mean - a * pred_mean

    a_map = a.unsqueeze(-1).unsqueeze(-1)
    c_map = c.unsqueeze(-1).unsqueeze(-1)

    pred_aligned = a_map * pred + c_map
    return pred_aligned, a, c


def piston_align(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-image offset-only alignment: ``pred + c ~ gt`` (mean piston removal).

    Absolute phase is defined only up to a global additive constant without a
    reference; removing ``c`` is physically justified. Unlike ``affine_align``
    this does NOT fit a scale ``a`` to the ground truth.

    Returns:
        pred_aligned [B,1,H,W], c [B,1].
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    gt_flat = gt.reshape(B, -1)
    c = gt_flat.mean(dim=1, keepdim=True) - pred_flat.mean(dim=1, keepdim=True)
    c_map = c.unsqueeze(-1).unsqueeze(-1)
    return pred + c_map, c
