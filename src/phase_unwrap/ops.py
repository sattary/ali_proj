from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSobel(nn.Module):
    """
    Depthwise Sobel operator producing x and y gradients per channel.
    """

    def __init__(self) -> None:
        super().__init__()
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        gy = gx.t().contiguous()
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W]
        gx = self.gx.to(x.device, x.dtype)
        gy = self.gy.to(x.device, x.dtype)
        C = x.shape[1]
        kx, ky = gx.repeat(C, 1, 1, 1), gy.repeat(C, 1, 1, 1)
        px = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(px, kx, groups=C), F.conv2d(px, ky, groups=C)


def laplacian(u: torch.Tensor) -> torch.Tensor:
    """
    Simple 2D Laplacian with reflect padding.

    Args:
        u: Tensor of shape [B, 1, H, W].
    """
    u_pad = F.pad(u, (1, 1, 1, 1), mode="reflect")
    return (
        -4 * u_pad[..., 1:-1, 1:-1]
        + u_pad[..., 1:-1, 2:]
        + u_pad[..., 1:-1, :-2]
        + u_pad[..., 2:, 1:-1]
        + u_pad[..., :-2, 1:-1]
    )


def adaptive_curvature_loss(phi_pred: torch.Tensor, conf_mask: torch.Tensor) -> torch.Tensor:
    """
    Smoothness / curvature penalty, weighted by conf_mask in [0,1].

    Args:
        phi_pred: [B, 1, H, W] predicted phase.
        conf_mask: [B, 1, H, W] confidence weights in [0, 1].
    """
    curv = laplacian(phi_pred).abs()
    wcurv = conf_mask * curv
    return wcurv.mean()


def affine_align(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve per-image affine fit a * pred + c ≈ gt (least squares).

    Args:
        pred: [B, 1, H, W] predictions.
        gt:   [B, 1, H, W] ground truth.

    Returns:
        pred_aligned: [B, 1, H, W] affine-aligned predictions.
        a: [B, 1] scale coefficients.
        c: [B, 1] offset coefficients.
    """
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    gt_mean = gt_flat.mean(dim=1, keepdim=True)

    pred_centered = pred_flat - pred_mean
    gt_centered = gt_flat - gt_mean

    var_pred = (pred_centered ** 2).mean(dim=1, keepdim=True) + 1e-6
    cov_pg = (pred_centered * gt_centered).mean(dim=1, keepdim=True)

    a = cov_pg / var_pred
    c = gt_mean - a * pred_mean

    a_map = a.unsqueeze(-1).unsqueeze(-1)
    c_map = c.unsqueeze(-1).unsqueeze(-1)

    pred_aligned = a_map * pred + c_map
    return pred_aligned, a, c


