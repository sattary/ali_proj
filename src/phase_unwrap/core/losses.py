from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ops import FixedSobel


def compute_metrics(
    phi_pred: torch.Tensor, phi_gt: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute MAE, RMSE, and NRMSE between absolute phase predictions and ground truth.

    Both inputs are [B, 1, H, W] tensors. No phase wrapping is applied.
    NRMSE is normalized by the dynamic range of the ground truth.
    """
    diff = phi_pred - phi_gt
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff**2).mean().clamp_min(1e-12))

    gt_min = phi_gt.min()
    gt_max = phi_gt.max()
    denom = (gt_max - gt_min).clamp_min(1e-6)
    nrmse = rmse / denom

    return {"MAE": mae.detach(), "RMSE": rmse.detach(), "NRMSE": nrmse.detach()}


class MAEGradCore(nn.Module):
    """
    Core supervised loss on absolute phase:

      L_mae = |phi_abs - phi_gt|
      L_grad = |∇phi_abs - ∇phi_gt|

    Both terms can be masked by a confidence map, and the gradient term
    can optionally be weighted by sqrt(I_raw).
    """

    def __init__(
        self, w_mae: float = 1.0, w_grad: float = 0.1, intensity_weighted: bool = False
    ) -> None:
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_grad = float(w_grad)
        self.intensity_weighted = bool(intensity_weighted)
        self.sobel = FixedSobel()

    def forward(
        self,
        phi_pred_abs: torch.Tensor,
        phi_gt: torch.Tensor,
        I_raw: torch.Tensor | None,
        conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # phi_pred_abs, phi_gt: [B,1,H,W]
        # conf: [B,1,H,W] in [0,1]
        # I_raw: [B,1,H,W], optional

        abs_err = (phi_pred_abs - phi_gt).abs()

        pgx, pgy = self.sobel(phi_pred_abs)
        tgx, tgy = self.sobel(phi_gt)
        grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()

        if self.intensity_weighted and (I_raw is not None):
            wI = I_raw.clamp_min(1e-6).sqrt()
            grad_err = grad_err * wI

        abs_term = (conf * abs_err).mean()
        grad_term = (conf * grad_err).mean()

        total = self.w_mae * abs_term + self.w_grad * grad_term
        return total, {"mae": abs_term.detach(), "grad": grad_term.detach()}


class PhaseSupervisionLoss(nn.Module):
    """
    Supervision loss on absolute phase with an optional wrap-consistent term.

    In the default configuration, the wrap term is disabled (w_wrap = 0.0),
    so the loss acts purely on absolute (unwrapped) phase.
    """

    def __init__(
        self,
        w_mae: float = 1.0,
        w_grad: float = 0.1,
        w_wrap: float = 0.0,
        intensity_weighted: bool = False,
    ) -> None:
        super().__init__()
        self.core = MAEGradCore(
            w_mae=w_mae,
            w_grad=w_grad,
            intensity_weighted=intensity_weighted,
        )
        self.w_wrap = float(w_wrap)

    def forward(
        self,
        phi_pred_abs: torch.Tensor,
        phi_gt: torch.Tensor,
        I_raw: torch.Tensor | None,
        conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        L_core, parts = self.core(phi_pred_abs, phi_gt, I_raw, conf)

        # wrapping / periodic match (disabled by default via w_wrap = 0.0)
        s_pred = torch.sin(phi_pred_abs)
        c_pred = torch.cos(phi_pred_abs)
        s_gt = torch.sin(phi_gt)
        c_gt = torch.cos(phi_gt)
        wrap_err = torch.sqrt((s_pred - s_gt) ** 2 + (c_pred - c_gt) ** 2 + 1e-8)
        L_wrap = (conf * wrap_err).mean()

        total = L_core + self.w_wrap * L_wrap
        parts_out: Dict[str, torch.Tensor] = {
            "mae": parts["mae"],
            "grad": parts["grad"],
            "wrap": L_wrap.detach(),
        }
        return total, parts_out


class WrappedGradLoss(nn.Module):
    """
    Physics-informed loss: standardizes the logic that "gradients of absolute phase"
    must match "gradients of wrapped phase" (modulo 2pi).

    L_wg = | W(nabla phi_pred) - W(nabla phi_gt) |

    where W is the wrapping operator (-pi, pi].
    This allows the network to learn global phase jumps that disappear in the wrapped domain.
    """

    def __init__(self, w_grad=0.1):
        super().__init__()
        self.w_grad = w_grad
        self.sobel = FixedSobel()

    def wrap(self, x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(
        self,
        phi_pred_abs: torch.Tensor,
        phi_gt: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Compute gradients
        pgx, pgy = self.sobel(phi_pred_abs)
        tgx, tgy = self.sobel(phi_gt)

        # 2. Wrap the gradients
        # Note: If phi_pred is absolute, its gradient might be > pi.
        # Consistency requires W(grad(phi)) to match.
        w_pgx, w_pgy = self.wrap(pgx), self.wrap(pgy)
        w_tgx, w_tgy = self.wrap(tgx), self.wrap(tgy)

        # 3. Compute error in wrapped gradient domain
        # The error is the minimal difference modulo 2pi
        diff_x = self.wrap(w_pgx - w_tgx)
        diff_y = self.wrap(w_pgy - w_tgy)

        loss = (conf * (diff_x.abs() + diff_y.abs())).mean()
        return self.w_grad * loss


class ResidueLoss(nn.Module):
    """
    Binary Cross Entropy loss for residue detection.
    Encourages the model to identify topological phase jumps.
    """

    def __init__(self, w_res: float = 1.0):
        super().__init__()
        self.w_res = w_res
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, res_logit: torch.Tensor, res_gt: torch.Tensor, conf: torch.Tensor
    ) -> torch.Tensor:
        # res_gt: [B, 1, H, W] binary mask
        # logit: [B, 1, H, W]

        # Weighted BCE: Residues are sparse, so we weight them more if needed.
        # But for now, standard BCE with spatial confidence masking.
        loss = self.bce(res_logit, res_gt)

        # Optionally apply positive weight for sparsity directly in BCE if needed
        # For now, we scale by w_res and conf
        return self.w_res * (loss * conf).mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    Encourages the model to preserve local structure and contrast,
    preventing the 'blurry' predictions common with pure L1/L2 losses.
    """

    def __init__(
        self, w_ssim: float = 0.1, window_size: int = 11, in_channels: int = 1
    ):
        super().__init__()
        self.w_ssim = w_ssim
        self.window_size = window_size
        self.channel = in_channels
        self.window = self.create_window(window_size, in_channels)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor(
            [
                -((x - window_size // 2) ** 2) / float(2 * sigma**2)
                for x in range(window_size)
            ]
        )
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=channel
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=channel
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()

    def forward(
        self, phi_pred: torch.Tensor, phi_gt: torch.Tensor, conf: torch.Tensor = None
    ) -> torch.Tensor:
        # If window is on wrong device, move it
        if self.window.device != phi_pred.device:
            self.window = self.window.to(phi_pred.device)

        # SSIM assumes input in [0, 1] typically, but it works for any range if C1/C2 are small relative to range.
        # Phase is roughly [-pi, pi] or similar. We can normalize or just run it.
        # For simplicity, we run it directly on the phase values.

        return self.w_ssim * (
            1.0
            - self._ssim(phi_pred, phi_gt, self.window, self.window_size, self.channel)
        )
