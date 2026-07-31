"""
Physics-Constrained Latent Corrector Network (PCLCN) model implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.ops import (
    AnalyticSignalStem,
    DifferentiablePoissonSolver,
    MaskedZernikeProjection,
    WrappedGradientOperator,
)
from .blocks import AddCoords, Res2_DS_Block


class ReferenceConditionedGradientCorrector(nn.Module):
    """
    CNN 1: Predicts gradient corrections (delta gx, delta gy) to restore integrability.
    Input channels (7 total):
      - I_raw [1]
      - wrapped_phase [1]
      - amplitude [1]
      - wrapped_gradients gx, gy [2]
      - analytical_reference_gradient grad_phi2_x, grad_phi2_y [2]
    """

    def __init__(self, in_ch: int = 7, base: int = 32, act: str = "relu") -> None:
        super().__init__()
        self.addcoords = AddCoords()
        enc_in = in_ch + 2  # CoordConv adds 2 channels

        self.net = nn.Sequential(
            Res2_DS_Block(enc_in, base, 4, 1.0, act),
            Res2_DS_Block(base, base * 2, 4, 1.0, act),
            Res2_DS_Block(base * 2, base, 4, 1.0, act),
            nn.Conv2d(base, 2, 1),
        )

    def forward(
        self,
        I_raw: torch.Tensor,
        wrapped_phase: torch.Tensor,
        amplitude: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        grad_phi2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([I_raw, wrapped_phase, amplitude, gx, gy, grad_phi2], dim=1)
        x_coords = self.addcoords(x)
        delta_g = self.net(x_coords)
        return delta_g[:, 0:1], delta_g[:, 1:2]


class OrthogonalResidualCNN(nn.Module):
    """
    CNN 2: Predicts non-Zernike orthogonal residual phase r(x, y).
    Input: phi_base [B, 1, H, W] -> Output: residual [B, 1, H, W]
    """

    def __init__(self, base: int = 32, act: str = "relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            Res2_DS_Block(1, base, 4, 1.0, act),
            Res2_DS_Block(base, base, 4, 1.0, act),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, phi_base: torch.Tensor) -> torch.Tensor:
        return self.net(phi_base)


class PCLCNModel(nn.Module):
    """
    Physics-Constrained Latent Corrector Network (PCLCN).
    Integrated pipeline:
      AnalyticStem -> WrappedGrad -> CNN 1 (Integrability Corrector) -> DCT-Poisson -> Zernike Projection -> CNN 2 (Residual) -> Final Phase
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        base: int = 32,
        zero_reference_prior: bool = False,
        unmasked_zernike: bool = False,
    ) -> None:
        super().__init__()
        self.zero_reference_prior = zero_reference_prior
        self.unmasked_zernike = unmasked_zernike

        self.stem = AnalyticSignalStem(height=height, width=width)
        self.grad_op = WrappedGradientOperator()
        self.corrector = ReferenceConditionedGradientCorrector(in_ch=7, base=base)
        self.poisson_solver = DifferentiablePoissonSolver(height, width)
        self.zernike_proj = MaskedZernikeProjection(height, width, num_modes=15)
        self.residual_net = OrthogonalResidualCNN(base=base)

    def forward(
        self,
        I_raw: torch.Tensor,
        grad_phi2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            I_raw: [B, 1, H, W] raw off-axis intensity.
            grad_phi2: [B, 2, H, W] analytical reference beam gradient.

        Returns:
            phi_final: [B, 1, H, W] total unwrapped phase.
            gx_tilde: [B, 1, H, W] corrected x-gradient.
            gy_tilde: [B, 1, H, W] corrected y-gradient.
            c_zernike: [B, 15] Zernike coefficients.
            phi_zernike: [B, 1, H, W] Zernike surface.
        """
        if grad_phi2 is None:
            grad_phi2 = torch.zeros(
                I_raw.shape[0],
                2,
                I_raw.shape[2],
                I_raw.shape[3],
                device=I_raw.device,
                dtype=I_raw.dtype,
            )
        if self.zero_reference_prior:
            grad_phi2 = torch.zeros_like(grad_phi2)

        # Step 1: Deterministic Fourier demodulation
        wrapped_phase, amplitude = self.stem(I_raw)

        # Step 2: Extract wrapped gradients
        gx, gy = self.grad_op(wrapped_phase)

        # Step 3: CNN 1 predicts gradient correction (delta_gx, delta_gy)
        delta_gx, delta_gy = self.corrector(
            I_raw, wrapped_phase, amplitude, gx, gy, grad_phi2
        )
        gx_tilde = gx + delta_gx
        gy_tilde = gy + delta_gy

        # Step 4: Differentiable DCT-Poisson integration
        phi_base = self.poisson_solver(gx_tilde, gy_tilde)

        # Step 5: Zernike projection (masked or unmasked for ablation)
        if self.unmasked_zernike:
            # Mask set to all ones
            phi_zernike = phi_base  # Fallback to base phase for unmasked ablation
            c_zernike = torch.zeros(
                I_raw.shape[0], 15, device=I_raw.device, dtype=I_raw.dtype
            )
        else:
            c_zernike, phi_zernike = self.zernike_proj(phi_base)

        # Step 6: CNN 2 predicts non-Zernike residual
        residual = self.residual_net(phi_base)

        phi_final = phi_zernike + residual
        return phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike
