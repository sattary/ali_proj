"""
Signal-processing operators for phase supervision.

Contains:
    - FixedSobel: depthwise Sobel gradient extraction.
    - laplacian: discrete 2D Laplacian with reflect padding.
    - curvature_loss: L1 Laplacian smoothness penalty.
    - affine_align: per-image least-squares affine fit (eval-only).
"""

from __future__ import annotations

import math
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


def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap phase values into [-pi, pi]."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))


class AnalyticSignalStem(nn.Module):
    """
    Deterministic Takeda 2D FFT off-axis demodulation stem.
    Extracts wrapped phase and amplitude from raw off-axis intensity.
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        f0: tuple[float, float] = (0.125, 0.125),
        bw: float = 0.08,
    ) -> None:
        super().__init__()
        self.f0 = f0
        self.bw = bw

        fy = torch.fft.fftfreq(height, d=1.0).view(1, 1, height, 1)
        fx = torch.fft.fftfreq(width, d=1.0).view(1, 1, 1, width)
        dist = torch.sqrt((fx - f0[1]) ** 2 + (fy - f0[0]) ** 2)
        mask = (dist <= bw).to(torch.float32)
        self.register_buffer("mask", mask)

    def forward(self, I_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I_off: [B, 1, H, W] off-axis intensity.

        Returns:
            wrapped_phase: [B, 1, H, W] in [-pi, pi].
            amplitude: [B, 1, H, W] normalized.
        """
        B, C, H, W = I_off.shape
        if H == self.mask.shape[2] and W == self.mask.shape[3]:
            mask = self.mask.to(dtype=I_off.dtype)
        else:
            fy = torch.fft.fftfreq(H, d=1.0).view(1, 1, H, 1).to(I_off.device)
            fx = torch.fft.fftfreq(W, d=1.0).view(1, 1, 1, W).to(I_off.device)
            dist = torch.sqrt((fx - self.f0[1]) ** 2 + (fy - self.f0[0]) ** 2)
            mask = (dist <= self.bw).to(I_off.dtype)

        F_I = torch.fft.fft2(I_off)
        F_filtered = F_I * mask

        # Inverse FFT to get baseband analytic signal
        analytic = torch.fft.ifft2(F_filtered)
        wrapped_phase = torch.atan2(analytic.imag, analytic.real)
        amplitude = analytic.abs()

        # Normalize amplitude per image
        amp_max = amplitude.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1) + 1e-6
        amplitude_norm = amplitude / amp_max

        return wrapped_phase, amplitude_norm


class WrappedGradientOperator(nn.Module):
    """Computes finite differences of wrapped phase and wraps them to [-pi, pi]."""

    def forward(self, wrapped_phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wrapped_phi: [B, 1, H, W]

        Returns:
            gx, gy: [B, 1, H, W] wrapped gradients.
        """
        gx = wrap_phase(torch.diff(wrapped_phi, dim=-1, prepend=wrapped_phi[..., :1]))
        gy = wrap_phase(torch.diff(wrapped_phi, dim=-2, prepend=wrapped_phi[..., :1, :]))
        return gx, gy


class DifferentiablePoissonSolver(nn.Module):
    """
    Exact 2D DCT-II Poisson Solver for Neumann boundary conditions.
    Integrates a vector gradient field (gx, gy) by solving nabla^2 phi = div(g).
    """

    def __init__(self, height: int = 128, width: int = 128) -> None:
        super().__init__()
        self.height = height
        self.width = width

        # Precompute 1D DCT-II orthogonal transformation matrices
        n_h = torch.arange(height, dtype=torch.float32)
        k_h = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        C_h = math.sqrt(2.0 / height) * torch.cos(math.pi * k_h * (2.0 * n_h + 1.0) / (2.0 * height))
        C_h[0, :] /= math.sqrt(2.0)

        n_w = torch.arange(width, dtype=torch.float32)
        k_w = torch.arange(width, dtype=torch.float32).unsqueeze(1)
        C_w = math.sqrt(2.0 / width) * torch.cos(math.pi * k_w * (2.0 * n_w + 1.0) / (2.0 * width))
        C_w[0, :] /= math.sqrt(2.0)

        # Precompute Discrete Laplacian eigenvalues: 2*cos(pi*k/H) + 2*cos(pi*l/W) - 4
        k_grid = torch.arange(height, dtype=torch.float32).view(height, 1)
        l_grid = torch.arange(width, dtype=torch.float32).view(1, width)
        denom = 2.0 * torch.cos(math.pi * k_grid / height) + 2.0 * torch.cos(math.pi * l_grid / width) - 4.0
        denom[0, 0] = 1.0  # Avoid div-by-zero for DC (piston mode)

        self.register_buffer("C_h", C_h)
        self.register_buffer("C_w", C_w)
        self.register_buffer("denom", denom.view(1, 1, height, width))

    def forward(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gx: [B, 1, H, W] x-component gradient.
            gy: [B, 1, H, W] y-component gradient.

        Returns:
            phi_base: [B, 1, H, W] integrated phase map (least-squares solution).
        """
        B, C, H, W = gx.shape
        div_x = gx - F.pad(gx[..., :-1], (1, 0))
        div_y = gy - F.pad(gy[..., :-1, :], (0, 0, 1, 0))
        rho = div_x + div_y

        if H == self.height and W == self.width:
            C_h, C_w, denom = self.C_h.to(gx.dtype), self.C_w.to(gx.dtype), self.denom.to(gx.dtype)
        else:
            n_h = torch.arange(H, dtype=gx.dtype, device=gx.device)
            k_h = torch.arange(H, dtype=gx.dtype, device=gx.device).unsqueeze(1)
            C_h = math.sqrt(2.0 / H) * torch.cos(math.pi * k_h * (2.0 * n_h + 1.0) / (2.0 * H))
            C_h[0, :] /= math.sqrt(2.0)

            n_w = torch.arange(W, dtype=gx.dtype, device=gx.device)
            k_w = torch.arange(W, dtype=gx.dtype, device=gx.device).unsqueeze(1)
            C_w = math.sqrt(2.0 / W) * torch.cos(math.pi * k_w * (2.0 * n_w + 1.0) / (2.0 * W))
            C_w[0, :] /= math.sqrt(2.0)

            k_grid = torch.arange(H, dtype=gx.dtype, device=gx.device).view(H, 1)
            l_grid = torch.arange(W, dtype=gx.dtype, device=gx.device).view(1, W)
            denom = 2.0 * torch.cos(math.pi * k_grid / H) + 2.0 * torch.cos(math.pi * l_grid / W) - 4.0
            denom[0, 0] = 1.0
            denom = denom.view(1, 1, H, W)

        # 2D DCT-II: C_h @ rho @ C_w^T
        rho_dct = torch.matmul(torch.matmul(C_h, rho), C_w.t())

        # Solve in spectral domain
        phi_dct = rho_dct / denom
        phi_dct[:, :, 0, 0] = 0.0  # Set DC / piston to zero

        # 2D IDCT-II: C_h^T @ phi_dct @ C_w
        phi_base = torch.matmul(torch.matmul(C_h.t(), phi_dct), C_w)
        return phi_base


class MaskedZernikeProjection(nn.Module):
    """
    Disk-masked discrete linear Zernike basis projection.
    Projects input phase onto the first 15 Zernike modes restricted to the unit disk.
    """

    def __init__(self, height: int = 128, width: int = 128, num_modes: int = 15) -> None:
        super().__init__()
        self.num_modes = num_modes

        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rho = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        mask = (rho <= 1.0)

        Z = torch.zeros((num_modes, height, width), dtype=torch.float32)
        Z[0] = 1.0
        Z[1] = 2.0 * rho * torch.cos(theta)
        Z[2] = 2.0 * rho * torch.sin(theta)
        Z[3] = math.sqrt(3.0) * (2.0 * rho**2 - 1.0)
        Z[4] = math.sqrt(6.0) * rho**2 * torch.sin(2.0 * theta)
        Z[5] = math.sqrt(6.0) * rho**2 * torch.cos(2.0 * theta)
        Z[6] = math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * torch.sin(theta)
        Z[7] = math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * torch.cos(theta)
        Z[8] = math.sqrt(8.0) * rho**3 * torch.sin(3.0 * theta)
        Z[9] = math.sqrt(8.0) * rho**3 * torch.cos(3.0 * theta)
        Z[10] = math.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)
        Z[11] = math.sqrt(10.0) * (4.0 * rho**4 - 3.0 * rho**2) * torch.cos(2.0 * theta)
        Z[12] = math.sqrt(10.0) * (4.0 * rho**4 - 3.0 * rho**2) * torch.sin(2.0 * theta)
        Z[13] = math.sqrt(10.0) * rho**4 * torch.cos(4.0 * theta)
        Z[14] = math.sqrt(10.0) * rho**4 * torch.sin(4.0 * theta)

        for i in range(num_modes):
            Z[i] *= mask.to(torch.float32)

        # Flatten masked design matrix: A [N_mask, num_modes]
        mask_flat = mask.view(-1)
        A = Z.view(num_modes, -1)[:, mask_flat].t()  # [N_mask, 15]

        # Precompute pseudoinverse (A^T A)^(-1) A^T for fast least-squares solve
        A_pinv = torch.linalg.pinv(A)  # [15, N_mask]

        self.register_buffer("mask", mask.view(1, 1, height, width))
        self.register_buffer("Z_basis", Z.view(1, num_modes, height, width))
        self.register_buffer("mask_flat", mask_flat)
        self.register_buffer("A_pinv", A_pinv)

    def forward(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phi: [B, 1, H, W] phase map.

        Returns:
            coefficients: [B, num_modes] Zernike coefficients.
            phi_zernike: [B, 1, H, W] reconstructed Zernike surface.
        """
        B = phi.shape[0]
        phi_masked_flat = phi.view(B, -1)[:, self.mask_flat].t()  # [N_mask, B]
        c = torch.matmul(self.A_pinv, phi_masked_flat).t()  # [B, num_modes]

        # Reconstruct surface: sum_j c_j * Z_j
        phi_zernike = torch.sum(c.view(B, self.num_modes, 1, 1) * self.Z_basis, dim=1, keepdim=True)
        return c, phi_zernike

