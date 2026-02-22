"""
Optical curriculum noise augmentation module.

Faithfully implements physical interferogram degradations (multiplicative speckle,
additive Gaussian noise, low-frequency biases, defocus blur, etc.) with a
scalar curriculum level schedule synced to the training epochs.

Rationale: ARCHITECTURE
    Isolating this stochastic math inside `data/` instead of `training/train.py`
    allows the DataLoader's multiprocessing CPU threadpool to calculate the
    noise overlays asynchronously, avoiding a massive GPU serialization bottleneck.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F


class NoiseScheduler:
    """
    Piecewise noise level schedule over epochs.
    warmup_epochs: noise=0 during [1 .. warmup]
    full_epoch:    reaches noise=1 by this epoch
    profile: 'linear' or 'cosine'
    """

    def __init__(
        self, warmup_epochs: int = 0, full_epoch: int = 20, profile: str = "cosine"
    ):
        self.warmup = max(0, int(warmup_epochs))
        self.full = max(1, int(full_epoch))
        self.profile = profile

    def level(self, epoch: int) -> float:
        """Calculate the normalized [0, 1] noise strength for the given epoch."""
        if epoch <= self.warmup:
            return 0.0
        if epoch >= self.full:
            return 1.0
        t = (epoch - self.warmup) / max(1, (self.full - self.warmup))
        if self.profile == "cosine":
            return 0.5 * (1 - np.cos(np.pi * t))
        return float(t)  # linear


class NoiseAug:
    """
    Input-only augmentation with a schedulable 'level' in [0,1].
    At level=0 -> no noise; at level=1 -> full configured strength.
    Acts on I_raw and I_norm, optionally perturbs phi_hint by a global offset.
    """

    def __init__(
        self,
        gauss_std: float = 0.02,
        speckle_std: float = 0.05,
        poisson_scale: float = 0.0,
        lowfreq_amp: float = 0.05,
        lowfreq_sigma: int = 21,
        blur_prob: float = 0.2,
        blur_sigma: tuple[float, float] = (0.5, 1.0),
        dropout_prob: float = 0.02,
        s_and_p_prob: float = 0.0,
        gain_jitter: tuple[float, float] = (0.9, 1.1),
        offset_jitter: tuple[float, float] = (-0.05, 0.05),
        hint_offset_std: float = 0.2,
        enable: bool = True,
    ):
        self.base = dict(
            gauss_std=gauss_std,
            speckle_std=speckle_std,
            poisson_scale=poisson_scale,
            lowfreq_amp=lowfreq_amp,
            lowfreq_sigma=int(max(3, lowfreq_sigma // 2 * 2 + 1)),
            blur_prob=blur_prob,
            blur_sigma=blur_sigma,
            dropout_prob=dropout_prob,
            s_and_p_prob=s_and_p_prob,
            gain_min=gain_jitter[0],
            gain_max=gain_jitter[1],
            off_min=offset_jitter[0],
            off_max=offset_jitter[1],
            hint_offset_std=hint_offset_std,
        )
        self.enable = enable
        self.level = 0.0  # start silent

    def set_level(self, level: float) -> None:
        """Manually push the curriculum strength."""
        self.level = float(min(1.0, max(0.0, level)))

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        ksize = int(max(3, round(6 * sigma) // 2 * 2 + 1))
        t = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = g / g.sum()
        g1 = g.view(1, 1, 1, ksize)
        g2 = g.view(1, 1, ksize, 1)
        x = F.pad(x, (ksize // 2,) * 4, mode="reflect")
        x = F.conv2d(x, g1, groups=1)
        x = F.conv2d(x, g2, groups=1)
        return x

    def __call__(
        self,
        I_raw: torch.Tensor,
        I_norm: torch.Tensor,
        phi_hint: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply noise mathematically, modifying Raw/Norm tensors gracefully."""
        if not self.enable or self.level <= 0:
            return I_raw, I_norm, phi_hint

        L = self.level
        p = self.base
        device = I_raw.device
        dtype = I_raw.dtype
        _, H, W = I_raw.shape

        # Level-scaled magnitudes / probabilities
        gauss_std = p["gauss_std"] * L
        speckle_std = p["speckle_std"] * L
        poisson_sc = p["poisson_scale"] * L
        lowfreq_amp = p["lowfreq_amp"] * L
        blur_prob = p["blur_prob"] * L
        dropout_p = p["dropout_prob"] * L
        sap_p = p["s_and_p_prob"] * L
        hint_std = p["hint_offset_std"] * L

        # Photometric jitter interpolate to identity at L=0
        gain_min = 1.0 + (p["gain_min"] - 1.0) * L  # type: ignore
        gain_max = 1.0 + (p["gain_max"] - 1.0) * L  # type: ignore
        off_min = 0.0 + p["off_min"] * L  # type: ignore
        off_max = 0.0 + p["off_max"] * L  # type: ignore

        # Gain/offset jitter
        g = torch.empty(1, device=device).uniform_(gain_min, gain_max).item()
        b = torch.empty(1, device=device).uniform_(off_min, off_max).item()
        img: torch.Tensor = float(g) * I_raw + float(b)

        # Additive Gaussian (scaled by per-image std)
        if gauss_std > 0:
            stdI = img.std().clamp_min(1e-6)
            img = img + torch.randn_like(img) * (gauss_std * float(stdI))

        # Speckle (multiplicative)
        if speckle_std > 0:
            img = img * (1.0 + torch.randn_like(img) * speckle_std)  # type: ignore

        # Poisson (optional)
        if poisson_sc > 0:
            lam = (img - img.min()) / (img.max() - img.min() + 1e-6) * poisson_sc
            lam = lam.clamp_min(0.0)
            img = torch.poisson(lam) / (poisson_sc + 1e-6)
            img = (img - img.mean()) / (img.std() + 1e-6)

        # Low-frequency bias
        if lowfreq_amp > 0:
            lf = torch.randn(1, 1, H, W, device=device, dtype=dtype)
            lf = self._gaussian_blur(lf, p["lowfreq_sigma"] / 6.0)[0]  # type: ignore
            lf = lf / (lf.std() + 1e-6) * lowfreq_amp
            img = img + lf

        # Random blur
        if bool(torch.rand(1, device=device) < blur_prob):
            smin, smax = p["blur_sigma"]  # type: ignore
            sigma = float(torch.empty(1, device=device).uniform_(smin, smax))
            img = self._gaussian_blur(img.unsqueeze(0), sigma)[0]

        # Pixel dropout
        if dropout_p > 0:
            mask = (torch.rand_like(img) > dropout_p).float()
            img = img * mask

        # Salt & pepper
        if sap_p > 0:
            r = torch.rand_like(img)
            img = torch.where(r < sap_p / 2, img.min(), img)  # type: ignore
            img = torch.where(r > 1 - sap_p / 2, img.max(), img)  # type: ignore

        # Corrupt the hint with a global radian shift
        if phi_hint is not None and hint_std > 0:
            delta = torch.randn(1, device=device, dtype=dtype) * hint_std
            phi_hint = phi_hint + delta.view(1, 1, 1)

        # Re-normalize for network
        mean_final = img.mean(dim=(1, 2), keepdim=True)
        std_final = img.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        I_norm_noisy = (img - mean_final) / std_final

        return img, I_norm_noisy, phi_hint