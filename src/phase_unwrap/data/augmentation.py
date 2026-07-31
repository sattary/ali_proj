"""Physically ordered, explicitly seeded camera/noise augmentation."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_intensity(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


class NoiseAug(nn.Module):
    def __init__(
        self,
        gauss_std=0.02,
        speckle_std=0.05,
        lowfreq_amp=0.05,
        blur_prob=0.2,
        blur_sigma=(0.5, 1.0),
        dropout_prob=0.02,
        s_and_p_prob=0.0,
        gain_jitter=(0.9, 1.1),
        offset_jitter=(-0.05, 0.05),
        photon_min=256.0,
        photon_max=4096.0,
        sensor_min=0.0,
        sensor_max=4.0,
        enable=True,
    ):
        super().__init__()
        self.gauss_std, self.speckle_std, self.lowfreq_amp = (
            gauss_std,
            speckle_std,
            lowfreq_amp,
        )
        self.blur_prob, self.blur_sigma = blur_prob, blur_sigma
        self.dropout_prob, self.sap_prob = dropout_prob, s_and_p_prob
        self.gain_min, self.gain_max = gain_jitter
        self.off_min, self.off_max = offset_jitter
        self.photon_min, self.photon_max = photon_min, photon_max
        self.sensor_min, self.sensor_max, self.enable = sensor_min, sensor_max, enable

    @staticmethod
    def _blur(x, sigma):
        if sigma <= 0:
            return x
        n = max(3, int(round(6 * sigma)) // 2 * 2 + 1)
        t = torch.arange(n, device=x.device, dtype=x.dtype) - (n - 1) / 2
        g = torch.exp(-0.5 * (t / sigma).square())
        g = g / g.sum()
        x = F.pad(x, (n // 2,) * 4, mode="reflect")
        x = F.conv2d(x, g.view(1, 1, 1, n))
        return F.conv2d(x, g.view(1, 1, n, 1))

    def forward(self, I_clean, severity, *, generator=None):
        if generator is None:
            generator = torch.Generator(device=I_clean.device)
            generator.manual_seed(0)
        level = float(max(0.0, min(1.0, float(severity))))
        if not self.enable or level == 0:
            return I_clean.clone(), normalize_intensity(I_clean)
        b, _, h, w = I_clean.shape
        dev, dt = I_clean.device, I_clean.dtype
        rand = lambda *shape: torch.rand(
            *shape, device=dev, dtype=dt, generator=generator
        )
        normal = lambda *shape: torch.randn(
            *shape, device=dev, dtype=dt, generator=generator
        )
        img = I_clean.clone()
        # 1. nonnegative multiplicative speckle (unit-mean exponential mixture)
        if self.speckle_std:
            q = max(1e-3, self.speckle_std * level)
            img = img * ((1.0 - q) + q * (-torch.log(rand(b, 1, h, w).clamp_min(1e-6))))
        # 2. optical blur
        if float(rand(1)) < self.blur_prob * level:
            sigma = float(
                torch.empty((), device=dev).uniform_(
                    *self.blur_sigma, generator=generator
                )
            )
            img = self._blur(img, sigma)
        # 3. low-frequency illumination/background
        if self.lowfreq_amp:
            lf = F.interpolate(
                normal(b, 1, 4, 4), size=(h, w), mode="bicubic", align_corners=False
            )
            lf = (
                lf
                / lf.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                * self.lowfreq_amp
                * level
            )
            img = img + lf
        # 4. gain/black level
        gain = torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(
            1 + (self.gain_min - 1) * level,
            1 + (self.gain_max - 1) * level,
            generator=generator,
        )
        off = torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(
            self.off_min * level, self.off_max * level, generator=generator
        )
        img = gain * img + off
        # 5. Poisson shot noise: severity means fewer photons
        photons = self.photon_max + (self.photon_min - self.photon_max) * level
        rates = (
            img.clamp_min(self.sensor_min) / max(self.sensor_max, 1e-6) * photons
        ).clamp_min(0)
        img = torch.poisson(rates, generator=generator) / photons * self.sensor_max
        # 6. Gaussian read noise
        img = img + normal(b, 1, h, w) * (self.gauss_std * level * self.sensor_max)
        # 7. saturation/quantization and defective pixels
        img = img.clamp(self.sensor_min, self.sensor_max)
        if self.dropout_prob * level:
            img = torch.where(
                rand(b, 1, h, w) < self.dropout_prob * level, torch.zeros_like(img), img
            )
        if self.sap_prob * level:
            q = rand(b, 1, h, w)
            img = torch.where(q < self.sap_prob * level / 2, self.sensor_min, img)
            img = torch.where(q > 1 - self.sap_prob * level / 2, self.sensor_max, img)
        img = torch.round(img / self.sensor_max * 255) / 255 * self.sensor_max
        return img, normalize_intensity(img)
