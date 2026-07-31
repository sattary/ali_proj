"""Reproducible, MATLAB-equivalent on-the-fly interferogram simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SimulationBatch:
    I_clean: torch.Tensor
    phi_gt: torch.Tensor
    wrapped_dp: torch.Tensor
    piston_gt: torch.Tensor
    grad_phi2: torch.Tensor
    sample_ids: torch.Tensor


class MatlabSimulator(nn.Module):
    """Professor generator; all phase math is performed in float64."""

    def __init__(self, height: int = 128, width: int = 128) -> None:
        super().__init__()
        x = torch.linspace(-1e-3, 1e-3, width, dtype=torch.float64)
        y = torch.linspace(-1e-3, 1e-3, height, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        r2 = torch.sqrt(xx.square() + yy.square() + 0.5**2)
        self.register_buffer("x", xx)
        self.register_buffer("y", yy)
        self.register_buffer("r2", r2)
        self.register_buffer(
            "k", torch.tensor(2.0 * math.pi / 632.8e-9, dtype=torch.float64)
        )

    @staticmethod
    def _resize_deviation(v: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # MATLAB imresize(..., 'bicubic') equivalent separable bicubic path.
        return F.interpolate(
            v[None, None], size=(height, width), mode="bicubic", align_corners=False
        )[0, 0]

    def sample(
        self, batch_size: int, *, generator: torch.Generator, first_sample_id: int = 0
    ) -> SimulationBatch:
        device = self.x.device
        h, w = self.x.shape
        clean, target, wrapped, pistons, grads = [], [], [], [], []
        with torch.autocast(device_type=device.type, enabled=False):
            x, y, r2, k = (
                t.to(device=device, dtype=torch.float64)
                for t in (self.x, self.y, self.r2, self.k)
            )
            for _ in range(batch_size):
                u = torch.rand(
                    3, generator=generator, device=device, dtype=torch.float64
                )
                r1 = torch.sqrt(
                    u[0] * 80.0 + x / 50.0 * (u[1] - 0.5) + y / 50.0 * (u[2] - 0.5)
                )
                phi1 = k * r1
                n = int(
                    torch.randint(1, 4, (), generator=generator, device=device).item()
                )
                deviation = (
                    torch.rand(
                        n, generator=generator, device=device, dtype=torch.float64
                    )
                    * 20.0
                )
                phi1 = phi1 + self._resize_deviation(deviation, h, w)
                alpha = (
                    torch.rand(
                        (), generator=generator, device=device, dtype=torch.float64
                    )
                    - 0.5
                )
                phi2 = k * r2 * alpha
                dp = phi1 - phi2
                piston = dp.amin()
                dphi = dp - piston
                clean.append(2.0 + 2.0 * torch.cos(dp))
                target.append(dphi)
                wrapped.append(torch.remainder(dp + math.pi, 2.0 * math.pi) - math.pi)
                pistons.append(piston)
                grads.append(torch.stack((k * alpha * x / r2, k * alpha * y / r2)))
        return SimulationBatch(
            torch.stack(clean)[:, None].float(),
            torch.stack(target)[:, None].float(),
            torch.stack(wrapped)[:, None].float(),
            torch.stack(pistons)[:, None, None, None],
            torch.stack(grads).float(),
            torch.arange(
                first_sample_id,
                first_sample_id + batch_size,
                device=device,
                dtype=torch.long,
            ),
        )
