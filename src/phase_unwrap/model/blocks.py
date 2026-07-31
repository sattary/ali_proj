"""
Building blocks for UNetRes2 and neural network architectures.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def meshgrid_ij(x: torch.Tensor, y: torch.Tensor, **kw):
    """Compatibility wrapper for older PyTorch meshgrid API."""
    try:
        return torch.meshgrid(x, y, indexing="ij", **kw)
    except TypeError:
        return torch.meshgrid(x, y)


class AddCoords(nn.Module):
    """Concatenate normalized (x, y) coordinate channels to the input."""

    def __init__(self, h: int = 128, w: int = 128) -> None:
        super().__init__()
        # Precompute the normalized grid and register it to the module's device state
        yy, xx = meshgrid_ij(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32),
        )
        self.register_buffer("coords", torch.stack([xx, yy], dim=0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Dynamic fallback (should never trigger in ali_proj, but covers edge cases)
        if h != self.coords.shape[2] or w != self.coords.shape[3]:  # type: ignore
            yy, xx = meshgrid_ij(
                torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            )
            dyn_coords = torch.stack([xx, yy], 0).unsqueeze(0).expand(b, -1, -1, -1)
            return torch.cat([x, dyn_coords], dim=1)

        # Standard fast-path loop (zero allocations)
        batched_coords = self.coords.expand(b, -1, -1, -1)  # type: ignore
        return torch.cat([x, batched_coords], dim=1)


class Res2_DS_Block(nn.Module):
    """
    Res2-style residual block with depthwise-separated channel groups.

    1x1 expand -> split into ``s`` groups -> per-group DW 3x3 with
    hierarchical residual connections -> concat -> 1x1 project + shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        s: int = 4,
        expansion: float = 1.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        mid = max(1, int(out_ch * expansion))
        self.s = max(2, s)

        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "mish":
            self.act = nn.Mish(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

        base = mid // self.s
        rem = mid - base * self.s
        self.sizes = [base + (1 if i < rem else 0) for i in range(self.s)]
        self.offsets = [sum(self.sizes[:i]) for i in range(self.s)]

        self.dw = nn.ModuleList(
            [
                nn.Conv2d(
                    self.sizes[i],
                    self.sizes[i],
                    3,
                    padding=1,
                    groups=self.sizes[i],
                    bias=False,
                )
                for i in range(self.s)
            ]
        )

        self.pw = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = in_ch != out_ch
        if self.shortcut:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))

        c0 = self.offsets[0]
        c1 = c0 + self.sizes[0]
        out_0 = self.act(self.dw[0](y[:, c0:c1]))

        c0 = self.offsets[1]
        c1 = c0 + self.sizes[1]
        out_1 = self.act(self.dw[1](y[:, c0:c1] + out_0))

        c0 = self.offsets[2]
        c1 = c0 + self.sizes[2]
        out_2 = self.act(self.dw[2](y[:, c0:c1] + out_1))

        c0 = self.offsets[3]
        c1 = c0 + self.sizes[3]
        out_3 = self.act(self.dw[3](y[:, c0:c1] + out_2))

        if self.s == 4:
            y2 = torch.cat([out_0, out_1, out_2, out_3], dim=1)
        else:
            outs = [out_0, out_1, out_2, out_3]
            for i in range(4, self.s):
                cc0, cc1 = self.offsets[i], self.offsets[i] + self.sizes[i]
                outs.append(self.act(self.dw[i](y[:, cc0:cc1] + outs[-1])))
            y2 = torch.cat(outs, dim=1)

        y2 = self.bn2(self.pw(y2))

        if self.shortcut:
            x = self.proj(x)
        return self.act(x + y2)


class EMA:
    """Exponential moving average shadow model."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.m = deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for p_ema, p in zip(self.m.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1 - d)
        for b_ema, b in zip(self.m.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)
