"""
UNetRes2 model for absolute phase reconstruction.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import ModelConfig


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
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

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

        # Split conceptually via list slice, replacing the sequential list append
        # with pre-allocated tensors or unrolled variables based on self.s scale (which is usually 4)
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

        # If scaling is strictly s=4 (ali_proj default), avoid looping entirely
        if self.s == 4:
            y2 = torch.cat([out_0, out_1, out_2, out_3], dim=1)
        else:
            # Fallback for non-standard configurations
            outs = [out_0, out_1, out_2, out_3]
            for i in range(4, self.s):
                cc0, cc1 = self.offsets[i], self.offsets[i] + self.sizes[i]
                outs.append(self.act(self.dw[i](y[:, cc0:cc1] + outs[-1])))
            y2 = torch.cat(outs, dim=1)

        y2 = self.bn2(self.pw(y2))

        if self.shortcut:
            x = self.proj(x)
        return self.act(x + y2)


class UpBlockRes2(nn.Module):
    """Bilinear upsample + skip concatenation + Res2 block."""

    def __init__(
        self,
        in_ch_cat: int,
        out_ch: int,
        s: int = 4,
        expansion: float = 1.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = Res2_DS_Block(in_ch_cat, out_ch, s, expansion, act)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNetRes2_AbsPhase(nn.Module):
    """
    UNet encoder-decoder with Res2 blocks and a global offset head.

    Input:  [B, 2, H, W]  (I_norm, phi_hint)
    Output: (phi_raw [B,1,H,W], k_off [B,1,1,1])
    """

    def __init__(
        self,
        in_ch: int = 2,
        base: int = 32,
        act: str = "relu",
        final_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.addcoords = AddCoords()

        def C(m: int) -> int:
            return int(min(base * m, 1024))

        self.enc1 = nn.Sequential(
            Res2_DS_Block(in_ch + 2, C(1), 4, 1.0, act),
            Res2_DS_Block(C(1), C(1), 4, 1.0, act),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), Res2_DS_Block(C(1), C(2), 4, 1.0, act)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), Res2_DS_Block(C(2), C(4), 4, 1.0, act)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2), Res2_DS_Block(C(4), C(8), 4, 1.0, act)
        )
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2), Res2_DS_Block(C(8), C(16), 4, 1.0, act)
        )
        self.bott = nn.Sequential(Res2_DS_Block(C(16), C(32), 4, 1.0, act))

        self.up4 = UpBlockRes2(C(32) + C(16), C(16), 4, 1.0, act)
        self.up3 = UpBlockRes2(C(16) + C(8), C(8), 4, 1.0, act)
        self.up2 = UpBlockRes2(C(8) + C(4), C(4), 4, 1.0, act)
        self.up1 = UpBlockRes2(C(4) + C(2), C(2), 4, 1.0, act)
        self.up0 = UpBlockRes2(C(2) + C(1), C(1), 4, 1.0, act)

        self.final_dropout = nn.Dropout2d(final_dropout)
        self.head_pix = nn.Conv2d(C(1), 1, 1)

        self.off_conv = nn.Conv2d(C(32), C(8), 1)
        self.off_act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.off_fc = nn.Conv2d(C(8), 1, 1)

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.addcoords(x_in)

        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bott(e5)

        d4 = self.up4(b, e5)
        d3 = self.up3(d4, e4)
        d2 = self.up2(d3, e3)
        d1 = self.up1(d2, e2)
        d0 = self.up0(d1, e1)

        d0 = self.final_dropout(d0)
        phi_raw = self.head_pix(d0)

        z = self.off_conv(b)
        z = self.off_act(z)
        k_off = self.off_fc(z)
        k_off = k_off.mean(dim=(2, 3), keepdim=True)

        return phi_raw, k_off


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


def build_model(cfg: ModelConfig) -> UNetRes2_AbsPhase:
    """Construct the model from configuration."""
    return UNetRes2_AbsPhase(
        in_ch=2,
        base=cfg.base,
        act=cfg.activation,
        final_dropout=cfg.final_dropout,
    )