from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_unwrap.config import ModelConfig


def meshgrid_ij(x: torch.Tensor, y: torch.Tensor, **kw):
    """
    Compatibility wrapper around torch.meshgrid for older PyTorch versions.
    """
    try:
        return torch.meshgrid(x, y, indexing="ij", **kw)
    except TypeError:
        return torch.meshgrid(x, y)


class AddCoords(nn.Module):
    """
    Adds normalized x, y coordinate channels to the input tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        yy, xx = meshgrid_ij(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
        )
        coords = torch.stack([xx, yy], 0).expand(b, -1, -1, -1)
        return torch.cat([x, coords], 1)


class Res2_DS_Block(nn.Module):
    """
    Res2-style block with depthwise splits.
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

        # Split mid channels into s groups
        self.sizes = []
        base = mid // self.s
        rem = mid - base * self.s
        for i in range(self.s):
            self.sizes.append(base + (1 if i < rem else 0))
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
        outs = []
        for i in range(self.s):
            c0, c1 = self.offsets[i], self.offsets[i] + self.sizes[i]
            xi = y[:, c0:c1]
            zi = self.act(self.dw[i](xi if i == 0 else (xi + outs[i - 1])))
            outs.append(zi)
        y2 = torch.cat(outs, 1)
        y2 = self.bn2(self.pw(y2))
        if self.shortcut:
            x = self.proj(x)
        return self.act(x + y2)


class UpBlockRes2(nn.Module):
    """Upsampling block with skip connection and Res2 block."""

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
    UNet-style encoder-decoder with Res2 blocks and a global offset head.

    Input:
        I_input [B, 2, H, W] = [I_norm, phi_hint]
        After AddCoords -> 4 channels going into the first conv.

    Outputs:
        - phi_raw   [B, 1, H, W]: local phase structure
        - a_pred    [B, 1, H, W]: optional / unused background term
        - b_pred_raw[B, 1, H, W]: amplitude (softplus if used)
        - conf_logit[B, 1, H, W]: confidence logit (sigmoid if used)
        - k_off     [B, 1, 1, 1]: global scalar offset to add to phi_raw
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
        C = lambda m: int(min(base * m, 1024))

        # encoder
        self.enc1 = nn.Sequential(
            Res2_DS_Block(in_ch + 2, C(1), 4, 1.0, act),
            Res2_DS_Block(C(1), C(1), 4, 1.0, act),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(1), C(2), 4, 1.0, act),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(2), C(4), 4, 1.0, act),
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(4), C(8), 4, 1.0, act),
        )
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(8), C(16), 4, 1.0, act),
        )

        # bottleneck
        self.bott = nn.Sequential(
            Res2_DS_Block(C(16), C(32), 4, 1.0, act),
        )

        # decoder
        self.up4 = UpBlockRes2(C(32) + C(16), C(16), 4, 1.0, act)
        self.up3 = UpBlockRes2(C(16) + C(8), C(8), 4, 1.0, act)
        self.up2 = UpBlockRes2(C(8) + C(4), C(4), 4, 1.0, act)
        self.up1 = UpBlockRes2(C(4) + C(2), C(2), 4, 1.0, act)
        self.up0 = UpBlockRes2(C(2) + C(1), C(1), 4, 1.0, act)

        self.final_dropout = nn.Dropout2d(final_dropout)

        # pixelwise head for [phi_raw, a_pred, b_raw, conf_logit]
        self.head_pix = nn.Conv2d(C(1), 4, 1)

        # global offset head k_off
        self.off_conv = nn.Conv2d(C(32), C(8), 1)
        self.off_act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.off_fc = nn.Conv2d(C(8), 1, 1)

    def forward(
        self, x_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_in: [B, 2, H, W]
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

        pix_out = self.head_pix(d0)
        phi_raw = pix_out[:, 0:1]
        a_pred = pix_out[:, 1:2]
        b_pred_raw = pix_out[:, 2:3]
        conf_logit = pix_out[:, 3:4]

        # global offset from bottleneck
        z = self.off_conv(b)
        z = self.off_act(z)
        k_off = self.off_fc(z)
        k_off = k_off.mean(dim=(2, 3), keepdim=True)

        return phi_raw, a_pred, b_pred_raw, conf_logit, k_off


class EMA:
    """
    Exponential moving average shadow model for evaluation and visualization.
    """

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
    """
    Construct the UNetRes2_AbsPhase model from configuration.
    """
    if cfg.model_type == "swin":
        from .model_swin import SwinUNet

        # Map config params to SwinUNet args if needed, or use defaults for now.
        # SwinUNet defaults are: embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]
        # We can map 'base' to 'embed_dim' roughly.
        return SwinUNet(
            embed_dim=cfg.base * 4,  # e.g. 16*4 = 64
            drop_rate=cfg.final_dropout,
            # using default depths/heads for now
        )

    return UNetRes2_AbsPhase(
        in_ch=2,
        base=cfg.base,
        act=cfg.activation,
        final_dropout=cfg.final_dropout,
    )
