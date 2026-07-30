"""
GradCAM visualization core logic for UNetRes2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

class GradCAM:
    """GradCAM with forward/backward hooks on any target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad()
        phi_raw, k_off = self.model(x)
        if isinstance(phi_raw, list):
            phi_abs = phi_raw[-1] + k_off
        else:
            phi_abs = phi_raw + k_off

        target = phi_abs.mean()
        target.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not fire. Check target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return cam.detach(), phi_abs.detach()
