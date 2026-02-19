"""
GradCAM visualization for UNetRes2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import load_train_config
from ..core.utils import pick_device
from ..data import build_dataloaders
from ..model import build_model
from ..visualize.style import DOUBLE_COL, nature_style, save_figure


class _GradCAM:
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


def plot_gradcam(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/gradcam.png",
    n_samples: int = 4,
    target_layer_name: str = "enc5",
    config_path: str | None = None,
) -> None:
    """
    Generate GradCAM overlay visualization (3-column: input, heatmap, overlay).

    Args:
        checkpoint_path:    Trained model checkpoint.
        data_dir:           Dataset directory.
        out_path:           Output figure path.
        n_samples:          Number of samples to visualize.
        target_layer_name:  Encoder stage: enc1..enc5, bott.
        config_path:        Optional config override.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    cfg.data.data_dir = data_dir
    cfg.data.augment = False

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
    model.final_dropout.eval()

    target_layer = getattr(model, target_layer_name, None)
    if target_layer is None:
        available = [n for n, _ in model.named_children()]
        raise ValueError(
            f"Layer '{target_layer_name}' not found. Available: {available}"
        )

    cam_extractor = _GradCAM(model, target_layer)

    _, val_loader = build_dataloaders(
        cfg.data, cfg.optim, device, seed=cfg.logging.seed
    )
    loader = (
        val_loader
        or build_dataloaders(cfg.data, cfg.optim, device, seed=cfg.logging.seed)[0]
    )

    I_input, phi_gt, I_raw = next(iter(loader))
    I_input = I_input[:n_samples].to(device).requires_grad_(True)
    I_raw = I_raw[:n_samples]

    cam, phi_abs = cam_extractor(I_input)

    cam_np = cam.cpu().numpy()
    I_raw_np = I_raw.numpy()

    with nature_style():
        fig, axes = plt.subplots(
            n_samples,
            3,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.3 * n_samples),
        )
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Interferogram", f"GradCAM ({target_layer_name})", "Overlay"]

        for row in range(n_samples):
            I_img = I_raw_np[row, 0]
            heatmap = cam_np[row, 0]

            I_disp = (I_img - I_img.min()) / (I_img.max() - I_img.min() + 1e-8)

            axes[row, 0].imshow(I_disp, cmap="gray", aspect="equal")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(heatmap, cmap="jet", aspect="equal", vmin=0, vmax=1)
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

            axes[row, 2].imshow(I_disp, cmap="gray", aspect="equal")
            axes[row, 2].imshow(
                heatmap, cmap="jet", alpha=0.4, aspect="equal", vmin=0, vmax=1
            )
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])

            for col in range(3):
                axes[row, col].spines[:].set_visible(False)

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=7, pad=4)

        fig.tight_layout(pad=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved GradCAM: {out_path}")
