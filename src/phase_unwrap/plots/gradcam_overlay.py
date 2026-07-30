from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from .style import DOUBLE_COL, nature_style, save_figure
from ..analysis.gradcam import GradCAM

def plot_gradcam(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/gradcam.png",
    n_samples: int = 4,
    target_layer_name: str = "enc5",
    config_path: str | None = None,
    subset: str = "val",
) -> None:
    from ..core.inference import load_inference_state
    from ..data.augmentation import prepare_batch

    model, cfg, loader, device = load_inference_state(
        checkpoint_path,
        data_dir=data_dir,
        config_path=config_path,
        subset=subset,
    )

    if not hasattr(model, target_layer_name):
        raise ValueError(f"Unknown layer {target_layer_name!r}")
    target_layer = getattr(model, target_layer_name)
    if isinstance(target_layer, torch.nn.Sequential) and len(target_layer) > 0:
        target_layer = target_layer[-1]
    cam_extractor = GradCAM(model, target_layer)

    I_raw, phi_gt = next(iter(loader))
    I_raw = I_raw[:n_samples].to(device)
    phi_gt = phi_gt[:n_samples].to(device)
    hint_mode = getattr(cfg.aug, "hint_mode", "zero")
    I_input, _, _, I_raw_c = prepare_batch(
        I_raw, phi_gt, noise_aug=None, hint_mode=hint_mode
    )
    I_input = I_input.requires_grad_(True)

    cam, phi_abs = cam_extractor(I_input)

    cam_np = cam.detach().cpu().numpy()
    I_raw_np = I_raw_c.detach().cpu().numpy()

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
