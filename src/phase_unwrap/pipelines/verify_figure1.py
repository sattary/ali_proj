import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from phase_unwrap.core.models.unet import build_model
from phase_unwrap.core.ops import affine_align, get_residue_mask
from phase_unwrap.config import TrainConfig
import h5py
from phase_unwrap.core.data import _to_chw


def generate_figure1(
    ckpt_path: str,
    data_path: str,
    out_path: str = "figure1_composite.png",
    device: str = "cpu",
):
    print(f"Generating Figure 1 from {ckpt_path} on {data_path}...")

    # 1. Load Model
    # We need a dummy config to build the model structure
    # In a real scenario we'd load config.json from the run dir
    from phase_unwrap.config import ModelConfig

    model_cfg = ModelConfig(
        model_type="unet", base=16
    )  # adjust if your checkpoint differs
    model = build_model(model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_ema") or ckpt.get("model")
    model.load_state_dict(state)
    model.eval()

    # 2. Load One Sample
    with h5py.File(data_path, "r") as f:
        # Pick a sample with some complexity (e.g. index 0)
        I_np = f["I"][0]
        phi_gt_np = f["phi"][0]

    I_np = _to_chw(I_np)
    phi_gt_np = _to_chw(phi_gt_np)

    I_raw_t = torch.from_numpy(I_np).float().unsqueeze(0).to(device)  # [1, 1, H, W]
    phi_gt_t = torch.from_numpy(phi_gt_np).float().unsqueeze(0).to(device)

    # Normalize I
    mean = I_raw_t.mean(dim=(2, 3), keepdim=True)
    std = I_raw_t.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    I_norm_t = (I_raw_t - mean) / std

    # create hint
    _, _, H, W = phi_gt_t.shape
    cy, cx = H // 2, W // 2
    ref_val = float(phi_gt_t[0, 0, cy, cx].item())
    phi_hint = torch.full_like(phi_gt_t, ref_val)

    I_input = torch.cat([I_norm_t, phi_hint], dim=1)  # [1, 2, H, W]

    # 3. Inference
    with torch.no_grad():
        phi_raw, _, _, res_logit, k_off = model(I_input)
        phi_pred = phi_raw + k_off

        # Align
        phi_pred_aligned, _, _ = affine_align(phi_pred, phi_gt_t)

    # 4. Compute Residues & Error
    residue_gt = get_residue_mask(phi_gt_t)
    residue_pred = torch.sigmoid(res_logit)  # Probability map

    error_map = torch.abs(phi_pred_aligned - phi_gt_t)

    # 5. Plotting (The "Figure 1")
    # Layout:
    # [Wrapped Input] [GT Phase] [Pred Phase] [Residue GT] [Residue Pred] [Error Map]

    # Convert to numpy
    def to_np(t):
        return t[0, 0].cpu().numpy()

    img_wrapped = np.angle(
        np.exp(1j * to_np(phi_gt_t))
    )  # Re-wrap GT to show input complexity
    img_gt = to_np(phi_gt_t)
    img_pred = to_np(phi_pred_aligned)
    img_res_gt = to_np(residue_gt)
    img_res_pred = to_np(residue_pred)
    img_error = to_np(error_map)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    ax = axes[0]
    im = ax.imshow(img_wrapped, cmap="hsv")
    ax.set_title("Input (Wrapped)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(img_gt, cmap="jet")
    ax.set_title("Ground Truth")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im = ax.imshow(img_pred, cmap="jet")
    ax.set_title("Swin-UNet Prediction")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[3]
    im = ax.imshow(img_res_gt, cmap="gray")
    ax.set_title("Residues (GT)")

    ax = axes[4]
    im = ax.imshow(img_res_pred, cmap="magma")
    ax.set_title("Residues (Pred Prob)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[5]
    im = ax.imshow(img_error, cmap="inferno")
    ax.set_title(f"Error (MAE={np.mean(img_error):.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved Figure 1 to {out_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage:
    # Adjust paths as needed
    ckpt = "runs/affine_align/best.pth"
    data = "data/synthetic_h5_test/train_shard_000.h5"

    if os.path.exists(ckpt) and os.path.exists(data):
        generate_figure1(ckpt, data)
    else:
        print(f"Skipping run: Checkpoint {ckpt} or Data {data} not found.")
