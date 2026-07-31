"""
Evaluation script: Generate 10 synthetic samples and test `runs/exp_primary/best.pth`.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from phase_unwrap.core.config import load_train_config
from phase_unwrap.core.losses import compute_metrics
from phase_unwrap.core.ops import FixedSobel, piston_align
from phase_unwrap.core.utils import pick_device, set_seed
from phase_unwrap.data.augmentation import prepare_batch
from phase_unwrap.data.generate import _build_grid, generate_sample
from phase_unwrap.model import build_model


def main() -> None:
    set_seed(42)
    try:
        device = pick_device("auto")
    except Exception:
        device = torch.device("cpu")

    checkpoint_path = "runs/exp_primary/best.pth"
    config_path = "runs/exp_primary/config.yaml"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[eval] Loading config: {config_path}")
    cfg = load_train_config(config_path)

    # Force CPU if CUDA CC is unsupported on local machine
    if device.type == "cuda":
        try:
            torch.cuda.init()
        except Exception:
            device = torch.device("cpu")

    print(f"[eval] Building model on device={device}...")
    model = build_model(cfg.model).to(device)

    print(f"[eval] Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("ema_state_dict", ckpt.get("model", ckpt.get("model_state_dict", ckpt))))
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.eval()

    # 1. Generate 10 synthetic test samples
    print("[eval] Generating 10 synthetic interferogram samples (seed=42)...")
    rng = np.random.default_rng(42)
    x, y, r2 = _build_grid()

    I_samples = []
    phi_samples = []
    for _ in range(10):
        I_img, phi_img = generate_sample(x, y, r2, rng)
        I_samples.append(I_img)
        phi_samples.append(phi_img)

    I_raw_t = torch.from_numpy(np.stack(I_samples)[:, None, :, :]).to(device=device, dtype=torch.float32)
    phi_gt_t = torch.from_numpy(np.stack(phi_samples)[:, None, :, :]).to(device=device, dtype=torch.float32)

    # 2. Preprocess with Option B zero hint
    I_input, phi_gt_t, _, _ = prepare_batch(I_raw_t, phi_gt_t, noise_aug=None, hint_mode="zero")
    I_input = I_input.contiguous(memory_format=torch.channels_last)

    # 3. Model Inference
    sobel = FixedSobel().to(device)
    print("[eval] Running forward pass...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
            phi_raw, k_off = model(I_input)
            phi_abs = (phi_raw + k_off).float()

    phi_aligned, _ = piston_align(phi_abs, phi_gt_t)
    phi_aligned = phi_aligned.float()

    # 4. Compute Metrics
    print("\n" + "=" * 65)
    print(f"{'Sample':<8} | {'AbsMAE (rad)':<12} | {'TopoMAE (rad)':<12} | {'SSIM':<8} | {'PSNR (dB)':<10}")
    print("-" * 65)

    abs_maes, topo_maes, ssims, psnrs = [], [], [], []

    out_dir = Path("results/figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(10, 4, figsize=(16, 30))

    for i in range(10):
        p_abs = phi_abs[i:i+1]
        p_align = phi_aligned[i:i+1]
        p_gt = phi_gt_t[i:i+1]

        m_abs = compute_metrics(p_abs, p_gt)
        m_topo = compute_metrics(p_align, p_gt)

        gt_min = p_gt.amin(dim=(1, 2, 3), keepdim=True)
        gt_max = p_gt.amax(dim=(1, 2, 3), keepdim=True)
        dr = (gt_max - gt_min).clamp_min(1e-6)

        pred_norm = (p_align - gt_min) / dr
        gt_norm = (p_gt - gt_min) / dr
        ssim_val = float(ssim_fn(pred_norm, gt_norm, data_range=1.0).item())

        mse = ((p_align - p_gt) ** 2).mean()
        psnr_val = float((10.0 * torch.log10((dr.mean() ** 2) / mse.clamp_min(1e-12))).item())

        amae = float(m_abs["MAE"].item())
        tmae = float(m_topo["MAE"].item())

        abs_maes.append(amae)
        topo_maes.append(tmae)
        ssims.append(ssim_val)
        psnrs.append(psnr_val)

        print(f"{i+1:<8} | {amae:<12.4f} | {tmae:<12.4f} | {ssim_val:<8.4f} | {psnr_val:<10.2f}")

        # Visualization row
        I_img = I_raw_t[i, 0].cpu().numpy()
        gt_img = phi_gt_t[i, 0].cpu().numpy()
        pred_img = phi_abs[i, 0].cpu().numpy()
        err_img = (phi_abs[i, 0] - phi_gt_t[i, 0]).cpu().numpy()

        axs[i, 0].imshow(I_img, cmap="gray")
        axs[i, 0].set_title(f"Sample {i+1}: Intensity I")
        axs[i, 0].axis("off")

        im1 = axs[i, 1].imshow(gt_img, cmap="viridis")
        axs[i, 1].set_title("GT Unwrapped φ")
        axs[i, 1].axis("off")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.046)

        im2 = axs[i, 2].imshow(pred_img, cmap="viridis")
        axs[i, 2].set_title(f"Pred φ_abs (MAE={amae:.2f}rad)")
        axs[i, 2].axis("off")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.046)

        emax = np.percentile(np.abs(err_img), 98)
        im3 = axs[i, 3].imshow(err_img, cmap="PuOr", vmin=-emax, vmax=emax)
        axs[i, 3].set_title(f"Error Map (SSIM={ssim_val:.3f})")
        axs[i, 3].axis("off")
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.046)

    plt.tight_layout()
    plot_path = out_dir / "eval_10_samples.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("=" * 65)
    print(f"AVERAGE  | {np.mean(abs_maes):<12.4f} | {np.mean(topo_maes):<12.4f} | {np.mean(ssims):<8.4f} | {np.mean(psnrs):<10.2f}")
    print("=" * 65)
    print(f"\n[eval] Saved 10-sample evaluation plot to: {plot_path}\n")


if __name__ == "__main__":
    main()
