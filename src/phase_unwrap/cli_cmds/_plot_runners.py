"""
Plot execution wrappers for loading checkpoints and rendering paper figures.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from phase_unwrap.core.inference import load_inference_state
from phase_unwrap.core.ops import piston_align
from phase_unwrap.data import build_dataloaders
from phase_unwrap.plots.fig1_architecture import plot_f1_architecture
from phase_unwrap.plots.fig2_baseline_comparison import plot_f2_baseline_comparison
from phase_unwrap.plots.fig5_diagnostics import plot_f5_diagnostics
from phase_unwrap.plots.fig6_physics_zernike import plot_f6_physics_zernike


def run_f1_qualitative(
    checkpoint: str,
    data_dir: Optional[str],
    out: str,
    n_samples: int,
    show_noise: bool,
    subset: str,
    config: Optional[str],
) -> None:
    model, cfg, _, device = load_inference_state(checkpoint, data_dir=data_dir, config_path=config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = test_loader if subset == "test" else (val_loader if subset == "val" else train_loader)

    batch = next(iter(loader))
    I_raw, phi_gt, grad_phi2 = batch[0][:n_samples].to(device), batch[1][:n_samples].to(device), batch[2][:n_samples].to(device)

    with torch.no_grad():
        phi_abs, _, _, _, _ = model(I_raw, grad_phi2)

    phi_aligned, _ = piston_align(phi_abs, phi_gt)

    plot_f1_architecture(
        raw_clean_np=I_raw.cpu().numpy(),
        raw_noisy_np=I_raw.cpu().numpy(),
        gt_np=phi_gt.cpu().numpy(),
        pred_np=phi_aligned.cpu().numpy(),
        show_noise=show_noise,
        filepath=out,
    )


def run_f2_baseline(
    checkpoint: str,
    data_dir: Optional[str],
    out: str,
    n_samples: int,
    subset: str,
    config: Optional[str],
) -> None:
    model, cfg, _, device = load_inference_state(checkpoint, data_dir=data_dir, config_path=config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = test_loader if subset == "test" else (val_loader if subset == "val" else train_loader)

    batch = next(iter(loader))
    I_raw, phi_gt, grad_phi2 = batch[0][:n_samples].to(device), batch[1][:n_samples].to(device), batch[2][:n_samples].to(device)

    with torch.no_grad():
        phi_abs, _, _, _, _ = model(I_raw, grad_phi2)

    phi_aligned, _ = piston_align(phi_abs, phi_gt)
    pclcn_np = phi_aligned.cpu().numpy()
    gt_np = phi_gt.cpu().numpy()
    raw_np = I_raw.cpu().numpy()

    itoh_imgs = []
    lsq_imgs = []

    from phase_unwrap.analysis.baselines import _unwrap_itoh, _unwrap_skimage

    for row in range(len(I_raw)):
        I_img = raw_np[row, 0]
        gt_img = gt_np[row, 0]

        try:
            itoh_pred = _unwrap_itoh(I_img)
            itoh_t = torch.from_numpy(itoh_pred).unsqueeze(0).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).float()
            itoh_aligned, _ = piston_align(itoh_t, gt_t)
            itoh_imgs.append(itoh_aligned.numpy()[0])
        except Exception:
            itoh_imgs.append(np.zeros_like(gt_img)[np.newaxis, ...])

        try:
            lsq_pred = _unwrap_skimage(I_img)
            lsq_t = torch.from_numpy(lsq_pred).unsqueeze(0).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).float()
            lsq_aligned, _ = piston_align(lsq_t, gt_t)
            lsq_imgs.append(lsq_aligned.numpy()[0])
        except Exception:
            lsq_imgs.append(np.zeros_like(gt_img)[np.newaxis, ...])

    itoh_np = np.stack(itoh_imgs)
    lsq_np = np.stack(lsq_imgs)

    plot_f2_baseline_comparison(
        raw_i_np=raw_np,
        gt_np=gt_np,
        itoh_np=itoh_np,
        lsq_np=lsq_np,
        unet_np=pclcn_np,
        filepath=out,
    )


def run_f5_diagnostics(
    checkpoint: str,
    data_dir: Optional[str],
    out: str,
    subset: str,
    config: Optional[str],
) -> None:
    model, cfg, _, device = load_inference_state(checkpoint, data_dir=data_dir, config_path=config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = test_loader if subset == "test" else (val_loader if subset == "val" else train_loader)

    batch = next(iter(loader))
    I_raw, phi_gt, grad_phi2 = batch[0][:1].to(device), batch[1][:1].to(device), batch[2][:1].to(device)

    with torch.no_grad():
        phi_abs, _, _, _, _ = model(I_raw, grad_phi2)

    phi_aligned, _ = piston_align(phi_abs, phi_gt)

    plot_f5_diagnostics(
        gt_np=phi_gt.cpu().numpy(),
        pred_np=phi_aligned.cpu().numpy(),
        filepath=out,
    )


def run_f6_physics_zernike(
    checkpoint: str,
    data_dir: Optional[str],
    out: str,
    subset: str,
    config: Optional[str],
) -> None:
    model, cfg, _, device = load_inference_state(checkpoint, data_dir=data_dir, config_path=config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = test_loader if subset == "test" else (val_loader if subset == "val" else train_loader)

    batch = next(iter(loader))
    I_raw, phi_gt, grad_phi2 = batch[0][:1].to(device), batch[1][:1].to(device), batch[2][:1].to(device)

    with torch.no_grad():
        phi_abs, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)
        wrapped_phase, _ = model.stem(I_raw)
        gx, gy = model.grad_op(wrapped_phase)

        curl_raw = (torch.diff(gy, dim=-1, prepend=gy[..., :1]) - torch.diff(gx, dim=-2, prepend=gx[..., :1, :])).abs()
        curl_corr = (torch.diff(gy_tilde, dim=-1, prepend=gy_tilde[..., :1]) - torch.diff(gx_tilde, dim=-2, prepend=gx_tilde[..., :1, :])).abs()

        c_gt, _ = model.zernike_proj(phi_gt)

    phi_aligned, _ = piston_align(phi_abs, phi_gt)

    plot_f6_physics_zernike(
        curl_raw_np=curl_raw.cpu().numpy(),
        curl_corr_np=curl_corr.cpu().numpy(),
        c_gt_np=c_gt.cpu().numpy(),
        c_pred_np=c_zernike.cpu().numpy(),
        gt_2d_np=phi_gt.cpu().numpy(),
        pred_2d_np=phi_aligned.cpu().numpy(),
        filepath=out,
    )
