"""
Training loop for absolute phase reconstruction.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore", message=".*spectral_angle_mapper.*")

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from tqdm.auto import tqdm

from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import MAEGradLoss, PCLCNLoss, compute_metrics
from ..core.ops import FixedSobel, piston_align
from ..core.utils import ensure_dir, pick_device, set_seed
from ..data import build_dataloaders
from ..data.augmentation import NoiseAug, NoiseScheduler, prepare_batch
from ..model import EMA, build_model


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------
from .callbacks import CSV_COLUMNS, CSVLogger, CheckpointManager, VisualizationDispatcher

def _unpack_batch(batch: tuple, device: torch.device):
    I_raw = batch[0].to(device, non_blocking=True)
    phi_gt = batch[1].to(device, non_blocking=True)
    grad_phi2 = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
    return I_raw, phi_gt, grad_phi2


# ---------------------------------------------------------------------------
# Extended evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    sobel: FixedSobel,
    hint_mode: str = "zero",
) -> Dict[str, float]:
    """
    Evaluate with Option-B inputs by default (zero hint).

    TopoMAE key retained for CSV compatibility but is **piston-only** aligned
    (no GT scale fit). AbsMAE is raw absolute error (primary selection metric).
    """
    if loader is None:
        return {
            k: float("nan")
            for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
        }

    sums: Dict[str, float] = {
        k: 0.0 for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
    }
    n = 0

    pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True)
    for batch in pbar:
        I_raw, phi_gt, grad_phi2 = _unpack_batch(batch, device)
        bs = I_raw.size(0)

        with autocast(device_type=device.type, enabled=use_amp):
            if hasattr(model, "corrector"):
                if grad_phi2 is None:
                    grad_phi2 = torch.zeros(bs, 2, I_raw.shape[2], I_raw.shape[3], device=device)
                phi_abs, _, _, _, _ = model(I_raw, grad_phi2)
            else:
                I_input, phi_gt, _, _ = prepare_batch(
                    I_raw, phi_gt, noise_aug=None, hint_mode=hint_mode
                )
                I_input = I_input.contiguous(memory_format=torch.channels_last)
                phi_gt = phi_gt.contiguous(memory_format=torch.channels_last)
                phi_raw, k_off = model(I_input)
                phi_abs = phi_raw + k_off

        # TopoMAE key retained for CSV compat; now offset-only (piston) aligned.
        phi_aligned, _ = piston_align(phi_abs, phi_gt)

        m_abs = compute_metrics(phi_abs, phi_gt)
        m_topo = compute_metrics(phi_aligned, phi_gt)
        sums["AbsMAE"] += float(m_abs["MAE"]) * bs
        sums["TopoMAE"] += float(m_topo["MAE"]) * bs
        sums["RMSE"] += float(m_topo["RMSE"]) * bs

        max_err = (phi_aligned - phi_gt).abs().amax(dim=(1, 2, 3)).mean()
        sums["MaxErr"] += float(max_err) * bs

        pgx, pgy = sobel(phi_aligned)
        tgx, tgy = sobel(phi_gt)
        grad_mae = ((pgx - tgx).abs() + (pgy - tgy).abs()).mean()
        sums["GradMAE"] += float(grad_mae) * bs

        gt_min = phi_gt.amin(dim=(1, 2, 3), keepdim=True)
        gt_max = phi_gt.amax(dim=(1, 2, 3), keepdim=True)
        data_range = (gt_max - gt_min).clamp_min(1e-6)
        pred_norm = (phi_aligned - gt_min) / data_range
        gt_norm = (phi_gt - gt_min) / data_range
        ssim_val = ssim_fn(pred_norm, gt_norm, data_range=1.0)
        sums["SSIM"] += float(ssim_val) * bs

        mse = ((phi_aligned - phi_gt) ** 2).mean()
        data_range_mean = data_range.mean()
        psnr = 10.0 * torch.log10(data_range_mean**2 / mse.clamp_min(1e-12))
        sums["PSNR"] += float(psnr) * bs

        n += bs

    if n == 0:
        return {k: float("nan") for k in sums}
    return {k: sums[k] / n for k in sums}


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------
def _build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    cosine_T_max: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.SequentialLR:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cosine_T_max), eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(
    cfg: TrainConfig,
    resume_path: Optional[str] = None,
) -> None:
    """Main training entrypoint."""
    set_seed(cfg.logging.seed)

    device = pick_device(cfg.model.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(use_cuda and cfg.model.use_amp)

    run_dir = cfg.logging.run_dir
    vis_dir = cfg.logging.vis_dir
    ensure_dir(run_dir)
    ensure_dir(vis_dir)

    metrics_path = os.path.join(run_dir, "metrics.csv")

    print(
        f"[startup] device={device} | amp={use_amp} | "
        f"batch={cfg.optim.batch_size} | workers={cfg.data.workers}"
    )
    print(f"[run_dir] {run_dir}")

    config_snap_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_snap_path):
        Path(config_snap_path).write_text(config_to_yaml(cfg))

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )
    print(
        f"[data] dir={cfg.data.data_dir} pattern={cfg.data.pattern} | "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset) if val_loader is not None else 0} "
        f"test={len(test_loader.dataset) if test_loader is not None else 0}"
    )

    model = build_model(cfg.model).to(device=device)

    if getattr(cfg.model, "arch", "unetres2").lower() == "pclcn":
        loss_fn = PCLCNLoss(
            w_phase=getattr(cfg.loss, "w_complex", 1.0),
            w_curl=getattr(cfg.loss, "w_curl", 0.1),
        )
    else:
        loss_fn = MAEGradLoss(
            w_mae=cfg.loss.w_mae,
            w_grad=cfg.loss.w_grad,
            intensity_weighted=cfg.loss.int_wgrad,
        )
    eval_sobel = FixedSobel().to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    total_steps = cfg.optim.epochs * len(train_loader)
    warmup_steps = min(cfg.optim.warmup_steps, total_steps // 2)
    sched = _build_warmup_scheduler(
        opt, warmup_steps, total_steps - warmup_steps, cfg.optim.eta_min
    )

    # Rationale (Mathematical Implication):
    # PyTorch 1.1+ requires optimizer.step() before scheduler.step().
    # WARNING: Because this is AdamW, `opt.step()` physically applies weight decay 
    # (w = w - lambda * w) even when gradients are zero. This slightly shrinks the seeded 
    # weight initialization before the first forward pass. It is deterministic, but non-zero.
    global_step = 0
    opt.step()
    sched.step()
    global_step += 1

    scaler = GradScaler(device=device.type, enabled=use_amp)
    ema = EMA(model, decay=cfg.model.ema_decay)

    if cfg.optim.compile:
        print("[startup] torch.compile enabled")
        model = torch.compile(model)
        ema.m = torch.compile(ema.m)

    best_mae = float("inf")
    start_epoch = 1
    
    if resume_path and os.path.exists(resume_path):
        print(f"[resume] Loading checkpoint from {resume_path}")
        start_epoch, best_mae = CheckpointManager.load(
            resume_path, model, ema, opt, sched, scaler, device
        )
        
        # Rationale (State Synchronization):
        # If the user explicitly overrides `--epochs` or `--batch-size` on a resumed run, 
        # the checkpoint's frozen scheduler state will physically overwrite the new `T_max` 
        # with the old stale math. We must discard the loaded scheduler, reconstruct it 
        # natively against the new configuration, and mathematically fast-forward it.
        global_step = (start_epoch - 1) * len(train_loader) + 1
        sched = _build_warmup_scheduler(
            opt, warmup_steps, total_steps - warmup_steps, cfg.optim.eta_min
        )
        for _ in range(global_step):
            sched.step()
    elif resume_path:
        print(f"[resume] Warning: Checkpoint not found at {resume_path}")

    csv_logger = CSVLogger(metrics_path, columns=CSV_COLUMNS)
    ckpt_manager = CheckpointManager(run_dir)
    vis_dispatcher = VisualizationDispatcher(vis_dir, vis_max=cfg.logging.vis_max)

    # Initialize dynamic curriculum augmentation
    train_aug = None
    noise_sched = None
    if getattr(cfg, "aug", None) and cfg.aug.enable:
        train_aug = NoiseAug(
            gauss_std=cfg.aug.gauss_std,
            speckle_std=cfg.aug.speckle_std,
            poisson_scale=cfg.aug.poisson_scale,
            lowfreq_amp=cfg.aug.lowfreq_amp,
            blur_prob=cfg.aug.blur_prob,
            blur_sigma=(cfg.aug.blur_min, cfg.aug.blur_max),
            dropout_prob=cfg.aug.dropout_prob,
            s_and_p_prob=cfg.aug.sap_prob,
            gain_jitter=(cfg.aug.gain_min, cfg.aug.gain_max),
            offset_jitter=(cfg.aug.off_min, cfg.aug.off_max),
            hint_offset_std=cfg.aug.hint_std,
            enable=True,
        )
        noise_sched = NoiseScheduler(
            warmup_ratio=cfg.aug.warmup_ratio,
            full_ratio=cfg.aug.full_ratio,
            profile=cfg.aug.profile,
            cycles=cfg.aug.cycles,
            stochastic_std=cfg.aug.stochastic_std,
        )
        noise_sched.set_total_epochs(cfg.optim.epochs)

    vis_batch = None
    if train_loader is not None:
        vis_batch = next(iter(train_loader))

    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        partial_epoch = False
        if noise_sched is not None and train_aug is not None:
            lvl = noise_sched.level(epoch)
            train_aug.set_level(lvl)
            print(f"[noise] epoch={epoch} level={lvl:.3f}")

        t0 = time.time()
        model.train()
        run_loss = 0.0
        run_mae = 0.0
        run_grad = 0.0
        run_curv = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.epochs}", leave=False)
        for batch in pbar:
            I_raw, phi_gt, grad_phi2 = _unpack_batch(batch, device)
            opt.zero_grad(set_to_none=True)

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    if hasattr(model, "corrector"):
                        if grad_phi2 is None:
                            grad_phi2 = torch.zeros(I_raw.size(0), 2, I_raw.shape[2], I_raw.shape[3], device=device)
                        phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)
                        L_phase, parts = loss_fn(phi_final, phi_gt, gx_tilde, gy_tilde)
                        phi_abs = phi_final
                    else:
                        I_input, phi_gt, I_raw_n, I_raw_c = prepare_batch(
                            I_raw, phi_gt, noise_aug=train_aug, hint_mode=cfg.aug.hint_mode
                        )
                        I_input = I_input.contiguous(memory_format=torch.channels_last)
                        phi_gt = phi_gt.contiguous(memory_format=torch.channels_last)
                        phi_raw, k_off = model(I_input)
                        if isinstance(phi_raw, list):
                            phi_abs = [p + k_off for p in phi_raw]
                        else:
                            phi_abs = phi_raw + k_off
                        L_phase, parts = loss_fn(
                            phi_abs,
                            phi_gt,
                            I_raw_n if cfg.loss.int_wgrad else None,
                        )
                    loss = cfg.loss.w_data * L_phase

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.optim.grad_clip,
                    error_if_nonfinite=False,
                )

                # AMP scaler.step may skip the optimizer step if gradients are NaN/Inf.
                # If it skips, we shouldn't step the LR scheduler.
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                scale_after = scaler.get_scale()

                # Only step the scheduler if the scaler didn't reduce the scale
                # (which indicates it skipped the opt.step due to nan/inf grads).
                # Skip EMA when AMP skipped the optimizer step (NaN/Inf grads).
                if scale_after >= scale_before:
                    sched.step()
                    global_step += 1
                    ema.update(model)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Rationale (Architecture - Resiliency):
                    # We implement dynamic batch-size halving on CUDA OOM. Deep networks like UNet 
                    # can occasionally spike VRAM on exceptionally noisy/complex gradients. 
                    # Instead of instantly killing a 48-hour unattended training run, we catch the OOM, 
                    # flush the VRAM, automatically slash the batch size, and seamlessly rebuild the 
                    # dataloaders to resume execution cleanly.
                    print(
                        "| WARNING | CUDA OOM Exception caught! Attempting dynamic recovery..."
                    )
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)

                    if cfg.optim.batch_size <= 1:
                        raise RuntimeError(
                            "Fatal CUDA OOM: Batch size is already 1."
                        ) from e

                    cfg.optim.batch_size = max(1, cfg.optim.batch_size // 2)
                    print(
                        f"| RECOVER | Reduced dynamic batch size to {cfg.optim.batch_size}. Re-building DataLoaders..."
                    )

                    # Rebuild the dataloaders immediately to apply the new batch size
                    train_loader, val_loader, test_loader = build_dataloaders(
                        cfg, device, seed=cfg.logging.seed
                    )

                    # Rationale (State Synchronization):
                    # 1. Update `config_effective.yaml` so anyone resuming or reproducing this run 
                    # doesn't crash on the old toxic batch size. We don't overwrite config.yaml 
                    # to preserve the original intent.
                    effective_snap_path = os.path.join(run_dir, "config_effective.yaml")
                    Path(effective_snap_path).write_text(config_to_yaml(cfg))
                    
                    recovery_log_path = os.path.join(run_dir, "recovery.log")
                    with open(recovery_log_path, "a") as f:
                        f.write(f"epoch={epoch} global_step={global_step} old_bs={cfg.optim.batch_size*2} new_bs={cfg.optim.batch_size}\\n")

                    # 2. Rebuild the LR scheduler. Because batch size halved, `len(train_loader)` 
                    # just doubled! The pre-computed CosineAnnealingLR max steps are now violently 
                    # out of sync. We must recalculate the total steps and fast-forward the new 
                    # scheduler back to the exact current `global_step` to prevent math corruption.
                    # Note: We do NOT add +1 to epochs because the current epoch loop breaks immediately.
                    # Note (Reproducibility): `new_total_steps` changes the `T_max` of the cosine schedule,
                    # so the LR trajectory will stretch and deviate from the original shape.
                    new_total_steps = global_step + (cfg.optim.epochs - epoch) * len(train_loader)
                    new_warmup = min(cfg.optim.warmup_steps, new_total_steps // 2)
                    sched = _build_warmup_scheduler(
                        opt, new_warmup, new_total_steps - new_warmup, cfg.optim.eta_min
                    )
                    for _ in range(global_step):
                        sched.step()

                    # If noise is enabled, immediately re-sync the level since we created a new loader
                    if noise_sched is not None and train_aug is not None:
                        lvl = noise_sched.level(epoch)
                        train_aug.set_level(lvl)

                    # Break the current inner epoch loop. The outer 'for epoch' loop will advance
                    # naturally and restart the progress bar on the next sequence with the smaller batch size,
                    # abandoning the current poisoned epoch cleanly rather than dying.
                    print(
                        "| RECOVER | Epoch aborted gracefully. Resuming at next epoch boundary."
                    )
                    partial_epoch = True
                    break
                raise

            bs = I_raw.size(0)
            run_loss += float(loss.item()) * bs
            mae_val = float(parts.get("mae", parts.get("phase_complex", 0.0)))
            grad_val = float(parts.get("grad", parts.get("curl", 0.0)))
            curv_val = float(parts.get("curv", 0.0))
            run_mae += mae_val * bs
            run_grad += grad_val * bs
            run_curv += curv_val * cfg.loss.w_curv * bs
            cnt += bs

            pbar.set_postfix(
                {
                    "tot": f"{float(loss.item()):.4f}",
                    "mae": f"{mae_val:.4f}",
                    "grad": f"{grad_val:.4f}",
                }
            )

        train_loss = run_loss / max(1, cnt)
        train_mae = run_mae / max(1, cnt)
        train_grad = run_grad / max(1, cnt)
        train_curv = run_curv / max(1, cnt)
        epoch_time = time.time() - t0

        eval_stats: Dict[str, float] = {}
        if (not partial_epoch) and (epoch % cfg.logging.val_interval == 0) and (val_loader is not None):
            eval_stats = run_eval(
                ema.m,
                val_loader,
                device,
                use_amp,
                eval_sobel,
                hint_mode=cfg.aug.hint_mode,
            )

            try:
                # Get visualization data from cached vis_batch
                if vis_batch is not None:
                    I_raw_v, phi_gt_v = vis_batch
                else:
                    I_raw_v, phi_gt_v = next(iter(train_loader))
                I_raw_v = I_raw_v.to(device, non_blocking=True)
                phi_gt_v = phi_gt_v.to(device, non_blocking=True)

                I_input_v, phi_gt_v, I_raw_n_v, I_raw_c_v = prepare_batch(
                    I_raw_v,
                    phi_gt_v,
                    noise_aug=train_aug,
                    hint_mode=cfg.aug.hint_mode,
                )

                I_input_v = I_input_v.contiguous(memory_format=torch.channels_last)
                phi_gt_v = phi_gt_v.contiguous(memory_format=torch.channels_last)

                with autocast(device_type=device.type, enabled=use_amp):
                    phi_raw_v, k_off_v = ema.m(I_input_v)
                    phi_abs_v = phi_raw_v + k_off_v
                # Epoch PNGs: piston-only (same as metrics); affine_align remains for other plots
                phi_aligned_v, _ = piston_align(phi_abs_v, phi_gt_v)

                # Get current noise level for visualization
                current_noise_level = 0.0
                if noise_sched is not None:
                    current_noise_level = noise_sched.level(epoch)

                vis_dispatcher.dispatch(
                    epoch=epoch,
                    I_input_v=I_input_v,
                    phi_aligned_v=phi_aligned_v,
                    phi_gt_v=phi_gt_v,
                    I_raw_c_v=I_raw_c_v,
                    I_raw_n_v=I_raw_n_v,
                    noise_level=current_noise_level,
                )

            except Exception as e:
                print(f"Warning: dispatching visuals failed: {e}")

            print(
                f"Epoch {epoch} | train={train_loss:.4f} | "
                f"AbsMAE={eval_stats.get('AbsMAE', 0):.4f} "
                f"TopoMAE={eval_stats.get('TopoMAE', 0):.4f} "
                f"RMSE={eval_stats.get('RMSE', 0):.4f} "
                f"SSIM={eval_stats.get('SSIM', 0):.4f} "
                f"PSNR={eval_stats.get('PSNR', 0):.2f} "
                f"MaxErr={eval_stats.get('MaxErr', 0):.4f} | "
                f"time={epoch_time:.1f}s"
            )

            # Primary selection: raw AbsMAE (no GT scale/piston fit). TopoMAE is diagnostic.
            if eval_stats.get("AbsMAE", float("inf")) < best_mae:
                best_mae = eval_stats["AbsMAE"]
                ckpt_manager.save(
                    "best.pth",
                    epoch,
                    model,
                    ema,
                    opt,
                    sched,
                    scaler,
                    best_mae,
                )
        else:
            print(f"Epoch {epoch} | train={train_loss:.4f} | time={epoch_time:.1f}s")

        ckpt_manager.save(
            "final.pth",
            epoch,
            model,
            ema,
            opt,
            sched,
            scaler,
            best_mae,
        )

        current_lr = opt.param_groups[0]["lr"]
        
        # Get current noise level to log it
        current_noise_level = 0.0
        if noise_sched is not None:
            current_noise_level = noise_sched.level(epoch)
            
        csv_logger.log(
            {
                "epoch": epoch,
                "partial": 1 if partial_epoch else 0,
                "noise_level": f"{current_noise_level:.4f}",
                "train_loss": f"{train_loss:.6f}",
                "train_mae": f"{train_mae:.6f}",
                "train_grad": f"{train_grad:.6f}",
                "train_curv": f"{train_curv:.6f}",
                "val_abs_mae": f"{eval_stats.get('AbsMAE', ''):.6f}" if eval_stats else "",
                "val_topo_mae": f"{eval_stats.get('TopoMAE', ''):.6f}" if eval_stats else "",
                "val_rmse": f"{eval_stats.get('RMSE', ''):.6f}" if eval_stats else "",
                "val_ssim": f"{eval_stats.get('SSIM', ''):.6f}" if eval_stats else "",
                "val_psnr": f"{eval_stats.get('PSNR', ''):.4f}" if eval_stats else "",
                "val_max_err": f"{eval_stats.get('MaxErr', ''):.6f}" if eval_stats else "",
                "val_grad_mae": f"{eval_stats.get('GradMAE', ''):.6f}" if eval_stats else "",
                "lr": f"{current_lr:.8f}",
                "epoch_time_s": f"{epoch_time:.1f}",
            }
        )
    if test_loader is not None:
        print("\n--- Final Evaluation on Held-Out Test Set ---")
        test_stats = run_eval(
            ema.m if ema else model,
            test_loader,
            device,
            use_amp,
            eval_sobel,
            hint_mode=cfg.aug.hint_mode,
        )
        print(
            f"[TEST] AbsMAE={test_stats.get('AbsMAE', 0):.4f} "
            f"TopoMAE(piston)={test_stats.get('TopoMAE', 0):.4f} "
            f"RMSE={test_stats.get('RMSE', 0):.4f} "
            f"SSIM={test_stats.get('SSIM', 0):.4f} "
            f"PSNR={test_stats.get('PSNR', 0):.2f} "
        )
        test_path = os.path.join(run_dir, "test_metrics.csv")
        test_csv_logger = CSVLogger(
            test_path,
            columns=["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
        )
        test_csv_logger.log(
            {
                "AbsMAE": f"{test_stats.get('AbsMAE', float('nan')):.6f}",
                "TopoMAE": f"{test_stats.get('TopoMAE', float('nan')):.6f}",
                "RMSE": f"{test_stats.get('RMSE', float('nan')):.6f}",
                "SSIM": f"{test_stats.get('SSIM', float('nan')):.6f}",
                "PSNR": f"{test_stats.get('PSNR', float('nan')):.4f}",
                "MaxErr": f"{test_stats.get('MaxErr', float('nan')):.6f}",
                "GradMAE": f"{test_stats.get('GradMAE', float('nan')):.6f}",
            }
        )

    print(f"\nDone. Best Val AbsMAE: {best_mae:.4f}")
