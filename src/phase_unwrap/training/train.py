"""
Training loop for absolute phase reconstruction.
"""

from __future__ import annotations

import csv
import os
import random
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from tqdm.auto import tqdm

from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import MAEGradLoss, compute_metrics
from ..core.ops import FixedSobel, affine_align, curvature_loss
from ..core.utils import ensure_dir, pick_device, set_seed
from ..data import build_dataloaders
from ..data.augmentation import NoiseAug, NoiseScheduler, prepare_batch
from ..model import EMA, build_model
from ..visualize import save_epoch_visuals

# Auto-push imports (optional, only used if callback provided)
try:
    from ..git_automation.callback import AutoPushCallback
except ImportError:
    AutoPushCallback = None

# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "train_mae",
    "train_grad",
    "val_mae",
    "val_rmse",
    "val_ssim",
    "val_psnr",
    "val_max_err",
    "val_grad_mae",
    "lr",
    "epoch_time_s",
]


def _init_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def _append_csv(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in CSV_COLUMNS])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    best_mae: float,
) -> None:
    # Handle DataParallel wrapper when saving
    from .multi_gpu import get_model_state_dict

    state = {
        "epoch": epoch,
        "best_mae": best_mae,
        "model": get_model_state_dict(model),
        "model_ema": get_model_state_dict(ema.m),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    """Load full training state. Returns (start_epoch, best_mae)."""
    from .multi_gpu import load_model_state_dict

    # weights_only=False: loading trusted checkpoint from own training runs
    ckpt = torch.load(path, map_location=device, weights_only=False)
    load_model_state_dict(model, ckpt["model"])
    load_model_state_dict(ema.m, ckpt["model_ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.random.set_rng_state(ckpt["rng_torch"])
    if "rng_cuda" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    return ckpt["epoch"], ckpt["best_mae"]


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
) -> Dict[str, float]:
    """Evaluate using affine-aligned predictions."""
    if loader is None:
        return {
            k: float("nan")
            for k in ["MAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
        }

    sums: Dict[str, float] = {
        k: 0.0 for k in ["MAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
    }
    n = 0

    for I_raw, phi_gt in loader:
        I_raw = I_raw.to(device, non_blocking=True)
        phi_gt = phi_gt.to(device, non_blocking=True)
        
        I_input, phi_gt, _, _ = prepare_batch(I_raw, phi_gt, noise_aug=None)
        
        I_input = I_input.contiguous(memory_format=torch.channels_last)
        phi_gt = phi_gt.contiguous(memory_format=torch.channels_last)
        bs = I_input.size(0)

        with autocast(device_type=device.type, enabled=use_amp):
            phi_raw, k_off = model(I_input)
            phi_abs = phi_raw + k_off

        phi_aligned, _, _ = affine_align(phi_abs, phi_gt)

        m = compute_metrics(phi_aligned, phi_gt)
        sums["MAE"] += float(m["MAE"]) * bs
        sums["RMSE"] += float(m["RMSE"]) * bs

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
    auto_push_callback: Optional[Any] = None,
    multi_gpu: bool = False,
    gpu_ids: Optional[list[int]] = None,
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

    model = build_model(cfg.model).to(device=device, memory_format=torch.channels_last)
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("[startup] torch.compile enabled")
        except Exception as e:
            print(f"[startup] torch.compile failed: {e}")

    # Setup multi-GPU if requested
    if multi_gpu and use_cuda:
        from .multi_gpu import print_gpu_info, setup_multi_gpu

        print_gpu_info()
        model = setup_multi_gpu(model, gpu_ids=gpu_ids)

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

    # PyTorch 1.1+ requires optimizer.step() before scheduler.step().
    # We step them once here (with fake zero-gradients) to initialize the LR correctly
    # and silence the warning on the first epoch.
    opt.step()
    sched.step()

    scaler = GradScaler(device=device.type, enabled=use_amp)
    ema = EMA(model, decay=cfg.model.ema_decay)

    best_mae = float("inf")
    start_epoch = 1
    
    if resume_path and os.path.exists(resume_path):
        print(f"[resume] Loading checkpoint from {resume_path}")
        start_epoch, best_mae = load_checkpoint(
            resume_path, model, ema, opt, sched, scaler, device
        )
    elif resume_path:
        print(f"[resume] Warning: Checkpoint not found at {resume_path}")

    if start_epoch == 1:
        _init_csv(metrics_path)

    # Initialize dynamic curriculum augmentation
    train_aug = None
    noise_sched = None
    if getattr(cfg, "aug", None) and cfg.aug.enable:
        train_aug = NoiseAug(
            gauss_std=cfg.aug.gauss_std,
            speckle_std=cfg.aug.speckle_std,
            poisson_scale=cfg.aug.poisson_scale,
            lowfreq_amp=cfg.aug.lowfreq_amp,
            lowfreq_sigma=cfg.aug.lowfreq_sigma,
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

    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        if noise_sched is not None and train_aug is not None:
            lvl = noise_sched.level(epoch)
            train_aug.set_level(lvl)
            print(f"[noise] epoch={epoch} level={lvl:.3f}")

        t0 = time.time()
        model.train()
        run_loss = 0.0
        run_mae = 0.0
        run_grad = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.epochs}", leave=False, mininterval=2.0)
        for I_raw, phi_gt in pbar:
            I_raw = I_raw.to(device, non_blocking=True)
            phi_gt = phi_gt.to(device, non_blocking=True)
            
            I_input, phi_gt, I_raw_n, I_raw_c = prepare_batch(I_raw, phi_gt, noise_aug=train_aug)
            
            I_input = I_input.contiguous(memory_format=torch.channels_last)
            phi_gt = phi_gt.contiguous(memory_format=torch.channels_last)
            I_raw_n = I_raw_n.contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    phi_raw, k_off = model(I_input)

                    if isinstance(phi_raw, list):
                        phi_abs = [p + k_off for p in phi_raw]
                        # Ensure curvature loss is only calculated on the finest resolution
                        phi_abs_fine = phi_abs[-1]
                    else:
                        phi_abs = phi_raw + k_off
                        phi_abs_fine = phi_abs

                    L_phase, parts = loss_fn(
                        phi_abs,
                        phi_gt,
                        I_raw_n if cfg.loss.int_wgrad else None,
                    )
                    L_curv = cfg.loss.w_curv * curvature_loss(phi_abs_fine)
                    loss = cfg.loss.w_data * L_phase + L_curv

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
                if scale_after >= scale_before:
                    sched.step()

                ema.update(model)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
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
                    break
                raise

            bs = I_input.size(0)
            run_loss += float(loss.item()) * bs
            run_mae += float(parts["mae"].item()) * bs
            run_grad += float(parts["grad"].item()) * bs
            cnt += bs

            pbar.set_postfix(
                {
                    "tot": f"{float(loss.item()):.4f}",
                    "mae": f"{float(parts['mae']):.4f}",
                    "grad": f"{float(parts['grad']):.4f}",
                }
            )

        train_loss = run_loss / max(1, cnt)
        train_mae = run_mae / max(1, cnt)
        train_grad = run_grad / max(1, cnt)
        epoch_time = time.time() - t0

        eval_stats: Dict[str, float] = {}
        if (epoch % cfg.logging.val_interval == 0) and (val_loader is not None):
            eval_stats = run_eval(ema.m, val_loader, device, use_amp, eval_sobel)

            try:
                # Get visualization data from train_loader to show actual noisy data
                I_raw_v, phi_gt_v = next(iter(train_loader))
                I_raw_v = I_raw_v.to(device, non_blocking=True)
                phi_gt_v = phi_gt_v.to(device, non_blocking=True)
                
                I_input_v, phi_gt_v, I_raw_n_v, I_raw_c_v = prepare_batch(I_raw_v, phi_gt_v, noise_aug=train_aug)
                
                I_input_v = I_input_v.contiguous(memory_format=torch.channels_last)
                phi_gt_v = phi_gt_v.contiguous(memory_format=torch.channels_last)
                
                with autocast(device_type=device.type, enabled=use_amp):
                    phi_raw_v, k_off_v = ema.m(I_input_v)
                    phi_abs_v = phi_raw_v + k_off_v
                phi_aligned_v, _, _ = affine_align(phi_abs_v, phi_gt_v)

                # Get current noise level for visualization
                current_noise_level = 0.0
                if noise_sched is not None:
                    current_noise_level = noise_sched.level(epoch)

                save_epoch_visuals(
                    I_input_v.cpu(),
                    phi_aligned_v.detach().cpu(),
                    phi_gt_v.cpu(),
                    vis_dir,
                    epoch,
                    cfg.logging.vis_max,
                    I_raw_clean=I_raw_c_v.cpu(),
                    I_raw_noisy=I_raw_n_v.cpu(),
                    noise_level=current_noise_level,
                    samples_per_file=4,
                )
            except Exception as e:
                print(f"Warning: saving visuals failed: {e}")

            print(
                f"Epoch {epoch} | train={train_loss:.4f} | "
                f"MAE={eval_stats.get('MAE', 0):.4f} "
                f"RMSE={eval_stats.get('RMSE', 0):.4f} "
                f"SSIM={eval_stats.get('SSIM', 0):.4f} "
                f"PSNR={eval_stats.get('PSNR', 0):.2f} "
                f"MaxErr={eval_stats.get('MaxErr', 0):.4f} | "
                f"time={epoch_time:.1f}s"
            )

            if eval_stats.get("MAE", float("inf")) < best_mae:
                best_mae = eval_stats["MAE"]
                save_checkpoint(
                    os.path.join(run_dir, "best.pth"),
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

        save_checkpoint(
            os.path.join(run_dir, "final.pth"),
            epoch,
            model,
            ema,
            opt,
            sched,
            scaler,
            best_mae,
        )

        current_lr = opt.param_groups[0]["lr"]
        _append_csv(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_mae": f"{train_mae:.6f}",
                "train_grad": f"{train_grad:.6f}",
                "val_mae": f"{eval_stats.get('MAE', ''):.6f}" if eval_stats else "",
                "val_rmse": f"{eval_stats.get('RMSE', ''):.6f}" if eval_stats else "",
                "val_ssim": f"{eval_stats.get('SSIM', ''):.6f}" if eval_stats else "",
                "val_psnr": f"{eval_stats.get('PSNR', ''):.4f}" if eval_stats else "",
                "val_max_err": f"{eval_stats.get('MaxErr', ''):.6f}"
                if eval_stats
                else "",
                "val_grad_mae": f"{eval_stats.get('GradMAE', ''):.6f}"
                if eval_stats
                else "",
                "lr": f"{current_lr:.8f}",
                "epoch_time_s": f"{epoch_time:.2f}",
            },
        )

        # Auto-push callback
        if auto_push_callback is not None:
            auto_push_callback.on_epoch_end(
                epoch=epoch,
                total_epochs=cfg.optim.epochs,
                metrics=eval_stats if eval_stats else None,
            )

    # Final auto-push callback
    if auto_push_callback is not None:
        auto_push_callback.on_train_end(final_metrics={"best_mae": best_mae})

    if test_loader is not None:
        print("\n--- Final Evaluation on Held-Out Test Set ---")
        test_stats = run_eval(
            ema.m if ema else model, test_loader, device, use_amp, eval_sobel
        )
        print(
            f"[TEST] MAE={test_stats.get('MAE', 0):.4f} "
            f"RMSE={test_stats.get('RMSE', 0):.4f} "
            f"SSIM={test_stats.get('SSIM', 0):.4f} "
            f"PSNR={test_stats.get('PSNR', 0):.2f} "
        )

    print(f"\nDone. Best Val MAE: {best_mae:.4f}")
