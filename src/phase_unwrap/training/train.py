"""
Training loop for absolute phase reconstruction.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from tqdm.auto import tqdm

from ..core import _torch_compat  # noqa: F401
from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import PCLCNLoss, compute_metrics
from ..core.ops import FixedSobel, piston_align
from ..core.utils import ensure_dir, pick_device, set_seed


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
from ..data import build_dataloaders, build_otf_loaders
from ..data.augmentation import NoiseAug, normalize_intensity
from ..model import EMA, build_model
from .callbacks import (
    CSV_COLUMNS,
    CheckpointManager,
    CSVLogger,
    VisualizationDispatcher,
)

warnings.filterwarnings("ignore", message=".*spectral_angle_mapper.*")


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    sobel: FixedSobel,
    noise_aug: NoiseAug | None = None,
    severity: float = 0.0,
    noise_seed: int = 0,
    observation: str = "intensity",
) -> Dict[str, float]:
    """Evaluate validation dataset metrics."""
    if loader is None:
        return {
            k: float("nan")
            for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
        }

    sums: Dict[str, float] = {
        k: 0.0
        for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]
    }
    n = 0

    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(noise_seed)
    pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True)
    for batch in pbar:
        I_raw, phi_gt, grad_phi2 = (x.to(device, non_blocking=True) for x in batch[:3])
        bs = I_raw.size(0)

        with autocast(device_type=device.type, enabled=use_amp):
            if observation in {"wrapped_dp", "two_frame"}:
                I_input = I_raw
            elif observation == "two_frame_noisy":
                I1, I2 = I_raw[:, 0:1], I_raw[:, 1:2]
                I1_noisy, _ = noise_aug(I1, severity, generator=noise_generator)
                I2_noisy, _ = noise_aug(I2, severity, generator=noise_generator)
                I_input = torch.atan2(2.0 - I2_noisy, I1_noisy - 2.0)
            else:
                I_input = (
                    normalize_intensity(I_raw)
                    if noise_aug is None
                    else noise_aug(I_raw, severity, generator=noise_generator)[1]
                )
            phi_abs, _, _, _, _ = model(
                I_input,
                grad_phi2
                if getattr(model, "reference_mode", "reference_free") == "calibrated"
                else None,
            )

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


@torch.no_grad()
def _run_eval_snr(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    sobel: FixedSobel,
    snr_db: float,
    noise_seed: int = 0,
) -> Dict[str, float]:
    """Evaluate model on test set with additive Gaussian noise at a given SNR (dB).

    Reuses the same metric pipeline as ``run_eval`` but replaces the curriculum
    ``NoiseAug`` pipeline with standard Gaussian noise.  ``snr_db=inf`` means
    clean (no noise added).
    """
    if loader is None:
        return {k: float("nan") for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]}

    sums: Dict[str, float] = {k: 0.0 for k in ["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"]}
    n = 0

    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(noise_seed)

    pbar = tqdm(loader, desc=f"SNR={snr_db}" if not math.isinf(snr_db) else "SNR=clean", leave=False, dynamic_ncols=True)
    for batch in pbar:
        I_raw, phi_gt, grad_phi2 = (x.to(device, non_blocking=True) for x in batch[:3])
        bs = I_raw.size(0)

        # Add Gaussian noise at the requested SNR (dB)
        if math.isinf(snr_db):
            I_noisy = I_raw
        else:
            signal_power = I_raw.square().mean(dim=(-2, -1), keepdim=True)
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(I_raw) * noise_power.sqrt()
            I_noisy = I_raw + noise

        I_input = normalize_intensity(I_noisy)

        with autocast(device_type=device.type, enabled=use_amp):
            phi_abs, _, _, _, _ = model(
                I_input,
                grad_phi2
                if getattr(model, "reference_mode", "reference_free") == "calibrated"
                else None,
            )

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
    observation = getattr(cfg.data, "observation", "intensity")
    if observation not in {"intensity", "wrapped_dp", "two_frame", "two_frame_noisy"}:
        raise ValueError(f"Unknown data.observation: {observation}")
    if observation in {"wrapped_dp", "two_frame"}:
        if getattr(cfg.data, "backend", "otf") != "otf":
            raise ValueError(f"data.observation={observation} requires data.backend=otf")
        if getattr(cfg, "aug", None) and cfg.aug.enable:
            raise ValueError(f"Disable aug for the {observation} diagnostic")
    if observation == "two_frame_noisy" and not getattr(cfg.aug, "enable", False):
        raise ValueError("data.observation=two_frame_noisy requires aug.enable=true")

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
    print(f"[observation] {observation}")

    config_snap_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_snap_path):
        Path(config_snap_path).write_text(config_to_yaml(cfg))

    if getattr(cfg.data, "backend", "otf") == "otf":
        train_loader, val_loader, test_loader = build_otf_loaders(cfg, device)
    else:
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg, device, seed=cfg.logging.seed
        )
    print(
        f"[data] dir={cfg.data.data_dir} pattern={cfg.data.pattern} | "
        f"train={len(train_loader)} "
        f"val={len(val_loader) if val_loader is not None else 0} "
        f"test={len(test_loader) if test_loader is not None else 0}"
    )

    model = build_model(cfg.model).to(device=device)
    model.reference_mode = cfg.model.reference_mode
    model.observation_mode = "two_frame" if observation == "two_frame_noisy" else observation
    loss_fn = PCLCNLoss(
        w_phase=getattr(cfg.loss, "w_complex", 1.0),
        w_curl=getattr(cfg.loss, "w_curl", 0.1),
        w_grad_curv=getattr(cfg.loss, "w_grad_curv", 0.1),
    )
    eval_sobel = FixedSobel().to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    total_steps = cfg.optim.epochs * (
        cfg.data.steps_per_epoch
        if getattr(cfg.data, "backend", "otf") == "otf"
        else len(train_loader)
    )
    warmup_steps = min(cfg.optim.warmup_steps, total_steps // 2)
    sched = _build_warmup_scheduler(
        opt, warmup_steps, total_steps - warmup_steps, cfg.optim.eta_min
    )


    global_step = 0
    opt.step()
    sched.step()
    global_step += 1

    scaler = GradScaler(device=device.type, enabled=use_amp)
    ema = EMA(model, decay=cfg.model.ema_decay)
    ema.m.reference_mode = cfg.model.reference_mode
    ema.m.observation_mode = model.observation_mode

    if cfg.optim.compile:
        print("[startup] torch.compile enabled")
        model = torch.compile(model)
        ema.m = torch.compile(ema.m)

    best_mae = float("inf")
    start_epoch = 1
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(getattr(cfg.data, "noise_seed", cfg.logging.seed + 1))

    is_resuming = bool(resume_path and os.path.exists(resume_path))
    if is_resuming:
        print(f"[resume] Loading checkpoint from {resume_path}")
        loaded = CheckpointManager.load(
            resume_path,
            model,
            ema,
            opt,
            sched,
            scaler,
            device,
            phase_generator=getattr(train_loader, "generator", None),
            noise_generator=noise_generator,
        )
        start_epoch, best_mae = loaded[:2]
        if hasattr(train_loader, "next_sample_id"):
            train_loader.next_sample_id = loaded[2]


        global_step = (start_epoch - 1) * len(train_loader) + 1
    elif resume_path:
        print(f"[resume] Warning: Checkpoint not found at {resume_path}")

    csv_logger = CSVLogger(metrics_path, columns=CSV_COLUMNS, append=is_resuming)
    ckpt_manager = CheckpointManager(run_dir)
    vis_dispatcher = VisualizationDispatcher(vis_dir, vis_max=cfg.logging.vis_max)

    # Initialize dynamic curriculum augmentation
    train_aug = None
    if getattr(cfg, "aug", None) and cfg.aug.enable:
        train_aug = NoiseAug(
            gauss_std=cfg.aug.gauss_std,
            speckle_std=cfg.aug.speckle_std,
            photon_min=cfg.aug.photon_min,
            photon_max=cfg.aug.photon_max,
            sensor_min=cfg.aug.sensor_min,
            sensor_max=cfg.aug.sensor_max,
            lowfreq_amp=cfg.aug.lowfreq_amp,
            blur_prob=cfg.aug.blur_prob,
            blur_sigma=(cfg.aug.blur_min, cfg.aug.blur_max),
            dropout_prob=cfg.aug.dropout_prob,
            s_and_p_prob=cfg.aug.sap_prob,
            gain_jitter=(cfg.aug.gain_min, cfg.aug.gain_max),
            offset_jitter=(cfg.aug.off_min, cfg.aug.off_max),
            enable=True,
        ).to(device)

    vis_batch = None
    if train_loader is not None:
        vis_batch = next(iter(train_loader))

    for epoch in range(start_epoch, cfg.optim.epochs + 1):

        current_noise_level = min(1.0, (epoch - 1) / max(1, cfg.optim.epochs - 1))

        if train_aug is not None:
            print(f"[noise] epoch={epoch} level={current_noise_level:.3f}")

        t0 = time.time()
        model.train()
        run_loss = 0.0
        run_mae = 0.0
        run_grad = 0.0
        run_curv = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.epochs}", leave=False)
        for batch in pbar:
            I_raw, phi_gt, grad_phi2 = (
                x.to(device, non_blocking=True) for x in batch[:3]
            )
            opt.zero_grad(set_to_none=True)

            if observation in {"wrapped_dp", "two_frame"}:
                I_norm_noisy = I_raw
            elif observation == "two_frame_noisy":
                I1, I2 = I_raw[:, 0:1], I_raw[:, 1:2]
                progress = global_step / max(1, total_steps)
                if progress < 0.05:
                    step_severity = 0.0
                else:
                    u = float(torch.rand((), device=device, generator=noise_generator))
                    if u < cfg.aug.clean_probability:
                        step_severity = 0.0
                    elif u < cfg.aug.clean_probability + cfg.aug.mid_probability:
                        step_severity = float(torch.empty((), device=device).uniform_(
                            0.05, 0.75, generator=noise_generator
                        ))
                    else:
                        step_severity = float(torch.empty((), device=device).uniform_(
                            0.75, 1.0, generator=noise_generator
                        ))
                I1_noisy, _ = train_aug(I1, step_severity, generator=noise_generator)
                I2_noisy, _ = train_aug(I2, step_severity, generator=noise_generator)
                I_norm_noisy = torch.atan2(2.0 - I2_noisy, I1_noisy - 2.0)
            elif train_aug is not None:
                progress = global_step / max(1, total_steps)
                if progress < 0.05:
                    step_severity = 0.0
                else:
                    u = float(torch.rand((), device=device, generator=noise_generator))
                    if u < cfg.aug.clean_probability:
                        step_severity = 0.0
                    elif u < cfg.aug.clean_probability + cfg.aug.mid_probability:
                        step_severity = float(
                            torch.empty((), device=device).uniform_(
                                0.05, 0.75, generator=noise_generator
                            )
                        )
                    else:
                        step_severity = float(
                            torch.empty((), device=device).uniform_(
                                0.75, 1.0, generator=noise_generator
                            )
                        )
                I_raw_noisy, I_norm_noisy = train_aug(
                    I_raw, step_severity, generator=noise_generator
                )
            else:
                I_norm_noisy = normalize_intensity(I_raw)

            with autocast(device_type=device.type, enabled=use_amp):
                phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(
                    I_norm_noisy,
                    grad_phi2 if cfg.model.reference_mode == "calibrated" else None,
                )
                phi_for_loss = (
                    piston_align(phi_final, phi_gt)[0]
                    if observation in {"wrapped_dp", "two_frame", "two_frame_noisy"}
                    else phi_final
                )
                L_phase, parts = loss_fn(
                    phi_for_loss, phi_gt, gx_tilde, gy_tilde
                )
                loss = cfg.loss.w_data * L_phase

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.optim.grad_clip,
                error_if_nonfinite=False,
            )

            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            scale_after = scaler.get_scale()

            # Only step scheduler/EMA if optimizer stepped (no NaN grads)
            if scale_after >= scale_before:
                sched.step()
                global_step += 1
                ema.update(model)

            bs = I_raw.size(0)
            run_loss += float(loss.item()) * bs
            mae_val = float(parts.get("mae", parts.get("phase", 0.0)))
            grad_val = float(parts.get("curl", 0.0))
            curv_val = float(parts.get("grad_curv", 0.0))
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
        if (
            (epoch % cfg.logging.val_interval == 0)
            and (val_loader is not None)
        ):
            eval_severities = [0.0] if observation in {"wrapped_dp", "two_frame"} else cfg.aug.fixed_severities
            severity_stats = {
                s: run_eval(
                    ema.m,
                    val_loader,
                    device,
                    use_amp,
                    eval_sobel,
                    noise_aug=train_aug,
                    severity=s,
                    noise_seed=cfg.data.noise_seed,
                    observation=observation,
                )
                for s in eval_severities
            }
            eval_stats = severity_stats[0.0]

            try:
                # Get visualization data from cached vis_batch
                I_raw_v, phi_gt_v, grad_phi2_v = (
                    x.to(device, non_blocking=True) for x in vis_batch[:3]
                )

                if observation in {"wrapped_dp", "two_frame"}:
                    I_raw_n_v = I_raw_v
                    I_input_v = I_raw_v
                elif observation == "two_frame_noisy":
                    I1_v, I2_v = I_raw_v[:, 0:1], I_raw_v[:, 1:2]
                    I1_n_v, _ = train_aug(I1_v, current_noise_level, generator=noise_generator)
                    I2_n_v, _ = train_aug(I2_v, current_noise_level, generator=noise_generator)
                    I_raw_n_v = torch.atan2(2.0 - I2_n_v, I1_n_v - 2.0)
                    I_input_v = I_raw_n_v
                elif train_aug is not None:
                    I_raw_n_v, I_input_v = train_aug(
                        I_raw_v, current_noise_level, generator=noise_generator
                    )
                else:
                    I_raw_n_v = I_raw_v
                    I_input_v = normalize_intensity(I_raw_v)

                with autocast(device_type=device.type, enabled=use_amp):
                    phi_abs_v, _, _, _, _ = ema.m(
                        I_input_v,
                        grad_phi2_v
                        if cfg.model.reference_mode == "calibrated"
                        else None,
                    )

                phi_aligned_v, _ = piston_align(phi_abs_v, phi_gt_v)

                vis_dispatcher.dispatch(
                    epoch=epoch,
                    I_input_v=I_input_v,
                    phi_aligned_v=phi_aligned_v,
                    phi_gt_v=phi_gt_v,
                    I_raw_c_v=I_raw_v,
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
            if observation == "two_frame_noisy":
                print(
                    "[val-noise] "
                    + " ".join(
                        f"{name}={severity_stats[level]['TopoMAE']:.4f}"
                        for name, level in (
                            ("clean", 0.0),
                            ("light", 0.25),
                            ("moderate", 0.5),
                            ("hard", 0.75),
                            ("extreme", 1.0),
                        )
                    )
                )

            # Selection based on piston-aligned MAE across noise levels
            selection_score = (
                eval_stats["TopoMAE"]
                if observation in {"wrapped_dp", "two_frame"}
                else sum(
                    severity_stats[s]["TopoMAE"] for s in (0.0, 0.25, 0.5)
                )
                / 3
            )
            save_kwargs = {
                "epoch": epoch,
                "model": model,
                "ema": ema,
                "optimizer": opt,
                "scheduler": sched,
                "scaler": scaler,
                "phase_generator": getattr(train_loader, "generator", None),
                "noise_generator": noise_generator,
                "next_sample_id": getattr(train_loader, "next_sample_id", 0),
            }
            if selection_score < best_mae:
                best_mae = selection_score
                ckpt_manager.save("best.pth", best_mae=best_mae, **save_kwargs)
        else:
            print(f"Epoch {epoch} | train={train_loss:.4f} | time={epoch_time:.1f}s")
            save_kwargs = {
                "epoch": epoch,
                "model": model,
                "ema": ema,
                "optimizer": opt,
                "scheduler": sched,
                "scaler": scaler,
                "phase_generator": getattr(train_loader, "generator", None),
                "noise_generator": noise_generator,
                "next_sample_id": getattr(train_loader, "next_sample_id", 0),
            }

        ckpt_manager.save("final.pth", best_mae=best_mae, **save_kwargs)

        current_lr = opt.param_groups[0]["lr"]

        csv_logger.log(
            {
                "epoch": epoch,
                "partial": 0,
                "noise_level": f"{current_noise_level:.4f}",
                "train_loss": f"{train_loss:.6f}",
                "train_mae": f"{train_mae:.6f}",
                "train_grad": f"{train_grad:.6f}",
                "train_curv": f"{train_curv:.6f}",
                "val_abs_mae": f"{eval_stats['AbsMAE']:.6f}" if eval_stats else "",
                "val_topo_mae": f"{eval_stats['TopoMAE']:.6f}" if eval_stats else "",
                "val_rmse": f"{eval_stats['RMSE']:.6f}" if eval_stats else "",
                "val_ssim": f"{eval_stats['SSIM']:.6f}" if eval_stats else "",
                "val_psnr": f"{eval_stats['PSNR']:.4f}" if eval_stats else "",
                "val_max_err": f"{eval_stats['MaxErr']:.6f}" if eval_stats else "",
                "val_grad_mae": f"{eval_stats['GradMAE']:.6f}" if eval_stats else "",
                "lr": f"{current_lr:.8f}",
                "epoch_time_s": f"{epoch_time:.1f}",
            }
        )
    if test_loader is not None:
        print("\n--- Final Evaluation on Held-Out Test Set ---")
        best_path = os.path.join(run_dir, "best.pth")
        if os.path.exists(best_path):
            best = torch.load(best_path, map_location=device, weights_only=True)
            ema.m.load_state_dict(best.get("model_ema", best["model"]))
        test_severities = [0.0] if observation in {"wrapped_dp", "two_frame"} else cfg.aug.fixed_severities
        test_by_severity = {
            s: run_eval(
                ema.m,
                test_loader,
                device,
                use_amp,
                eval_sobel,
                noise_aug=train_aug,
                severity=s,
                noise_seed=cfg.data.noise_seed + 1,
                observation=observation,
            )
            for s in test_severities
        }
        # Gaussian SNR sweep (standard metric, runs in seconds on GPU)
        snr_levels = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        test_by_snr: dict[float, dict[str, float]] = {}
        if observation == "intensity":
            for snr_db in snr_levels + [float("inf")]:
                test_by_snr[snr_db] = _run_eval_snr(
                    ema.m,
                    test_loader,
                    device,
                    use_amp,
                    eval_sobel,
                    snr_db=snr_db,
                    noise_seed=cfg.data.noise_seed + 2,
                )
        test_stats = test_by_severity[0.0]
        print(
            f"[TEST] AbsMAE={test_stats.get('AbsMAE', 0):.4f} "
            f"TopoMAE(piston)={test_stats.get('TopoMAE', 0):.4f} "
            f"RMSE={test_stats.get('RMSE', 0):.4f} "
            f"SSIM={test_stats.get('SSIM', 0):.4f} "
            f"PSNR={test_stats.get('PSNR', 0):.2f} "
        )
        # Print SNR sweep summary
        print("  SNR sweep (TopoMAE):")
        for snr_db, stats in test_by_snr.items():
            label = "clean" if math.isinf(snr_db) else f"{snr_db:.0f} dB"
            print(f"    {label:>8s}  TopoMAE={stats.get('TopoMAE', 0):.4f}")
        test_path = os.path.join(run_dir, "test_metrics.csv")
        test_csv_logger = CSVLogger(
            test_path,
            columns=["AbsMAE", "TopoMAE", "RMSE", "SSIM", "PSNR", "MaxErr", "GradMAE"],
            append=False,
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
        artifact = {
            "generator": "MatlabSimulator.float64.v1",
            "observation": observation,
            "reference_mode": cfg.model.reference_mode,
            "seeds": {
                k: getattr(cfg.data, k)
                for k in ("train_seed", "val_seed", "test_seed", "noise_seed")
            },
            "sample_id_next": getattr(train_loader, "next_sample_id", None),
            "noise": vars(cfg.aug),
            "test_by_severity": {str(k): v for k, v in test_by_severity.items()},
            "test_by_snr": {str(k): v for k, v in test_by_snr.items()},
            "checkpoint_sha256": _hash_file(best_path)
            if os.path.exists(best_path)
            else None,
        }
        Path(os.path.join(run_dir, "result.json")).write_text(
            json.dumps(artifact, indent=2, default=str)
        )

    print(f"\nDone. Best Val AbsMAE: {best_mae:.4f}")
