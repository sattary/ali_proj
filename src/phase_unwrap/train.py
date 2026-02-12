from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import build_dataloaders
from .losses import PhaseSupervisionLoss, compute_metrics
from .model import EMA, build_model
from .ops import adaptive_curvature_loss, affine_align
from .utils import ensure_dir, pick_device, set_seed
from .visualize import save_epoch_visuals, save_training_curve


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Evaluate using affine-aligned absolute phase.
    """
    if loader is None:
        return {"MAE": float("nan"), "RMSE": float("nan")}
    sums: Dict[str, float] = {"MAE": 0.0, "RMSE": 0.0}
    n = 0
    for I_input, phi_gt, I_raw in loader:
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)

        with autocast(enabled=use_amp):
            phi_raw, a_pred, b_pred_raw, conf_logit, k_off = model(I_input)
            phi_abs = phi_raw + k_off

        phi_abs_aligned, a_batch, c_batch = affine_align(phi_abs, phi_gt)

        m = compute_metrics(phi_abs_aligned, phi_gt)
        bs = I_input.size(0)
        for k in sums:
            sums[k] += float(m[k]) * bs
        n += bs
    if n == 0:
        return {"MAE": float("nan"), "RMSE": float("nan")}
    return {k: sums[k] / n for k in sums}


def train(cfg: TrainConfig) -> None:
    """
    Main training entrypoint operating on a structured configuration.
    """
    set_seed(cfg.logging.seed)

    device = pick_device(cfg.model.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(use_cuda and cfg.model.use_amp)

    print(
        f"[startup] device={device} | amp={use_amp} | "
        f"batch={cfg.optim.batch_size} | workers={cfg.data.workers}"
    )

    ensure_dir(cfg.logging.out_dir)
    ensure_dir(cfg.logging.vis_dir)

    train_loader, val_loader = build_dataloaders(cfg.data, cfg.optim, device, seed=cfg.logging.seed)
    print(
        f"[data] dir={cfg.data.data_dir} pattern={cfg.data.pattern} | "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader is not None else 0}"
    )

    model = build_model(cfg.model).to(device)

    loss_sup = PhaseSupervisionLoss(
        w_mae=cfg.loss.w_mae,
        w_grad=cfg.loss.w_grad,
        w_wrap=cfg.loss.w_wrap,
        intensity_weighted=cfg.loss.int_wgrad,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, cfg.optim.epochs),
        eta_min=cfg.optim.eta_min,
    )
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, decay=cfg.model.ema_decay)

    best_mae = float("inf")

    epochs_x = []
    train_losses = []
    val_maes = []

    for epoch in range(1, cfg.optim.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.epochs}", leave=False)
        for I_input, phi_gt, I_raw in pbar:
            I_input = I_input.to(device)
            phi_gt = phi_gt.to(device)
            I_raw = I_raw.to(device)

            opt.zero_grad(set_to_none=True)

            try:
                with autocast(enabled=use_amp):
                    phi_raw, a_pred, b_pred_raw, conf_logit, k_off = model(I_input)

                    # predicted raw absolute phase before alignment
                    phi_abs = phi_raw + k_off

                    # per-image affine align
                    phi_abs_align, a_batch, c_batch = affine_align(phi_abs, phi_gt)

                    # training confidence mask
                    conf_used = torch.ones_like(phi_abs_align)

                    # 1) absolute phase supervision (on aligned prediction)
                    L_phase, parts = loss_sup(
                        phi_abs_align,
                        phi_gt,
                        I_raw if cfg.loss.int_wgrad else None,
                        conf_used,
                    )

                    # 2) curvature smoothness on aligned phase
                    L_curv = cfg.loss.w_curv * adaptive_curvature_loss(phi_abs_align, conf_used)

                    # final total loss
                    loss = cfg.loss.w_data * L_phase + L_curv

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.optim.grad_clip,
                    error_if_nonfinite=False,
                )
                scaler.step(opt)
                scaler.update()
                ema.update(model)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM. Try smaller batch size.")
                    torch.cuda.empty_cache()
                    return
                else:
                    raise

            run_loss += float(loss.item()) * I_input.size(0)
            cnt += I_input.size(0)

            pbar.set_postfix(
                {
                    "tot": f"{float(loss.item()):.4f}",
                    "mae": f"{float(parts['mae']):.4f}",
                    "grad": f"{float(parts['grad']):.4f}",
                }
            )

        sched.step()
        train_loss = run_loss / max(1, cnt)

        # validation
        cur_val_mae = float("nan")
        if (epoch % cfg.logging.val_interval == 0) and (val_loader is not None):
            eval_stats = run_eval(ema.m, val_loader, device, use_amp)
            cur_val_mae = eval_stats["MAE"]

            # dump visuals using EMA model on a small batch from val
            try:
                I_input_v, phi_gt_v, I_raw_v = next(iter(val_loader))
                I_input_v = I_input_v.to(device)
                phi_gt_v = phi_gt_v.to(device)
                with autocast(enabled=use_amp):
                    phi_raw_v, a_pred_v, b_pred_raw_v, conf_logit_v, k_off_v = ema.m(I_input_v)
                    phi_abs_v = phi_raw_v + k_off_v

                phi_abs_v_align, a_dbg, c_dbg = affine_align(phi_abs_v, phi_gt_v)

                pred_min = float(phi_abs_v_align.min().item())
                pred_max = float(phi_abs_v_align.max().item())
                gt_min = float(phi_gt_v.min().item())
                gt_max = float(phi_gt_v.max().item())
                print(
                    f"[val debug] after align: pred_min={pred_min:.3f} "
                    f"pred_max={pred_max:.3f} | gt_min={gt_min:.3f} gt_max={gt_max:.3f}"
                )

                save_epoch_visuals(
                    I_input_v.cpu(),
                    phi_abs_v_align.detach().cpu(),
                    phi_gt_v.cpu(),
                    cfg.logging.vis_dir,
                    epoch,
                    cfg.logging.vis_max,
                )
            except Exception as e:
                print("Warning: saving visuals failed:", e)

            print(
                f"Epoch {epoch} | train={train_loss:.4f} | "
                f"val MAE={eval_stats['MAE']:.4f} RMSE={eval_stats['RMSE']:.4f} | "
                f"time={time.time() - t0:.1f}s"
            )

            if eval_stats["MAE"] < best_mae:
                best_mae = eval_stats["MAE"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "model_ema": ema.m.state_dict(),
                    },
                    os.path.join(cfg.logging.out_dir, "best.pth"),
                )
        else:
            print(
                f"Epoch {epoch} | train={train_loss:.4f} | "
                f"time={time.time() - t0:.1f}s"
            )

        # rolling checkpoint every epoch
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "model_ema": ema.m.state_dict()},
            os.path.join(cfg.logging.out_dir, "final.pth"),
        )

        epochs_x.append(epoch)
        train_losses.append(train_loss)
        val_maes.append(cur_val_mae)

    print("Done. Best MAE:", best_mae)

    # save training curve
    save_training_curve(epochs_x, train_losses, val_maes, cfg.logging.out_dir)


