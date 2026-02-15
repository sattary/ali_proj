from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # optional TensorBoard dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

try:
    import wandb
except ImportError:
    wandb = None

from ..config import TrainConfig
from ..core.data import build_dataloaders
from ..core.losses import PhaseSupervisionLoss, compute_metrics
from ..core.models.unet import EMA, build_model
from ..core.ops import adaptive_curvature_loss, affine_align, tv_loss
from ..utils.misc import ensure_dir, pick_device, set_seed
from ..utils.visualize import save_epoch_visuals, save_training_curve


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

    train_loader, val_loader = build_dataloaders(
        cfg.data, cfg.optim, device, seed=cfg.logging.seed
    )
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

    loss_wrapped_grad = None
    if cfg.loss.w_wrapped_grad > 0:
        from ..core.losses import WrappedGradLoss

        loss_wrapped_grad = WrappedGradLoss(w_grad=1.0)  # weight handled at total sum

    loss_res = None
    if cfg.loss.w_res > 0:
        from ..core.losses import ResidueLoss

        loss_res = ResidueLoss(w_res=1.0)

    loss_ssim = None
    if cfg.loss.u_ssim > 0:
        from ..core.losses import SSIMLoss

        loss_ssim = SSIMLoss(w_ssim=1.0)  # weight handled by cfg

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, cfg.optim.epochs - cfg.optim.warmup_epochs),
        eta_min=cfg.optim.eta_min,
    )
    if cfg.optim.warmup_epochs > 0:
        sched_warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=cfg.optim.warmup_epochs
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[sched_warmup, sched_cosine],
            milestones=[cfg.optim.warmup_epochs],
        )
    else:
        sched = sched_cosine
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, decay=cfg.model.ema_decay)

    start_epoch = 1
    best_mae = float("inf")

    # ------------------
    #  WandB Init
    # ------------------
    if cfg.logging.use_wandb:
        if wandb is None:
            print("Warning: use_wandb=True but wandb package not installed.")
        else:
            wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                config=cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
                resume="allow" if cfg.logging.resume_from else None,
            )

    # ------------------
    #  Resume Logic
    # ------------------
    if cfg.logging.resume_from:
        ckpt_path = cfg.logging.resume_from
        if os.path.exists(ckpt_path):
            print(f"[resume] Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            # Load model weights
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            if "model_ema" in ckpt:
                ema.m.load_state_dict(ckpt["model_ema"])

            # Load optimizer state
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])

            # Load scaler state
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])

            # Load epoch
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
                sched.last_epoch = start_epoch - 1  # sync scheduler
                print(f"[resume] Resumed from epoch {ckpt['epoch']}")

            if "best_mae" in ckpt:
                best_mae = ckpt["best_mae"]

        else:
            print(
                f"Warning: resume_from={ckpt_path} does not exist. Starting from scratch."
            )

    epochs_x: list[int] = []
    train_losses: list[float] = []
    val_maes: list[float] = []

    writer = None
    if cfg.logging.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=cfg.logging.log_dir)

    csv_file = None
    if cfg.logging.log_csv:
        os.makedirs(cfg.logging.out_dir, exist_ok=True)
        csv_path = os.path.join(cfg.logging.out_dir, "train_log.csv")
        # append mode; write header if file is new/empty
        csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        csv_file = open(csv_path, "a", encoding="utf-8")
        if not csv_exists:
            csv_file.write("epoch,train_loss,val_mae,val_rmse,best_mae\n")

    for epoch in range(start_epoch, cfg.optim.epochs + 1):
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

                    # training confidence mask (optionally learned from conf_logit)
                    if cfg.loss.use_conf_weight:
                        conf_used = torch.sigmoid(conf_logit).detach()
                    else:
                        conf_used = torch.ones_like(phi_abs_align)

                    # 1) absolute phase supervision (on aligned prediction)
                    L_phase, parts = loss_sup(
                        phi_abs_align,
                        phi_gt,
                        I_raw if cfg.loss.int_wgrad else None,
                        conf_used,
                    )

                    # 2) curvature smoothness on aligned phase
                    L_curv = cfg.loss.w_curv * adaptive_curvature_loss(
                        phi_abs_align, conf_used
                    )

                    # 3) total variation regularization on aligned phase
                    L_tv = cfg.loss.w_tv * tv_loss(
                        phi_abs_align, conf_used if cfg.loss.use_conf_weight else None
                    )

                    # optional confidence regularizer to avoid degenerate maps
                    L_conf_reg = torch.tensor(0.0, device=device)
                    if cfg.loss.use_conf_weight and cfg.loss.w_conf_reg > 0.0:
                        conf_mean = conf_used.mean()
                        target = 0.5
                        L_conf_reg = (conf_mean - target).abs()

                    # final total loss
                    loss = (
                        cfg.loss.w_data * L_phase
                        + L_curv
                        + L_tv
                        + cfg.loss.w_conf_reg * L_conf_reg
                    )

                    if loss_wrapped_grad is not None:
                        # We apply it on aligned phase; you might argue for raw, but aligned is safer for unwrapping logic?
                        # Actually wrapped grad consistency should hold for both. Aligned is better scaled.
                        L_wg = loss_wrapped_grad(phi_abs_align, phi_gt, conf_used)
                        # The weight is inside the class (w_grad=1.0) * cfg.loss.w_wrapped_grad
                        # Wait, I initialized it with w_grad=1.0. So I should multiply by cfg.loss.w_wrapped_grad here or pass it in init.
                        # In init I passed w_grad=1.0. So I should multiply here.
                        loss = loss + cfg.loss.w_wrapped_grad * L_wg

                    if loss_res is not None:
                        from ..core.ops import get_residue_mask

                        res_gt = get_residue_mask(phi_gt)
                        # conf_logit is repurposed as res_logit in SwinUNet forward
                        L_res = loss_res(conf_logit, res_gt, conf_used)
                        loss = loss + cfg.loss.w_res * L_res
                        parts["res"] = L_res.detach()

                    if loss_ssim is not None:
                        # SSIM expects [B, 1, H, W] in [0, 1] usually.
                        # Our phase is ~[-pi, pi].
                        # A quick hack is to map it to [0, 1] for SSIM calc or just run on raw values.
                        # The implementation in losses.py handles raw values fine but assumes local similarity.
                        # We apply it on aligned phase.
                        L_ssim = loss_ssim(phi_abs_align, phi_gt)
                        loss = loss + cfg.loss.u_ssim * L_ssim
                        parts["ssim"] = L_ssim.detach()

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
                    "ssim": f"{float(parts.get('ssim', 0.0)):.4f}",
                }
            )

        sched.step()
        train_loss = run_loss / max(1, cnt)

        # validation
        cur_val_mae = float("nan")
        cur_val_rmse = float("nan")
        if (epoch % cfg.logging.val_interval == 0) and (val_loader is not None):
            eval_stats = run_eval(ema.m, val_loader, device, use_amp)
            cur_val_mae = eval_stats["MAE"]
            cur_val_rmse = eval_stats["RMSE"]

            # dump visuals using EMA model on a small batch from val
            try:
                I_input_v, phi_gt_v, I_raw_v = next(iter(val_loader))
                I_input_v = I_input_v.to(device)
                phi_gt_v = phi_gt_v.to(device)
                with autocast(enabled=use_amp):
                    phi_raw_v, a_pred_v, b_pred_raw_v, conf_logit_v, k_off_v = ema.m(
                        I_input_v
                    )
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
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_mae": best_mae,
                    },
                    os.path.join(cfg.logging.out_dir, "best.pth"),
                )
        else:
            print(
                f"Epoch {epoch} | train={train_loss:.4f} | time={time.time() - t0:.1f}s"
            )

        # rolling checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "model_ema": ema.m.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "best_mae": best_mae,
            },
            os.path.join(cfg.logging.out_dir, "final.pth"),
        )

        # log to TensorBoard / CSV
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/mae", cur_val_mae, epoch)
            if not np.isnan(cur_val_rmse):
                writer.add_scalar("val/rmse", cur_val_rmse, epoch)
            for i, group in enumerate(opt.param_groups):
                writer.add_scalar(f"optim/lr_group_{i}", group.get("lr", 0.0), epoch)
            writer.add_scalar("val/best_mae", best_mae, epoch)

        if cfg.logging.use_wandb and wandb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "val/mae": cur_val_mae,
                    "val/rmse": cur_val_rmse if not np.isnan(cur_val_rmse) else 0.0,
                    "val/best_mae": best_mae,
                    "epoch": epoch,
                    "lr": opt.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if csv_file is not None:
            csv_file.write(
                f"{epoch},{train_loss},{cur_val_mae},{cur_val_rmse},{best_mae}\n"
            )
            csv_file.flush()

        epochs_x.append(epoch)
        train_losses.append(train_loss)
        val_maes.append(cur_val_mae)

    print("Done. Best MAE:", best_mae)

    # save training curve
    save_training_curve(epochs_x, train_losses, val_maes, cfg.logging.out_dir)

    if writer is not None:
        writer.close()
    if csv_file is not None:
        csv_file.close()
