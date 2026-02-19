"""
Optuna-based hyperparameter tuning.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.amp import GradScaler, autocast

from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import MAEGradLoss
from ..core.ops import affine_align, curvature_loss
from ..core.utils import ensure_dir, pick_device, set_seed
from ..data import build_dataloaders
from ..model import EMA, build_model


def _create_objective(base_cfg: TrainConfig, tune_epochs: int):
    """Build an Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        cfg.optim.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        cfg.loss.w_grad = trial.suggest_float("w_grad", 0.01, 1.0)
        cfg.loss.w_curv = trial.suggest_float("w_curv", 0.0, 0.05)
        cfg.optim.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        cfg.model.base = trial.suggest_categorical("base", [8, 16, 32])
        cfg.optim.warmup_steps = trial.suggest_categorical(
            "warmup_steps", [100, 500, 1000]
        )

        cfg.logging.run_name = f"optuna/trial_{trial.number:04d}"
        cfg.optim.epochs = tune_epochs

        set_seed(cfg.logging.seed)
        device = pick_device(cfg.model.device)
        use_amp = device.type == "cuda" and cfg.model.use_amp

        run_dir = cfg.logging.run_dir
        ensure_dir(run_dir)
        Path(os.path.join(run_dir, "config.yaml")).write_text(config_to_yaml(cfg))

        try:
            train_loader, val_loader = build_dataloaders(
                cfg.data, cfg.optim, device, seed=cfg.logging.seed
            )
        except Exception as e:
            print(f"  Trial {trial.number}: data loading failed: {e}")
            raise optuna.TrialPruned()

        model = build_model(cfg.model).to(device)
        loss_fn = MAEGradLoss(
            w_mae=cfg.loss.w_mae,
            w_grad=cfg.loss.w_grad,
            intensity_weighted=cfg.loss.int_wgrad,
        )

        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )

        total_steps = tune_epochs * len(train_loader)
        warmup_steps = min(cfg.optim.warmup_steps, total_steps // 2)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.optim.eta_min
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )
        scaler = GradScaler(device=device.type, enabled=use_amp)
        ema = EMA(model, decay=cfg.model.ema_decay)

        best_mae = float("inf")

        for epoch in range(1, tune_epochs + 1):
            model.train()
            for I_input, phi_gt, I_raw in train_loader:
                I_input = I_input.to(device)
                phi_gt = phi_gt.to(device)
                I_raw = I_raw.to(device)

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    phi_raw, k_off = model(I_input)
                    phi_abs = phi_raw + k_off
                    L_phase, _ = loss_fn(
                        phi_abs, phi_gt, I_raw if cfg.loss.int_wgrad else None
                    )
                    L_curv = cfg.loss.w_curv * curvature_loss(phi_abs)
                    loss = cfg.loss.w_data * L_phase + L_curv

                if not torch.isfinite(loss):
                    print(f"  Trial {trial.number}: NaN/Inf at epoch {epoch}")
                    raise optuna.TrialPruned()

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.grad_clip, error_if_nonfinite=False
                )
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(model)

            if val_loader is not None:
                ema.m.eval()
                total_mae = 0.0
                n = 0
                with torch.no_grad():
                    for I_input, phi_gt, _ in val_loader:
                        I_input = I_input.to(device)
                        phi_gt = phi_gt.to(device)
                        with autocast(device_type=device.type, enabled=use_amp):
                            phi_raw, k_off = ema.m(I_input)
                            phi_abs = phi_raw + k_off
                        aligned, _, _ = affine_align(phi_abs, phi_gt)
                        total_mae += float(
                            (aligned - phi_gt).abs().mean()
                        ) * I_input.size(0)
                        n += I_input.size(0)

                val_mae = total_mae / max(1, n)
                best_mae = min(best_mae, val_mae)

                trial.report(val_mae, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                print(
                    f"  Trial {trial.number} | epoch {epoch}/{tune_epochs} | "
                    f"val MAE={val_mae:.4f} (best={best_mae:.4f})"
                )

        return best_mae

    return objective


def run_tuning(
    cfg: TrainConfig,
    n_trials: int = 50,
    tune_epochs: int = 15,
    study_name: str = "phase_unwrap_hpo",
    storage: Optional[str] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search (TPE + MedianPruner, SQLite resume)."""
    optuna_dir = os.path.join(cfg.logging.runs_root, "optuna")
    ensure_dir(optuna_dir)

    if storage is None:
        db_path = os.path.join(optuna_dir, f"{study_name}.db")
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=cfg.logging.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )

    objective = _create_objective(cfg, tune_epochs)

    remaining = n_trials - len(study.trials)
    if remaining <= 0:
        print(f"Study already has {len(study.trials)} trials (requested {n_trials}).")
    else:
        print(
            f"Running {remaining} trials "
            f"({len(study.trials)} existing, {n_trials} target)"
        )
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  Best val MAE: {study.best_value:.6f}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    best_cfg = deepcopy(cfg)
    best_cfg.optim.lr = study.best_params["lr"]
    best_cfg.loss.w_grad = study.best_params["w_grad"]
    best_cfg.loss.w_curv = study.best_params["w_curv"]
    best_cfg.optim.batch_size = study.best_params["batch_size"]
    best_cfg.model.base = study.best_params["base"]
    best_cfg.optim.warmup_steps = study.best_params["warmup_steps"]

    best_config_path = os.path.join(optuna_dir, "best_config.yaml")
    Path(best_config_path).write_text(config_to_yaml(best_cfg))
    print(f"\nBest config saved: {best_config_path}")

    return study
