"""
Optuna-based hyperparameter tuning for PCLCNModel.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import PCLNCLoss
from ..core.ops import piston_align
from ..core.utils import ensure_dir, pick_device, set_seed
from ..data import build_dataloaders
from ..model import EMA, build_model


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _create_objective(
    base_cfg: TrainConfig, tune_epochs: int, batch_size_override: Optional[int] = None
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        cfg.optim.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        cfg.loss.w_curl = trial.suggest_float("w_curl", 0.01, 1.0)
        cfg.loss.w_data = trial.suggest_float("w_data", 0.5, 2.0)

        if batch_size_override is not None:
            cfg.optim.batch_size = batch_size_override

        cfg.logging.run_name = f"optuna/trial_{trial.number:04d}"
        cfg.optim.epochs = tune_epochs

        set_seed(cfg.logging.seed)
        device = pick_device(cfg.model.device)

        run_dir = cfg.logging.run_dir
        ensure_dir(run_dir)
        Path(os.path.join(run_dir, "config.yaml")).write_text(config_to_yaml(cfg))

        try:
            train_loader, val_loader, _ = build_dataloaders(cfg, device, seed=cfg.logging.seed)
        except Exception as e:
            _log(f"  Trial {trial.number}: data loading failed: {e}")
            raise optuna.TrialPruned()

        model = build_model(cfg.model).to(device)
        loss_fn = PCLNCLoss(
            w_data=cfg.loss.w_data,
            w_curl=cfg.loss.w_curl,
            w_zernike=cfg.loss.w_zernike,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        ema = EMA(model, decay=cfg.model.ema_decay)

        best_mae = float("inf")

        for epoch in range(1, tune_epochs + 1):
            model.train()
            for batch in train_loader:
                I_raw, phi_gt, grad_phi2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                opt.zero_grad(set_to_none=True)

                phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)
                loss, _ = loss_fn(phi_final, phi_gt, gx_tilde, gy_tilde, c_zernike, phi_zernike)

                if not torch.isfinite(loss):
                    _log(f"  Trial {trial.number}: NaN/Inf at epoch {epoch}")
                    raise optuna.TrialPruned()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                opt.step()
                ema.update(model)

            if val_loader is not None:
                ema.m.eval()
                total_mae = 0.0
                n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        I_raw, phi_gt, grad_phi2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        phi_abs, _, _, _, _ = ema.m(I_raw, grad_phi2)
                        phi_aligned, _ = piston_align(phi_abs, phi_gt)
                        total_mae += float((phi_aligned - phi_gt).abs().mean()) * I_raw.size(0)
                        n += I_raw.size(0)

                val_mae = total_mae / max(1, n)
                best_mae = min(best_mae, val_mae)

                trial.report(val_mae, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        del model
        del opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_mae

    return objective


def run_tuning(
    cfg: TrainConfig,
    n_trials: int = 50,
    tune_epochs: int = 15,
    study_name: str = "phase_unwrap_hpo",
    storage: Optional[str] = None,
    n_workers: int = 1,
    gpu_ids: Optional[list[int]] = None,
    batch_size_override: Optional[int] = None,
) -> optuna.Study:
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

    remaining = n_trials - len(study.trials)
    if remaining <= 0:
        return study

    objective = _create_objective(cfg, tune_epochs, batch_size_override=batch_size_override)
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_cfg = deepcopy(cfg)
    best_cfg.optim.lr = study.best_params["lr"]
    best_cfg.loss.w_curl = study.best_params["w_curl"]
    best_cfg.loss.w_data = study.best_params["w_data"]

    best_config_path = os.path.join(optuna_dir, "best_config.yaml")
    Path(best_config_path).write_text(config_to_yaml(best_cfg))
    return study
