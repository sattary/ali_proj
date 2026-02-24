"""
Optuna-based hyperparameter tuning with multi-GPU parallel trial support.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional
import multiprocessing as mp
import time

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


def _log(msg: str) -> None:
    """Unbuffered output for Jupyter/multi-process compatibility."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _create_objective(
    base_cfg: TrainConfig, tune_epochs: int, batch_size_override: Optional[int] = None
):
    """Build an Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        cfg.optim.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        cfg.loss.w_grad = trial.suggest_float("w_grad", 0.01, 1.0)
        cfg.loss.w_curv = trial.suggest_float("w_curv", 0.0, 0.05)

        if batch_size_override is not None:
            cfg.optim.batch_size = batch_size_override
        else:
            cfg.optim.batch_size = trial.suggest_categorical(
                "batch_size", [8, 16, 32, 64]
            )

        cfg.model.base = trial.suggest_categorical("base", [8, 16, 32])
        cfg.model.activation = trial.suggest_categorical("activation", ["mish", "silu"])

        cfg.optim.warmup_steps = trial.suggest_categorical(
            "warmup_steps", [100, 500, 1000]
        )

        # Tune the final actual noise boundaries, assuming a full run of base_cfg.optim.epochs
        full_run_epochs = max(1, base_cfg.optim.epochs)
        if getattr(cfg, "aug", None) and cfg.aug.enable:
            cfg.aug.warmup_epochs = trial.suggest_int(
                "noise_warmup", 0, max(1, int(full_run_epochs * 0.4))
            )
            cfg.aug.full_epoch = trial.suggest_int(
                "noise_full", max(2, int(full_run_epochs * 0.5)), full_run_epochs
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
            train_loader, val_loader, _ = build_dataloaders(
                cfg, device, seed=cfg.logging.seed
            )
        except Exception as e:
            _log(f"  Trial {trial.number}: data loading failed: {e}")
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

        # Initialize noise scheduler for tuning
        noise_sched = None
        if getattr(cfg, "aug", None) and cfg.aug.enable:
            from ..data.augmentation import NoiseScheduler

            # Scale the tuned absolute epochs proportionally for this short tuning run
            scale_factor = tune_epochs / max(1, base_cfg.optim.epochs)
            scaled_warmup = int(cfg.aug.warmup_epochs * scale_factor)
            scaled_full_epoch = max(
                scaled_warmup + 1, int(cfg.aug.full_epoch * scale_factor)
            )

            noise_sched = NoiseScheduler(
                full_epoch=scaled_full_epoch,
                warmup_epochs=scaled_warmup,
            )

        best_mae = float("inf")

        for epoch in range(1, tune_epochs + 1):
            if noise_sched is not None and train_loader.dataset.noise_aug is not None:
                current_noise = noise_sched.level(epoch)
                train_loader.dataset.noise_aug.set_level(current_noise)

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
                    _log(f"  Trial {trial.number}: NaN/Inf at epoch {epoch}")
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

                _log(
                    f"  Trial {trial.number} | epoch {epoch}/{tune_epochs} | "
                    f"val MAE={val_mae:.4f} (best={best_mae:.4f})"
                )

        return best_mae

    return objective


def _run_trial_worker(
    gpu_id: int,
    cfg: TrainConfig,
    tune_epochs: int,
    study_name: str,
    storage: str,
    n_trials: int,
    worker_id: int,
    total_target: int,
):
    """Worker function to run trials on a specific GPU."""
    # Set device for this worker
    device = torch.device(f"cuda:{gpu_id}")
    cfg.model.device = str(device)

    # Create objective with this device
    objective = _create_objective(
        cfg, tune_epochs, batch_size_override=None
    )  # Override passed via cfg already outside this func, handled below

    # Load study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    _log(f"[Worker {worker_id} on GPU {gpu_id}] Starting...")

    # Run trials until we've reached n_trials for this worker
    trial_count = 0
    while trial_count < n_trials:
        try:
            # Check if study has reached total target
            study_summary = optuna.get_all_study_summaries(storage)
            current_trial_count = sum(
                s.n_trials for s in study_summary if s.study_name == study_name
            )

            if current_trial_count >= total_target:
                _log(
                    f"[Worker {worker_id} on GPU {gpu_id}] Study reached target ({current_trial_count}/{total_target})"
                )
                break

            study.optimize(objective, n_trials=1, show_progress_bar=False)
            trial_count += 1
            _log(f"[Worker {worker_id} on GPU {gpu_id}] Completed trial {trial_count}")
        except Exception as e:
            _log(f"[Worker {worker_id} on GPU {gpu_id}] Error: {e}")
            time.sleep(1)  # Brief pause before retry

    _log(f"[Worker {worker_id} on GPU {gpu_id}] Finished {trial_count} trials")


def run_tuning(
    cfg: TrainConfig,
    n_trials: int = 50,
    tune_epochs: int = 15,
    study_name: str = "phase_unwrap_hpo",
    storage: Optional[str] = None,
    n_workers: int = 1,
    gpu_ids: Optional[list[int]] = None,
    auto_push_callback: Optional[Callable[[], None]] = None,
    batch_size_override: Optional[int] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search (TPE + MedianPruner, SQLite resume).

    Args:
        n_workers: Number of parallel workers (default 1). Set to number of GPUs for parallel trials.
        gpu_ids: List of GPU IDs to use. If None, uses [0, 1, ..., n_workers-1].
        auto_push_callback: Optional callback to run after HPO completes (e.g., push to GitHub).
    """
    optuna_dir = os.path.join(cfg.logging.runs_root, "optuna")
    ensure_dir(optuna_dir)

    if storage is None:
        db_path = os.path.join(optuna_dir, f"{study_name}.db")
        storage = f"sqlite:///{db_path}"

    # Create study first (in main process)
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
        print(f"Study already has {len(study.trials)} trials (requested {n_trials}).")
        return study

    print(
        f"Running {remaining} trials ({len(study.trials)} existing, {n_trials} target)"
    )

    if n_workers > 1 and torch.cuda.device_count() > 1:
        # Parallel execution on multiple GPUs
        if gpu_ids is None:
            gpu_ids = list(range(min(n_workers, torch.cuda.device_count())))

        n_workers = min(n_workers, len(gpu_ids), remaining)
        trials_per_worker = remaining // n_workers
        extra_trials = remaining % n_workers

        print(f"Parallel HPO: {n_workers} workers on GPUs {gpu_ids}")
        print(f"Trials per worker: ~{trials_per_worker}")

        # Use spawn method for CUDA compatibility
        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(gpu_ids[:n_workers]):
            worker_trials = trials_per_worker + (1 if i < extra_trials else 0)
            p = mp.Process(
                target=_run_trial_worker,
                args=(
                    gpu_id,
                    cfg,
                    tune_epochs,
                    study_name,
                    storage,
                    worker_trials,
                    i,
                    n_trials,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all workers
        for p in processes:
            p.join()

        # Reload study to get results
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        # Sequential execution
        objective = _create_objective(
            cfg, tune_epochs, batch_size_override=batch_size_override
        )
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Check if any trials completed successfully
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("\n" + "=" * 60)
        print("WARNING: No trials completed successfully!")
        print("=" * 60)
        print("Possible causes:")
        print("  - Data directory not found or empty")
        print("  - Data loading errors")
        print("  - GPU out of memory")
        print("\nCheck logs above for trial errors.")
        print("=" * 60)

        # Run auto-push callback even if no trials (may include partial results)
        if auto_push_callback is not None:
            try:
                auto_push_callback()
            except Exception as e:
                print(f"\nWarning: Auto-push callback failed: {e}")

        return study

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
    best_cfg.model.activation = study.best_params["activation"]
    best_cfg.optim.warmup_steps = study.best_params["warmup_steps"]

    if "noise_warmup" in study.best_params and getattr(best_cfg, "aug", None):
        best_cfg.aug.warmup_epochs = study.best_params["noise_warmup"]
        best_cfg.aug.full_epoch = study.best_params["noise_full"]

    best_config_path = os.path.join(optuna_dir, "best_config.yaml")
    Path(best_config_path).write_text(config_to_yaml(best_cfg))
    print(f"\nBest config saved: {best_config_path}")

    # Run auto-push callback if provided (e.g., push optuna results to GitHub)
    if auto_push_callback is not None:
        try:
            auto_push_callback()
        except Exception as e:
            print(f"\nWarning: Auto-push callback failed: {e}")

    return study
