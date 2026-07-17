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
from tqdm.auto import tqdm

from ..core.config import TrainConfig, config_to_yaml
from ..core.losses import MAEGradLoss

from ..core.utils import ensure_dir, pick_device, set_seed
from ..data import build_dataloaders
from ..data.augmentation import NoiseAug, NoiseScheduler, prepare_batch
from ..model import EMA, build_model


def _log(msg: str) -> None:
    """Unbuffered output for Jupyter/multi-process compatibility."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _create_objective(
    base_cfg: TrainConfig, tune_epochs: int, batch_size_override: Optional[int] = None
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective closure.
    
    Rationale (Architecture & Memory):
    Optuna requires a closure that encapsulates the trial logic. We explicitly `deepcopy(base_cfg)` 
    at the start of every trial to prevent state mutation across sequential runs in the same worker.
    This guarantees mathematically isolated convergences and prevents memory leaks between trials.
    """

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        cfg.optim.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        cfg.loss.w_grad = trial.suggest_float("w_grad", 0.01, 1.0)
        cfg.loss.w_curv = trial.suggest_float("w_curv", 0.0, 0.05)

        if batch_size_override is not None:
            cfg.optim.batch_size = batch_size_override

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
            w_curv=cfg.loss.w_curv,
            intensity_weighted=cfg.loss.int_wgrad,
        ).to(device)

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
        # Rationale (Mathematical Implication):
        # We perform a dummy step to initialize the LR scheduler and silence PyTorch warnings.
        # WARNING: Because this is AdamW, `opt.step()` physically applies weight decay 
        # (w = w - lambda * w) even when gradients are zero. This slightly shrinks the seeded 
        # weight initialization before the first forward pass. It is deterministic, but mathematically non-zero.
        opt.step()
        sched.step()

        scaler = GradScaler(device=device.type, enabled=use_amp)
        ema = EMA(model, decay=cfg.model.ema_decay)

        # Initialize noise scheduler for tuning
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

        best_mae = float("inf")

        for epoch in range(1, tune_epochs + 1):
            if noise_sched is not None and train_aug is not None:
                current_noise = noise_sched.level(epoch)
                train_aug.set_level(current_noise)

            model.train()
            pbar = tqdm(
                train_loader,
                desc=f"Trial {trial.number} Ep {epoch}/{tune_epochs}",
                leave=False,
                file=sys.stderr,
            )
            run_loss = 0.0

            for step, (I_raw, phi_gt) in enumerate(pbar):
                I_raw = I_raw.to(device)
                phi_gt = phi_gt.to(device)
                
                I_input, phi_gt, I_raw_n, _ = prepare_batch(
                    I_raw, phi_gt, noise_aug=train_aug, hint_mode=cfg.aug.hint_mode
                )

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    phi_raw, k_off = model(I_input)

                    if isinstance(phi_raw, list):
                        phi_abs = [p + k_off for p in phi_raw]
                    else:
                        phi_abs = phi_raw + k_off

                    L_phase, _ = loss_fn(
                        phi_abs, phi_gt, I_raw_n if cfg.loss.int_wgrad else None
                    )
                    loss = cfg.loss.w_data * L_phase

                if not torch.isfinite(loss):
                    _log(f"  Trial {trial.number}: NaN/Inf at epoch {epoch}")
                    raise optuna.TrialPruned()

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.grad_clip, error_if_nonfinite=False
                )
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after >= scale_before:
                    sched.step()
                ema.update(model)

                run_loss += float(loss.detach())
                if step % 10 == 0:
                    pbar.set_postfix(loss=f"{run_loss / (step + 1):.4f}")

            if val_loader is not None:
                ema.m.eval()
                total_mae = 0.0
                n = 0
                with torch.no_grad():
                    for I_raw, phi_gt in val_loader:
                        I_raw = I_raw.to(device)
                        phi_gt = phi_gt.to(device)
                        I_input, phi_gt, _, _ = prepare_batch(
                            I_raw,
                            phi_gt,
                            noise_aug=None,
                            hint_mode=cfg.aug.hint_mode,
                        )
                        with autocast(device_type=device.type, enabled=use_amp):
                            phi_raw, k_off = ema.m(I_input)
                            phi_abs = phi_raw + k_off
                        # HPO objective: raw AbsMAE (no GT scale fit; matches train selection)
                        total_mae += float(
                            (phi_abs - phi_gt).abs().mean()
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

        # Rationale (Memory Efficiency):
        # We explicitly sever the Python references to the VRAM-heavy computational graphs 
        # and force a CUDA garbage collection. If we rely strictly on Python's GC during sequential 
        # trials in the same worker process, fragmentation will cause creeping OOMs by trial 15.
        del model
        del opt
        del scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    batch_size_override: Optional[int] = None,
) -> None:
    """Worker function to run trials on a specific GPU."""
    # Set device for this worker
    device = torch.device(f"cuda:{gpu_id}")
    cfg.model.device = str(device)

    # Create objective with this device
    objective = _create_objective(
        cfg, tune_epochs, batch_size_override=batch_size_override
    )

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
            # Rationale (Time Complexity):
            # Checking global completion via `len(study.trials)` scales O(T) within this specific study.
            # The previous implementation used `optuna.get_all_study_summaries()`, which parses the 
            # ENTIRE SQLite database metadata across all historic studies, causing an O(N^2) global 
            # bottleneck that paralyzes the database as trial counts increase.
            current_trial_count = len(study.trials)

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
    batch_size_override: Optional[int] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search (TPE + MedianPruner, SQLite resume).

    Args:
        n_workers: Number of parallel workers (default 1). Set to number of GPUs for parallel trials.
        gpu_ids: List of GPU IDs to use. If None, uses [0, 1, ..., n_workers-1].
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

    if n_workers > 1 and torch.cuda.is_available():
        # Parallel execution
        if gpu_ids is None:
            n_gpus = torch.cuda.device_count()
            gpu_ids = [i % n_gpus for i in range(n_workers)]

        n_workers = min(n_workers, len(gpu_ids), remaining)
        trials_per_worker = remaining // n_workers
        extra_trials = remaining % n_workers

        print(f"Parallel HPO: {n_workers} workers on GPUs {gpu_ids}")
        print(f"Trials per worker: ~{trials_per_worker}")

        # Rationale (Thread Safety):
        # We MUST force the multiprocessing start method to 'spawn' rather than 'fork'.
        # 'fork' blindly duplicates the memory space, which severely corrupts the CUDA runtime 
        # state and causes immediate deadlocks when initializing tensors across multiple child 
        # processes. 'spawn' guarantees a clean, uncorrupted Python interpreter boot for every worker.
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
                    batch_size_override,
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
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

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
    best_cfg.model.base = study.best_params["base"]
    best_cfg.optim.warmup_steps = study.best_params["warmup_steps"]

    best_config_path = os.path.join(optuna_dir, "best_config.yaml")
    Path(best_config_path).write_text(config_to_yaml(best_cfg))
    print(f"\nBest config saved: {best_config_path}")

    return study
