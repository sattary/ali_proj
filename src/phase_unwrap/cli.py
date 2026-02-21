"""
CLI for the phase unwrapping package.

Commands:
    train      -- Train the UNetRes2 model.
    generate   -- Generate synthetic interferogram data to HDF5 shards.
    multiseed  -- Run N training runs with different seeds, aggregate results.
    plot       -- Generate publication-quality figures (subcommands).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .core.config import TrainConfig, load_train_config
from .training.train import train as run_train

app = typer.Typer(help="Phase unwrapping / absolute phase reconstruction CLI.")


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="JSON or YAML config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    run_name: Optional[str] = typer.Option(
        None, "--run-name", help="Name for this training run."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="'auto', 'cuda', or 'cpu'."
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Override training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Path to checkpoint for resume."
    ),
    # Auto-push options
    auto_push_interval: Optional[int] = typer.Option(
        None,
        "--auto-push-interval",
        help="Enable auto-push every N epochs (Kaggle/Colab only).",
    ),
    auto_push_dry_run: bool = typer.Option(
        False,
        "--auto-push-dry-run",
        help="Test auto-push setup without actually pushing.",
    ),
    force_auto_push: bool = typer.Option(
        False,
        "--force-auto-push",
        help="Force auto-push even outside Kaggle/Colab (for testing).",
    ),
    auto_push_pat: Optional[str] = typer.Option(
        None, "--auto-push-pat", help="GitHub PAT (or set GITHUB_PAT env var)."
    ),
    # Multi-GPU options
    multi_gpu: bool = typer.Option(
        False,
        "--multi-gpu",
        help="Use all available GPUs with DataParallel.",
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU IDs (e.g., '0,1'). Default: use all.",
    ),
) -> None:
    """Train the UNetRes2 absolute phase reconstruction model."""
    import torch

    cfg: TrainConfig = load_train_config(config)

    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if run_name is not None:
        cfg.logging.run_name = run_name
    if device is not None:
        cfg.model.device = device
    if epochs is not None:
        cfg.optim.epochs = epochs
    if batch_size is not None:
        cfg.optim.batch_size = batch_size

    # Parse GPU IDs if provided
    gpu_id_list: Optional[list[int]] = None
    if gpu_ids is not None:
        gpu_id_list = [int(x.strip()) for x in gpu_ids.split(",")]

    # Auto-enable multi-GPU on Kaggle if 2+ GPUs detected
    if not multi_gpu and torch.cuda.is_available():
        from .training.multi_gpu import detect_kaggle_multi_gpu

        if detect_kaggle_multi_gpu():
            multi_gpu = True
            print("[kaggle] Auto-enabling multi-GPU mode")

    # Create auto-push callback if requested
    auto_push_callback = None
    if auto_push_interval is not None:
        from .git_automation import AutoPushCallback
        from .git_automation.cli_integration import validate_auto_push_config

        validate_auto_push_config(
            auto_push_interval=auto_push_interval,
            auto_push_dry_run=auto_push_dry_run,
            force_auto_push=force_auto_push,
        )

        auto_push_callback = AutoPushCallback(
            run_dir=cfg.logging.run_dir,
            push_interval=auto_push_interval,
            pat=auto_push_pat,
            dry_run=auto_push_dry_run,
            force=force_auto_push,
            include_checkpoints=True,
        )

    run_train(
        cfg,
        resume_path=str(resume) if resume else None,
        auto_push_callback=auto_push_callback,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_id_list,
    )


@app.command()
def generate(
    num_samples: int = typer.Option(
        180_000, "--num-samples", help="Total samples to generate."
    ),
    shard_size: int = typer.Option(
        1000, "--shard-size", help="Samples per HDF5 shard."
    ),
    out_dir: str = typer.Option(
        "data/full", "--out-dir", help="Output directory for shards."
    ),
    seed: int = typer.Option(1337, "--seed", help="RNG seed."),
) -> None:
    """Generate synthetic interferogram data to HDF5 shards."""
    from .data.generate import generate_to_h5

    generate_to_h5(
        out_dir=out_dir, num_samples=num_samples, shard_size=shard_size, seed=seed
    )


@app.command()
def multiseed(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="JSON or YAML config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    run_name: str = typer.Option(
        "multiseed", "--run-name", help="Base name for the run group."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="'auto', 'cuda', or 'cpu'."
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Override training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    seeds: Optional[str] = typer.Option(
        None, "--seeds", help="Comma-separated seeds (e.g. 1337,42,7)."
    ),
    num_seeds: int = typer.Option(
        3, "--num-seeds", help="Auto-generate N seeds (ignored if --seeds given)."
    ),
) -> None:
    """Run N training runs with different seeds, then aggregate results."""
    from .training.multiseed import run_multiseed

    cfg: TrainConfig = load_train_config(config)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device
    if epochs is not None:
        cfg.optim.epochs = epochs
    if batch_size is not None:
        cfg.optim.batch_size = batch_size

    seed_list: list[int]
    if seeds is not None:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    else:
        import random as _rnd

        seed_list = [_rnd.randint(0, 2**31) for _ in range(num_seeds)]

    run_multiseed(cfg, base_run_name=run_name, seeds=seed_list)


def _create_tune_auto_push_callback(
    optuna_dir: str,
    data_dir: str,
    push_interval: int,
    pat: Optional[str],
    dry_run: bool,
):
    """Create callback for pushing optuna results to GitHub after HPO completes."""
    from datetime import datetime
    from pathlib import Path
    import zipfile

    from .git_automation import GitPusher, ZipPacker

    def callback():
        """Push optuna results after HPO finishes."""
        from .git_automation.environment import is_kaggle, is_colab

        if not is_kaggle() and not is_colab() and not dry_run:
            print("[auto-push] Not on Kaggle/Colab, skipping push")
            return

        print("\n" + "=" * 60)
        print("Pushing Optuna results to GitHub...")
        print("=" * 60)

        # Load data config if exists
        data_config_path = Path(data_dir) / "data_config.yaml"
        if data_config_path.exists():
            import yaml

            with open(data_config_path) as f:
                data_config = yaml.safe_load(f)
            data_name = Path(data_config["data_dir"]).name
            num_samples = data_config.get("num_samples", 0)
            seed = data_config.get("seed", 0)
        else:
            data_name = Path(data_dir).name
            num_samples = 0
            seed = 0

        # Create zip packer for optuna directory
        optuna_path = Path(optuna_dir) / "optuna"
        if not optuna_path.exists():
            print(f"Optuna directory not found: {optuna_path}")
            return

        packer = ZipPacker(
            run_dir=str(optuna_path),
            include_checkpoints=False,
        )

        # Create zip filename following training convention
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_samples > 0:
            zip_name = f"optuna_{data_name}_n{num_samples}_s{seed}_{timestamp}.zip"
        else:
            zip_name = f"optuna_{data_name}_{timestamp}.zip"

        zip_path = optuna_path / zip_name

        # Copy data config to optuna dir for inclusion in zip
        if data_config_path.exists():
            import shutil

            dest_data_config = optuna_path / "data_config.yaml"
            shutil.copy2(data_config_path, dest_data_config)

        # Create zip
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in optuna_path.rglob("*"):
                if file_path.is_file() and file_path.suffix != ".zip":
                    arcname = file_path.relative_to(optuna_path)
                    zf.write(file_path, arcname)

        print(f"Created zip: {zip_path.name}")

        # Push to GitHub
        branch_name = "artifacts"
        pusher = GitPusher(
            repo_dir=str(Path.cwd()),
            branch_name=branch_name,
            pat=pat,
            dry_run=dry_run,
        )

        try:
            pusher.setup_branch()
            success = pusher.push_artifact(
                zip_path=str(zip_path),
                epoch=0,
                total_epochs=0,
                metrics={"n_trials": 0},
                is_final=True,
            )
            if success:
                print("=" * 60)
                print("✓ Optuna results pushed successfully!")
                print("=" * 60)
            else:
                print("✗ Push failed")
        except Exception as e:
            print(f"✗ Push error: {e}")

    return callback


@app.command()
def tune(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="JSON or YAML config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="'auto', 'cuda', or 'cpu'."
    ),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials."),
    tune_epochs: int = typer.Option(15, "--tune-epochs", help="Epochs per trial."),
    study_name: str = typer.Option(
        "phase_unwrap_hpo", "--study-name", help="Optuna study name."
    ),
    n_workers: int = typer.Option(
        1, "--n-workers", "-j", help="Parallel workers (use #GPUs for best speed)."
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU IDs for parallel trials (e.g., '0,1').",
    ),
    # Auto-push options
    auto_push: bool = typer.Option(
        False,
        "--auto-push",
        help="Push optuna results to GitHub after HPO completes.",
    ),
    auto_push_pat: Optional[str] = typer.Option(
        None, "--auto-push-pat", help="GitHub PAT (or set GITHUB_PAT env var)."
    ),
    auto_push_dry_run: bool = typer.Option(
        False,
        "--auto-push-dry-run",
        help="Test auto-push setup without actually pushing.",
    ),
) -> None:
    """Run Optuna hyperparameter search (TPE + MedianPruner).

    Use --n-workers 2 --gpu-ids 0,1 on Kaggle to run 2 trials in parallel on 2 GPUs.
    """
    import torch
    from .training.tune import run_tuning

    cfg: TrainConfig = load_train_config(config)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device

    # Auto-detect GPUs if not specified
    gpu_id_list: Optional[list[int]] = None
    if gpu_ids is not None:
        gpu_id_list = [int(x.strip()) for x in gpu_ids.split(",")]
    elif n_workers > 1 and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            gpu_id_list = list(range(min(n_workers, n_gpus)))
            print(f"Auto-detected {n_gpus} GPUs, using: {gpu_id_list}")

    # Auto-push callback for after HPO completes
    auto_push_callback = None
    if auto_push:
        from .git_automation.environment import is_kaggle, is_colab

        # Validate: need PAT or dry-run, and be on cloud environment
        if not auto_push_dry_run and not is_kaggle() and not is_colab():
            print(
                "[auto-push] Warning: Not on Kaggle/Colab. Use --auto-push-dry-run to test."
            )

        data_dir_str = str(data_dir) if data_dir else cfg.data.data_dir
        auto_push_callback = _create_tune_auto_push_callback(
            optuna_dir=cfg.logging.run_dir,
            data_dir=data_dir_str,
            push_interval=0,  # Push only at end
            pat=auto_push_pat,
            dry_run=auto_push_dry_run,
        )

        data_dir_str = str(data_dir) if data_dir else cfg.data.data_dir
        auto_push_callback = _create_tune_auto_push_callback(
            optuna_dir=cfg.logging.run_dir,
            data_dir=data_dir_str,
            push_interval=0,  # Push only at end
            pat=auto_push_pat,
            dry_run=auto_push_dry_run,
        )

    run_tuning(
        cfg,
        n_trials=n_trials,
        tune_epochs=tune_epochs,
        study_name=study_name,
        n_workers=n_workers,
        gpu_ids=gpu_id_list,
        auto_push_callback=auto_push_callback,
    )


# ---------------------------------------------------------------------------
# Plot sub-app
# ---------------------------------------------------------------------------
plot_app = typer.Typer(help="Generate publication-quality figures.")
app.add_typer(plot_app, name="plot")


@plot_app.command("training-curve")
def plot_training_curve_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
    no_lr: bool = typer.Option(False, "--no-lr", help="Omit LR subplot."),
) -> None:
    """Dual-axis training loss + val MAE. Supports multi-seed (shaded bands)."""
    from .visualize import plot_training_curve

    plot_training_curve(run_dir, out_path=out, show_lr=not no_lr)


@plot_app.command("qualitative")
def plot_qualitative_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/qualitative_grid.png", "--out"),
    n_samples: int = typer.Option(4, "--n-samples", help="Number of rows in the grid."),
    config: Optional[str] = typer.Option(
        None, "--config", help="Override config file."
    ),
) -> None:
    """Interferogram | GT | Prediction | Error grid."""
    from .visualize import plot_qualitative_grid

    plot_qualitative_grid(
        checkpoint, data_dir, out_path=out, n_samples=n_samples, config_path=config
    )


@plot_app.command("phase-profile")
def plot_phase_profile_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/phase_profile.png", "--out"),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Which sample to plot."),
    config: Optional[str] = typer.Option(
        None, "--config", help="Override config file."
    ),
) -> None:
    """1D cross-section through center row/column: GT vs prediction with residuals."""
    from .visualize import plot_phase_profile

    plot_phase_profile(
        checkpoint, data_dir, out_path=out, sample_idx=sample_idx, config_path=config
    )


@plot_app.command("error-hist")
def plot_error_hist_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/error_histogram.png", "--out"),
    config: Optional[str] = typer.Option(
        None, "--config", help="Override config file."
    ),
) -> None:
    """Per-sample MAE histogram + CDF with annotated percentiles."""
    from .visualize import plot_error_histogram

    plot_error_histogram(checkpoint, data_dir, out_path=out, config_path=config)


@plot_app.command("loss-landscape")
def plot_loss_landscape_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/loss_landscape.png", "--out"),
    grid_size: int = typer.Option(31, "--grid-size", help="Resolution of 2D grid."),
    alpha_range: float = typer.Option(1.0, "--alpha-range", help="Perturbation range."),
    num_eval_samples: int = typer.Option(
        500, "--num-eval-samples", help="Samples per loss evaluation."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Override config file."
    ),
) -> None:
    """2D loss surface contour (Li et al., 2018 filter-normalized)."""
    from .visualize import plot_loss_landscape

    plot_loss_landscape(
        checkpoint,
        data_dir,
        out_path=out,
        grid_size=grid_size,
        alpha_range=alpha_range,
        num_eval_samples=num_eval_samples,
        config_path=config,
    )


@plot_app.command("convergence")
def plot_convergence_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
) -> None:
    """Loss components + learning rate schedule."""
    from .visualize import plot_convergence

    plot_convergence(run_dir, out_path=out)


# -- GradCAM (plot sub-command)
@plot_app.command("gradcam")
def plot_gradcam_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/gradcam.png", "--out"),
    n_samples: int = typer.Option(4, "--n-samples"),
    layer: str = typer.Option("enc5", "--layer", help="Target encoder layer."),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """GradCAM attention overlay on interferograms."""
    from .analysis.gradcam import plot_gradcam

    plot_gradcam(
        checkpoint,
        data_dir,
        out_path=out,
        n_samples=n_samples,
        target_layer_name=layer,
        config_path=config,
    )


# -- Baseline Comparison Grid
@plot_app.command("baseline-comparison")
def plot_baseline_comparison_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/baseline_comparison.png", "--out"),
    n_samples: int = typer.Option(4, "--n-samples"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Visual Baseline Superiority Matrix."""
    from .visualize.baseline_comparison_grid import plot_baseline_comparison

    plot_baseline_comparison(
        checkpoint, data_dir, out_path=out, n_samples=n_samples, config_path=config
    )


# -- Noise Degradation Grid
@plot_app.command("noise-degradation")
def plot_noise_degradation_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option("results/figs/noise_degradation.png", "--out"),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Target dataset index."),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Iterative noise degradation evaluation."""
    from .visualize.noise_degradation_grid import plot_noise_degradation

    plot_noise_degradation(
        checkpoint, data_dir, out_path=out, sample_idx=sample_idx, config_path=config
    )


# ---------------------------------------------------------------------------
# Export commands
# ---------------------------------------------------------------------------
@app.command("export-onnx")
def export_onnx_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained checkpoint."),
    out: str = typer.Option("results/model.onnx", "--out"),
    opset: int = typer.Option(17, "--opset"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Export model to ONNX format."""
    from .analysis.export import export_onnx

    export_onnx(checkpoint, out_path=out, opset=opset, config_path=config)


@app.command("export-torchscript")
def export_ts_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained checkpoint."),
    out: str = typer.Option("results/model.pt", "--out"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Export model to TorchScript (traced) format."""
    from .analysis.export import export_torchscript

    export_torchscript(checkpoint, out_path=out, config_path=config)


@app.command("benchmark")
def benchmark_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained checkpoint."),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(1, "--batch-size"),
    n_runs: int = typer.Option(100, "--n-runs"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Benchmark inference latency and throughput."""
    from .analysis.export import benchmark_inference

    benchmark_inference(
        checkpoint,
        n_runs=n_runs,
        batch_size=batch_size,
        device=device,
        config_path=config,
    )


# ---------------------------------------------------------------------------
# Noise robustness sweep
# ---------------------------------------------------------------------------
@app.command("noise-sweep")
def noise_sweep_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained checkpoint."),
    out: str = typer.Option("results/figs/noise_robustness.png", "--out"),
    snr_min: float = typer.Option(5.0, "--snr-min", help="Min SNR in dB."),
    snr_max: float = typer.Option(40.0, "--snr-max", help="Max SNR in dB."),
    n_steps: int = typer.Option(8, "--n-steps", help="Number of SNR levels."),
    n_samples: int = typer.Option(100, "--n-samples"),
    device: str = typer.Option("auto", "--device"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Evaluate model robustness across SNR levels."""
    from .analysis.noise_sweep import noise_robustness_sweep

    noise_robustness_sweep(
        checkpoint,
        out_path=out,
        snr_range=(snr_min, snr_max),
        n_snr_steps=n_steps,
        n_samples=n_samples,
        device_str=device,
        config_path=config,
    )


# ---------------------------------------------------------------------------
# Baselines comparison
# ---------------------------------------------------------------------------
@app.command("baselines")
def baselines_cmd(
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", help="DL model checkpoint (optional)."
    ),
    n_samples: int = typer.Option(200, "--n-samples"),
    device: str = typer.Option("cpu", "--device"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Evaluate classical unwrapping baselines (+ DL if checkpoint given)."""
    from .analysis.baselines import evaluate_baselines, evaluate_dl_baseline

    print("Classical baselines:")
    evaluate_baselines(n_samples=n_samples)

    if checkpoint:
        print("\nDL model:")
        evaluate_dl_baseline(
            checkpoint, n_samples=n_samples, device_str=device, config_path=config
        )


# ---------------------------------------------------------------------------
# TTA evaluation
# ---------------------------------------------------------------------------
@app.command("tta")
def tta_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained checkpoint."),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    n_augments: int = typer.Option(
        8, "--n-augments", help="Number of TTA views (max 8)."
    ),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Evaluate with test-time augmentation (TTA)."""
    from .analysis.tta import evaluate_tta

    evaluate_tta(checkpoint, data_dir, n_augments=n_augments, config_path=config)


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------
@app.command("latex-table")
def latex_table_cmd(
    run_dir: str = typer.Option(
        ..., "--run-dir", help="Run directory with metrics.csv."
    ),
    out: str = typer.Option("results/tables/metrics.tex", "--out"),
    epoch: int = typer.Option(-1, "--epoch", help="Which epoch (-1 = last)."),
) -> None:
    """Export metrics to a LaTeX booktabs table."""
    from .analysis.export_latex import metrics_to_latex

    table = metrics_to_latex(run_dir, out_path=out, epoch=epoch)
    print(table)


# ---------------------------------------------------------------------------
# Master Cloud Payload Rendering Sequence
# ---------------------------------------------------------------------------
@app.command("plot-all-local")
def plot_all_local_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", help="Extracted payload directory."),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Data dir override."
    ),
    n_samples: int = typer.Option(4, "--n-samples", help="Samples for grids."),
):
    """Central CLI sequence to render downloaded Kaggle/Colab payload locally on CPU."""
    from .core.config import load_train_config
    import os

    print("=" * 60)
    print(f"Starting plot-all-local on payload: {run_dir}")
    print("=" * 60)

    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        raise typer.Exit(1)

    cfg_path = run_dir / "config.yaml"
    cfg = load_train_config(str(cfg_path) if cfg_path.exists() else None)

    dataset_dir = str(data_dir) if data_dir else cfg.data.data_dir
    final_figs_dir = run_dir / "final_figures"
    final_figs_dir.mkdir(parents=True, exist_ok=True)

    print("Forcing strictly CPU extraction logic...")

    from .visualize.training_curve import plot_training_curve
    from .visualize.qualitative_grid import plot_qualitative_grid
    from .visualize.phase_profile import plot_phase_profile
    from .visualize.error_histogram import plot_error_histogram
    from .visualize.loss_landscape import plot_loss_landscape
    from .visualize.convergence import plot_convergence
    from .analysis.gradcam import plot_gradcam
    from .visualize.baseline_comparison_grid import plot_baseline_comparison
    from .visualize.noise_degradation_grid import plot_noise_degradation

    cfg_arg = str(cfg_path) if cfg_path.exists() else None
    ckpt_arg = str(ckpt_path)
    dir_arg = str(run_dir)

    try:
        print("\n[1/8] Generating Training Curves...")
        plot_training_curve(
            dir_arg, out_path=str(final_figs_dir / "training_curve.png")
        )
        plot_convergence(dir_arg, out_path=str(final_figs_dir / "convergence.png"))
    except Exception as e:
        print(f"Skipped metrics plots: {e}")

    print(f"\n[2/8] Generating Qualitative Grid (n={n_samples})...")
    plot_qualitative_grid(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "qualitative_grid.png"),
        n_samples=n_samples,
        config_path=cfg_arg,
    )

    print("\n[3/8] Generating Phase Profile...")
    plot_phase_profile(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "phase_profile.png"),
        sample_idx=0,
        config_path=cfg_arg,
    )

    print("\n[4/8] Generating Error Histogram...")
    plot_error_histogram(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "error_histogram.png"),
        config_path=cfg_arg,
    )

    print("\n[5/8] Generating Loss Landscape...")
    plot_loss_landscape(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "loss_landscape.png"),
        num_eval_samples=50,
        grid_size=11,
        config_path=cfg_arg,
    )

    print("\n[6/8] Generating GradCAM (enc5)...")
    plot_gradcam(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "gradcam.png"),
        n_samples=n_samples,
        target_layer_name="enc5",
        config_path=cfg_arg,
    )

    print("\n[7/8] Generating Baseline Superiority Matrix...")
    plot_baseline_comparison(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "baseline_comparison.png"),
        n_samples=n_samples,
        config_path=cfg_arg,
    )

    print("\n[8/8] Generating Noise Degradation Grid...")
    plot_noise_degradation(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "noise_degradation.png"),
        sample_idx=0,
        config_path=cfg_arg,
    )

    print("\n" + "=" * 60)
    print(
        f"Done. All Nature-tier figures dynamically compiled and saved safely to CPU directory: {final_figs_dir}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
