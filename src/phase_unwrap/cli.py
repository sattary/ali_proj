"""
Phase Unwrapping CLI - Hierarchical Command Structure.

Usage:
    phase-unwrap data generate
    phase-unwrap train train
    phase-unwrap train tune
    phase-unwrap train ablation
    phase-unwrap eval baselines
    phase-unwrap export onnx
    phase-unwrap plot training-curve
    phase-unwrap run local
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .core.config import TrainConfig, load_train_config

app = typer.Typer(
    name="phase-unwrap",
    help="Phase unwrapping / absolute phase reconstruction CLI.",
    add_completion=False,
)

DataApp = typer.Typer(help="Data generation and management commands.")
TrainApp = typer.Typer(help="Training commands.")
EvalApp = typer.Typer(help="Evaluation commands.")
ExportApp = typer.Typer(help="Model export commands.")
PlotApp = typer.Typer(help="Visualization commands.")
RunApp = typer.Typer(help="Runtime and rendering commands.")

app.add_typer(DataApp, name="data")
app.add_typer(TrainApp, name="train")
app.add_typer(EvalApp, name="eval")
app.add_typer(ExportApp, name="export")
app.add_typer(PlotApp, name="plot")
app.add_typer(RunApp, name="run")


# ============================================================================
# DATA COMMANDS
# ============================================================================


@DataApp.command("generate")
def data_generate(
    num_samples: int = typer.Option(
        180_000, "--num-samples", help="Total number of samples to generate."
    ),
    shard_size: int = typer.Option(
        1000, "--shard-size", help="Number of samples per HDF5 shard."
    ),
    out_dir: str = typer.Option(
        "data/full", "--out-dir", help="Output directory for HDF5 shards."
    ),
    seed: int = typer.Option(1337, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Generate synthetic interferogram data to HDF5 shards."""
    from .data.generate import generate_to_h5

    generate_to_h5(
        out_dir=out_dir,
        num_samples=num_samples,
        shard_size=shard_size,
        seed=seed,
    )


# ============================================================================
# TRAINING COMMANDS
# ============================================================================


@TrainApp.command("train")
def train_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML or JSON config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: 'auto', 'cuda', or 'cpu'."
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Path to checkpoint to resume from."
    ),
    run_name: Optional[str] = typer.Option(
        None, "--run-name", help="Name for this training run."
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Override number of training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    multi_gpu: bool = typer.Option(
        False, "--multi-gpu", help="Use all available GPUs with DataParallel."
    ),
    gpu_ids: Optional[str] = typer.Option(
        None, "--gpu-ids", help="Comma-separated GPU IDs (e.g., '0,1')."
    ),
    auto_push_interval: Optional[int] = typer.Option(
        None, "--auto-push-interval", help="Enable auto-push every N epochs."
    ),
    auto_push_dry_run: bool = typer.Option(
        False, "--auto-push-dry-run", help="Test auto-push without actually pushing."
    ),
    force_auto_push: bool = typer.Option(
        False, "--force-auto-push", help="Force auto-push even outside Kaggle/Colab."
    ),
    auto_push_pat: Optional[str] = typer.Option(
        None, "--auto-push-pat", help="GitHub PAT (or set GITHUB_PAT env var)."
    ),
    use_amp: bool = typer.Option(
        False, "--use-amp", help="Enable Automatic Mixed Precision (AMP)."
    ),
) -> None:
    """Train the UNetRes2 absolute phase reconstruction model."""
    import torch
    from .training.train import train as run_train
    from .training.multi_gpu import detect_kaggle_multi_gpu

    cfg: TrainConfig = load_train_config(config)

    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device
    if run_name is not None:
        cfg.logging.run_name = run_name
    if epochs is not None:
        cfg.optim.epochs = epochs
    if batch_size is not None:
        cfg.optim.batch_size = batch_size
    if use_amp:
        cfg.model.use_amp = True

    gpu_id_list: Optional[list[int]] = None
    if gpu_ids is not None:
        gpu_id_list = [int(x.strip()) for x in gpu_ids.split(",")]

    if not multi_gpu and torch.cuda.is_available():
        if detect_kaggle_multi_gpu():
            multi_gpu = True
            typer.echo("[kaggle] Auto-enabling multi-GPU mode")

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


@TrainApp.command("tune")
def tune_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML or JSON config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: 'auto', 'cuda', or 'cpu'."
    ),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials."),
    tune_epochs: int = typer.Option(15, "--tune-epochs", help="Epochs per trial."),
    study_name: str = typer.Option(
        "phase_unwrap_hpo", "--study-name", help="Optuna study name."
    ),
    n_workers: int = typer.Option(
        1, "--n-workers", "-j", help="Number of parallel workers."
    ),
    gpu_ids: Optional[str] = typer.Option(
        None, "--gpu-ids", help="Comma-separated GPU IDs (e.g., '0,1')."
    ),
    auto_push: bool = typer.Option(
        False, "--auto-push", help="Push results to GitHub after HPO completes."
    ),
    auto_push_pat: Optional[str] = typer.Option(
        None, "--auto-push-pat", help="GitHub PAT (or set GITHUB_PAT env var)."
    ),
    auto_push_dry_run: bool = typer.Option(
        False, "--auto-push-dry-run", help="Test auto-push without actually pushing."
    ),
    use_amp: bool = typer.Option(
        False, "--use-amp", help="Enable Automatic Mixed Precision (AMP)."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size to prevent OOM."
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", help="Override dataloader workers (CPU config)."
    ),
) -> None:
    """Run Optuna hyperparameter search (TPE + MedianPruner)."""
    import torch
    from .training.tune import run_tuning
    from .git_automation.environment import is_kaggle, is_colab
    from .git_automation.tune_push import create_tune_auto_push_callback

    cfg: TrainConfig = load_train_config(config)

    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device
    if use_amp:
        cfg.model.use_amp = True
    if workers is not None:
        cfg.data.workers = workers

    gpu_id_list: Optional[list[int]] = None
    if gpu_ids is not None:
        gpu_id_list = [int(x.strip()) for x in gpu_ids.split(",")]
    elif n_workers > 1 and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            gpu_id_list = list(range(min(n_workers, n_gpus)))
            typer.echo(f"Auto-detected {n_gpus} GPUs, using: {gpu_id_list}")

    auto_push_callback = None
    if auto_push:
        if not auto_push_dry_run and not is_kaggle() and not is_colab():
            typer.echo(
                "[auto-push] Warning: Not on Kaggle/Colab. Use --auto-push-dry-run to test."
            )

        data_dir_str = str(data_dir) if data_dir else cfg.data.data_dir
        auto_push_callback = create_tune_auto_push_callback(
            optuna_dir=cfg.logging.run_dir,
            data_dir=data_dir_str,
            push_interval=0,
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
        batch_size_override=batch_size,
    )


@TrainApp.command("multiseed")
def multiseed_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML or JSON config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: 'auto', 'cuda', or 'cpu'."
    ),
    run_name: str = typer.Option(
        "multiseed", "--run-name", help="Base name for the run group."
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Override training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    seeds: Optional[str] = typer.Option(
        None, "--seeds", help="Comma-separated seeds (e.g., '1337,42,7')."
    ),
    num_seeds: int = typer.Option(3, "--num-seeds", help="Auto-generate N seeds."),
) -> None:
    """Run N training runs with different seeds, aggregate results."""
    from .training.multiseed import run_multiseed
    import random as _rnd

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
        seed_list = [_rnd.randint(0, 2**31) for _ in range(num_seeds)]

    run_multiseed(cfg, base_run_name=run_name, seeds=seed_list)


@TrainApp.command("ablation")
def ablation_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML or JSON config file."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: 'auto', 'cuda', or 'cpu'."
    ),
    run_name: str = typer.Option(
        "ablation", "--run-name", help="Base name for ablation run group."
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Override training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    seeds: Optional[str] = typer.Option(
        None, "--seeds", help="Comma-separated seeds (e.g., '1337,42,7')."
    ),
    num_seeds: int = typer.Option(
        3, "--num-seeds", help="Number of seeds per ablation."
    ),
    out_table: str = typer.Option(
        "results/tables/ablation.tex", "--out-table", help="Output LaTeX table path."
    ),
) -> None:
    """Run ablation study and produce LaTeX comparison table."""
    from .training.ablation import run_ablation
    import random as _rnd

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
        seed_list = [_rnd.randint(0, 2**31) for _ in range(num_seeds)]

    ablations = {
        "no_grad_loss": {"loss.w_grad": 0.0},
        "no_curv_loss": {"loss.w_curv": 0.0},
        "no_coordconv": {"model.use_coordconv": False},
        "no_ema": {"model.ema_decay": 0.0},
    }

    run_ablation(
        base_cfg=cfg,
        ablations=ablations,
        base_name=run_name,
        seeds=seed_list,
        out_table=out_table,
    )


# ============================================================================
# EVAL COMMANDS
# ============================================================================


@EvalApp.command("baselines")
def eval_baselines_cmd(
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", help="Optional DL model checkpoint."
    ),
    n_samples: int = typer.Option(
        200, "--n-samples", help="Number of samples to evaluate."
    ),
    device: str = typer.Option("cpu", "--device", help="Device for inference."),
    config: Optional[str] = typer.Option(
        None, "--config", help="Config file for DL model."
    ),
) -> None:
    """Evaluate classical unwrapping baselines (optionally with DL model)."""
    from .analysis.baselines import evaluate_baselines, evaluate_dl_baseline

    typer.echo("Classical baselines:")
    evaluate_baselines(n_samples=n_samples)

    if checkpoint:
        typer.echo("\nDL model:")
        evaluate_dl_baseline(
            checkpoint, n_samples=n_samples, device_str=device, config_path=config
        )


@EvalApp.command("tta")
def eval_tta_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    n_augments: int = typer.Option(
        8, "--n-augments", help="Number of TTA views (max 8)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Evaluate with test-time augmentation (TTA)."""
    from .analysis.tta import evaluate_tta

    evaluate_tta(checkpoint, data_dir, n_augments=n_augments, config_path=config)


@EvalApp.command("noise")
def eval_noise_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    out: str = typer.Option(
        "results/figs/noise_robustness.png", "--out", help="Output figure path."
    ),
    snr_min: float = typer.Option(5.0, "--snr-min", help="Minimum SNR in dB."),
    snr_max: float = typer.Option(40.0, "--snr-max", help="Maximum SNR in dB."),
    n_steps: int = typer.Option(8, "--n-steps", help="Number of SNR levels."),
    n_samples: int = typer.Option(
        100, "--n-samples", help="Number of evaluation samples."
    ),
    device: str = typer.Option("auto", "--device", help="Device for inference."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
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


@EvalApp.command("benchmark")
def eval_benchmark_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    device: str = typer.Option("cpu", "--device", help="Device for benchmarking."),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size for inference."),
    n_runs: int = typer.Option(100, "--n-runs", help="Number of inference runs."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
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


# ============================================================================
# EXPORT COMMANDS
# ============================================================================


@ExportApp.command("onnx")
def export_onnx_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    out: str = typer.Option(
        "results/model.onnx", "--out", help="Output ONNX file path."
    ),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Export model to ONNX format."""
    from .analysis.export import export_onnx

    export_onnx(checkpoint, out_path=out, opset=opset, config_path=config)


@ExportApp.command("torchscript")
def export_torchscript_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    out: str = typer.Option(
        "results/model.pt", "--out", help="Output TorchScript file path."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Export model to TorchScript (traced) format."""
    from .analysis.export import export_torchscript

    export_torchscript(checkpoint, out_path=out, config_path=config)


@ExportApp.command("table")
def export_table_cmd(
    run_dir: str = typer.Option(
        ..., "--run-dir", help="Run directory with metrics.csv."
    ),
    out: str = typer.Option(
        "results/tables/metrics.tex", "--out", help="Output LaTeX table path."
    ),
    epoch: int = typer.Option(-1, "--epoch", help="Which epoch (-1 = last)."),
) -> None:
    """Export training metrics to a LaTeX booktabs table."""
    from .analysis.export_latex import metrics_to_latex

    table = metrics_to_latex(run_dir, out_path=out, epoch=epoch)
    typer.echo(table)


# ============================================================================
# PLOT COMMANDS
# ============================================================================


@PlotApp.command("training-curve")
def plot_training_curve_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
    no_lr: bool = typer.Option(False, "--no-lr", help="Omit learning rate subplot."),
) -> None:
    """Plot dual-axis training loss + validation MAE (supports multi-seed)."""
    from .visualize import plot_training_curve

    plot_training_curve(run_dir, out_path=out, show_lr=not no_lr)


@PlotApp.command("convergence")
def plot_convergence_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
) -> None:
    """Plot loss components and learning rate schedule."""
    from .visualize import plot_convergence

    plot_convergence(run_dir, out_path=out)


@PlotApp.command("qualitative")
def plot_qualitative_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/qualitative_grid.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(4, "--n-samples", help="Number of rows in the grid."),
    show_noise: bool = typer.Option(
        True, "--show-noise/--no-show-noise", help="Show noisy input column."
    ),
    noise_level: float = typer.Option(
        1.0, "--noise-level", help="Noise level (0.0=clean, 1.0=max)."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot Clean | Noisy | GT | Wrapped | Pred | Error grid."""
    from .visualize import plot_qualitative_grid

    plot_qualitative_grid(
        checkpoint,
        data_dir,
        out_path=out,
        n_samples=n_samples,
        config_path=config,
        show_noise=show_noise,
        noise_level=noise_level,
        subset=subset,
    )


@PlotApp.command("phase-profile")
def plot_phase_profile_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/phase_profile.png", "--out", help="Output file path."
    ),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Sample index to plot."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot 1D cross-section through center row/column."""
    from .visualize import plot_phase_profile

    plot_phase_profile(
        checkpoint,
        data_dir,
        out_path=out,
        sample_idx=sample_idx,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("error-hist")
def plot_error_hist_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/error_histogram.png", "--out", help="Output file path."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot per-sample MAE histogram + CDF with percentiles."""
    from .visualize import plot_error_histogram

    plot_error_histogram(
        checkpoint, data_dir, out_path=out, config_path=config, subset=subset
    )


@PlotApp.command("loss-landscape")
def plot_loss_landscape_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/loss_landscape.png", "--out", help="Output file path."
    ),
    grid_size: int = typer.Option(31, "--grid-size", help="Resolution of 2D grid."),
    alpha_range: float = typer.Option(1.0, "--alpha-range", help="Perturbation range."),
    num_eval_samples: int = typer.Option(
        500, "--num-eval-samples", help="Samples per loss evaluation."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot 2D loss surface contour (Li et al., 2018 filter-normalized)."""
    from .visualize import plot_loss_landscape

    plot_loss_landscape(
        checkpoint,
        data_dir,
        out_path=out,
        grid_size=grid_size,
        alpha_range=alpha_range,
        num_eval_samples=num_eval_samples,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("gradcam")
def plot_gradcam_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/gradcam.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    layer: str = typer.Option("enc5", "--layer", help="Target encoder layer."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot GradCAM attention overlay on interferograms."""
    from .analysis.gradcam import plot_gradcam

    plot_gradcam(
        checkpoint,
        data_dir,
        out_path=out,
        n_samples=n_samples,
        target_layer_name=layer,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("baseline-comparison")
def plot_baseline_comparison_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/baseline_comparison.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot baseline superiority matrix."""
    from .visualize.baseline_comparison_grid import plot_baseline_comparison

    plot_baseline_comparison(
        checkpoint,
        data_dir,
        out_path=out,
        n_samples=n_samples,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("noise-comparison")
def plot_noise_comparison_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/noise_comparison.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    noise_level: float = typer.Option(
        1.0, "--noise-level", help="Noise level (0.0=clean, 1.0=max)."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot clean vs noisy inference comparison grid."""
    from .visualize import plot_noise_comparison_grid

    plot_noise_comparison_grid(
        checkpoint,
        data_dir,
        out_path=out,
        n_samples=n_samples,
        noise_level=noise_level,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("noise-degradation")
def plot_noise_degradation_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/noise_degradation.png", "--out", help="Output file path."
    ),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Target dataset index."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot iterative noise degradation evaluation."""
    from .visualize.noise_degradation_grid import plot_noise_degradation

    plot_noise_degradation(
        checkpoint,
        data_dir,
        out_path=out,
        sample_idx=sample_idx,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("curriculum-noise")
def plot_curriculum_noise_cmd(
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/curriculum_noise_grid.png", "--out", help="Output file path."
    ),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Target dataset index."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot dynamic curriculum noise progression over epochs."""
    from .visualize.curriculum_noise_grid import plot_curriculum_noise

    plot_curriculum_noise(
        data_dir, out_path=out, sample_idx=sample_idx, config_path=config
    )


@PlotApp.command("method-comparison")
def plot_method_comparison_cmd(
    results: str = typer.Option(
        ...,
        "--results",
        help="Comma-separated method:metric_value pairs (e.g., 'DL:0.05,Itoh:0.12').",
    ),
    out: str = typer.Option(
        "results/figs/method_comparison.png", "--out", help="Output file path."
    ),
    metrics: str = typer.Option(
        "MAE,RMSE,SSIM,PSNR", "--metrics", help="Comma-separated metrics."
    ),
) -> None:
    """Plot method comparison with statistical significance."""
    from .visualize import plot_method_comparison

    metric_list = [m.strip() for m in metrics.split(",")]
    results_dict: dict[str, dict[str, float]] = {}
    for pair in results.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            method, value = parts
            results_dict[method] = {metric_list[0]: float(value)}
    plot_method_comparison(results_dict, out_path=out, metrics=metric_list)


@PlotApp.command("multiseed-comparison")
def plot_multiseed_comparison_cmd(
    run_dirs: str = typer.Option(
        ...,
        "--run-dirs",
        help="Comma-separated 'label:path' pairs (e.g., 'run1:./runs/exp1,run2:./runs/exp2').",
    ),
    out: str = typer.Option(
        "results/figs/multiseed_comparison.png", "--out", help="Output file path."
    ),
) -> None:
    """Plot multi-seed comparison with confidence intervals."""
    from .visualize import plot_multiseed_comparison

    run_dir_dict: dict[str, str] = {}
    for pair in run_dirs.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            label, path = parts
            run_dir_dict[label] = path
    plot_multiseed_comparison(run_dir_dict, out_path=out)


@PlotApp.command("prediction-scatter")
def plot_prediction_scatter_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/prediction_scatter.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Maximum samples to scatter plot."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
    noise_level: Optional[float] = typer.Option(
        None,
        "--noise-level",
        help="Override curriculum noise level (0.0=clean, 1.0=max phase-7 destruction).",
    ),
) -> None:
    """Plot GT vs prediction scatter with regression line."""
    from .visualize import plot_prediction_scatter

    plot_prediction_scatter(
        checkpoint,
        data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
        noise_level=noise_level,
    )


@PlotApp.command("residual-analysis")
def plot_residual_analysis_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/residual_analysis.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Maximum samples to analyze."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
    noise_level: Optional[float] = typer.Option(
        None, "--noise-level", help="Override curriculum noise level."
    ),
) -> None:
    """Plot residual analysis with spatial heatmaps."""
    from .visualize import plot_residual_analysis

    plot_residual_analysis(
        checkpoint,
        data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
        noise_level=noise_level,
    )


@PlotApp.command("tta-benefit")
def plot_tta_benefit_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Dataset directory."),
    out: str = typer.Option(
        "results/figs/tta_benefit.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        100, "--max-samples", help="Maximum samples to evaluate."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot TTA benefit analysis."""
    from .visualize import plot_tta_benefit

    plot_tta_benefit(
        checkpoint,
        data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
    )


# ============================================================================
# RUN COMMANDS
# ============================================================================


@RunApp.command("local")
def run_local_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Extracted payload directory."),
    data_dir: Optional[str] = typer.Option(
        None, "--data-dir", help="Data directory override."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples for grids."
    ),
) -> None:
    """Render downloaded Kaggle/Colab payload locally on CPU."""
    from .visualize.runner import run_plot_all_local

    run_plot_all_local(
        run_dir=Path(run_dir),
        data_dir=Path(data_dir) if data_dir else None,
        n_samples=n_samples,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    app()


if __name__ == "__main__":
    main()
