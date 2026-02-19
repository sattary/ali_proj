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
) -> None:
    """Train the UNetRes2 absolute phase reconstruction model."""
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

    run_train(cfg, resume_path=str(resume) if resume else None)


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
) -> None:
    """Run Optuna hyperparameter search (TPE + MedianPruner)."""
    from .training.tune import run_tuning

    cfg: TrainConfig = load_train_config(config)
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device

    run_tuning(cfg, n_trials=n_trials, tune_epochs=tune_epochs, study_name=study_name)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
