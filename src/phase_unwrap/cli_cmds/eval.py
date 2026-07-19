import typer
from typing import Optional

EvalApp = typer.Typer(help="Evaluation commands.")

# ============================================================================
# EVAL COMMANDS
# ============================================================================


@EvalApp.command("baselines")
def eval_baselines_cmd(
    data_dir: str = typer.Option(
        ..., "--data-dir", help="Path to HDF5 dataset."
    ),
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
    evaluate_baselines(data_dir=data_dir, n_samples=n_samples)

    if checkpoint:
        typer.echo("\nDL model:")
        evaluate_dl_baseline(
            checkpoint, data_dir=data_dir, n_samples=n_samples, device_str=device, config_path=config
        )


@EvalApp.command("tta")
def eval_tta_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Trained model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
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
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
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
        data_dir=data_dir,
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


