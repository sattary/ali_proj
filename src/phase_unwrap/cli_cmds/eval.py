"""
Evaluation commands CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

EvalApp = typer.Typer(help="Evaluation commands.")


@EvalApp.command("eval")
def eval_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to model checkpoint."),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    subset: str = typer.Option("test", "--subset", help="Dataset split (train, val, test)."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Evaluate model checkpoint on dataset split."""
    from phase_unwrap.core.inference import load_inference_state
    from phase_unwrap.data import build_dataloaders
    from phase_unwrap.training.train import run_eval

    model, cfg, _, device = load_inference_state(checkpoint, data_dir=data_dir, config_path=config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
    loader = test_loader if subset == "test" else (val_loader if subset == "val" else train_loader)
    metrics = run_eval(model, loader, device, use_amp=cfg.model.use_amp)
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v:.6f}")


@EvalApp.command("baselines")
def eval_baselines_cmd(
    data_dir: str = typer.Option(..., "--data-dir", help="Path to HDF5 dataset."),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Optional DL model checkpoint."),
    n_samples: int = typer.Option(200, "--n-samples", help="Number of samples to evaluate."),
    device: str = typer.Option("cpu", "--device", help="Device for inference."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file for DL model."),
) -> None:
    """Evaluate classical unwrapping baselines (optionally with DL model)."""
    from phase_unwrap.analysis.baselines import evaluate_baselines, evaluate_dl_baseline

    typer.echo("Classical baselines:")
    evaluate_baselines(data_dir=data_dir, n_samples=n_samples)

    if checkpoint:
        typer.echo("\nDL model:")
        evaluate_dl_baseline(
            checkpoint, data_dir=data_dir, n_samples=n_samples, device_str=device, config_path=config
        )


@EvalApp.command("noise")
def eval_noise_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained model checkpoint."),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option("results/figs/noise_robustness.png", "--out", help="Output figure path."),
    snr_min: float = typer.Option(5.0, "--snr-min", help="Minimum SNR in dB."),
    snr_max: float = typer.Option(40.0, "--snr-max", help="Maximum SNR in dB."),
    n_steps: int = typer.Option(8, "--n-steps", help="Number of SNR levels."),
    n_samples: int = typer.Option(100, "--n-samples", help="Number of evaluation samples."),
    device: str = typer.Option("auto", "--device", help="Device for inference."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Evaluate model robustness across SNR levels."""
    from phase_unwrap.analysis.noise_sweep import compute_noise_robustness
    from phase_unwrap.plots.fig3_noise_robustness import plot_f3_noise_sweep as plot_noise_sweep

    results, all_maes_by_snr = compute_noise_robustness(
        checkpoint,
        data_dir=data_dir,
        snr_range=(snr_min, snr_max),
        n_snr_steps=n_steps,
        n_samples=n_samples,
        device_str=device,
        config_path=config,
    )
    plot_noise_sweep(results, all_maes_by_snr, filepath=out)


@EvalApp.command("benchmark")
def eval_benchmark_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained model checkpoint."),
    device: str = typer.Option("cpu", "--device", help="Device for benchmarking."),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size for inference."),
    n_runs: int = typer.Option(100, "--n-runs", help="Number of inference runs."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Benchmark inference latency and throughput."""
    from phase_unwrap.analysis.export import benchmark_inference

    benchmark_inference(
        checkpoint,
        n_runs=n_runs,
        batch_size=batch_size,
        device=device,
        config_path=config,
    )
