"""
Evaluation commands CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

EvalApp = typer.Typer(help="Evaluation commands.")





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
