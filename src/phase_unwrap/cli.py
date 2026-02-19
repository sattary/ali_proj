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

from .config import TrainConfig, load_train_config
from .train import train as run_train

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
    from .generate import generate_to_h5

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
    from .multiseed import run_multiseed

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
