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
        None,
        "--config",
        "-c",
        help="Path to a JSON or YAML configuration file.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Override the data directory specified in the config.",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Override the output directory for checkpoints and curves.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Override device selection: 'auto', 'cuda', or 'cpu'.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override number of training epochs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override training batch size.",
    ),
) -> None:
    """
    Train the UNetRes2 absolute phase reconstruction model.
    """
    cfg: TrainConfig = load_train_config(config)

    # Apply common CLI overrides.
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if out_dir is not None:
        cfg.logging.out_dir = str(out_dir)
    if device is not None:
        cfg.model.device = device
    if epochs is not None:
        cfg.optim.epochs = epochs
    if batch_size is not None:
        cfg.optim.batch_size = batch_size

    run_train(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

