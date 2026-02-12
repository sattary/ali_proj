from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import torch

from .config import TrainConfig, apply_overrides, load_train_config
from .data import MatPhaseDataset, _to_chw
from .model import build_model
from .ops import affine_align
from .losses import compute_metrics
from .utils import pick_device
from .train import train as run_train
from .visualize import save_inference_maps

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
    no_tensorboard: bool = typer.Option(
        False,
        "--no-tensorboard",
        help="Disable TensorBoard logging even if enabled in the config.",
    ),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Override config fields using dot-notation, e.g. --override loss.w_curv=0.005",
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
    if no_tensorboard:
        cfg.logging.use_tensorboard = False

    if override:
        cfg = apply_overrides(cfg, override)

    run_train(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


@app.command()
def infer(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-k", help="Path to a trained checkpoint (.pth)."),
    data_dir: Path = typer.Option(
        Path("data/imgs"),
        "--input",
        "-i",
        help="Directory containing .mat files for inference.",
    ),
    out_dir: Path = typer.Option(
        Path("inference"),
        "--output-dir",
        "-o",
        help="Directory to write inference visualizations.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional config file to match training hyperparameters.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Override device selection: 'auto', 'cuda', or 'cpu'.",
    ),
) -> None:
    """
    Run inference with a trained model on .mat files and save predictions.
    """
    cfg: TrainConfig = load_train_config(config)
    cfg.data.data_dir = str(data_dir)
    if device is not None:
        cfg.model.device = device

    dev = pick_device(cfg.model.device)

    # build dataset directly using MatPhaseDataset
    import glob as _glob
    import os as _os

    pattern = _os.path.join(str(data_dir), cfg.data.pattern)
    paths = sorted(_glob.glob(pattern))
    if not paths:
        raise typer.Exit(f"No files match {pattern}")

    ds = MatPhaseDataset(paths, I_key=cfg.data.I_key, phi_key=cfg.data.phi_key)

    model = build_model(cfg.model).to(dev)
    ckpt = torch.load(checkpoint, map_location=dev)
    state = ckpt.get("model_ema") or ckpt.get("model")
    if state is None:
        raise typer.Exit("Checkpoint does not contain 'model' or 'model_ema' state_dict.")
    model.load_state_dict(state)
    model.eval()

    out_dir_str = str(out_dir)

    all_mae = []
    all_rmse = []
    all_nrmse = []

    with torch.no_grad():
        for idx, (I_input, phi_gt, I_raw) in enumerate(ds):
            I_input = I_input.unsqueeze(0).to(dev)
            phi_gt = phi_gt.unsqueeze(0).to(dev)

            phi_raw, a_pred, b_pred_raw, conf_logit, k_off = model(I_input)
            phi_abs = phi_raw + k_off
            phi_abs_align, a_batch, c_batch = affine_align(phi_abs, phi_gt)

            m = compute_metrics(phi_abs_align, phi_gt)
            all_mae.append(float(m["MAE"]))
            all_rmse.append(float(m["RMSE"]))
            all_nrmse.append(float(m["NRMSE"]))

            save_inference_maps(
                I_input.cpu(),
                phi_abs_align.cpu(),
                phi_gt.cpu(),
                out_dir_str,
                tag=f"sample_{idx:04d}",
            )

    if all_mae:
        mean_mae = sum(all_mae) / len(all_mae)
        mean_rmse = sum(all_rmse) / len(all_rmse)
        mean_nrmse = sum(all_nrmse) / len(all_nrmse)
        typer.echo(
            f"Inference on {len(all_mae)} samples | "
            f"MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f}, NRMSE={mean_nrmse:.4f}"
        )

