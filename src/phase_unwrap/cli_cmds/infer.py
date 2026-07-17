import typer
from typing import Optional

InferApp = typer.Typer(help="Inference commands.")

# ============================================================================
# INFER COMMANDS
# ============================================================================


@InferApp.command("infer")
def run_infer_cmd(
    input_path: str = typer.Option(
        ..., "--input", help="Path to input interferogram (.npy, .png, etc)."
    ),
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to trained model .pth."
    ),

    out: str = typer.Option(
        "results/out", "--out", help="Output prefix (e.g., results/out)."
    ),
    device: str = typer.Option("auto", "--device", help="Device to run on (cuda, cpu, auto)."),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml."),
) -> None:
    """Run real-world inference on an arbitrary interferogram."""
    from .analysis.inference import run_inference

    run_inference(
        input_path=input_path,
        checkpoint_path=checkpoint,
        out_prefix=out,
        device_str=device,
        config_path=config,
    )



