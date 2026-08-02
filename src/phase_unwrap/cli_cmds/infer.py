import typer
from typing import Optional

InferApp = typer.Typer(help="Inference commands.")

# ============================================================================
# INFER COMMANDS
# ============================================================================


@InferApp.command("infer")
def run_infer_cmd(
    input_path: Optional[str] = typer.Option(
        None, "--input", help="Single-frame input (.npy, .png, etc)."
    ),
    mode: str = typer.Option("intensity", "--mode", help="intensity, two_frame, or otf."),
    input_i1: Optional[str] = typer.Option(None, "--input-i1", help="Calibrated I1 .npy frame."),
    input_i2: Optional[str] = typer.Option(None, "--input-i2", help="Calibrated I2 .npy frame."),
    num_samples: int = typer.Option(10, "--num-samples", min=1, help="OTF samples to generate."),
    seed: int = typer.Option(2026, "--seed", help="OTF simulator seed."),
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to trained model .pth."
    ),

    out: str = typer.Option("results/out", "--out", help="Output directory/prefix."),
    device: str = typer.Option("auto", "--device", help="Device to run on (cuda, cpu, auto)."),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml."),
) -> None:
    """Run legacy single-frame, calibrated two-frame, or OTF inference."""
    from phase_unwrap.analysis.inference import (
        run_inference,
        run_otf_two_frame_inference,
        run_two_frame_inference,
    )
    from phase_unwrap.plots.inference_output import save_inference_outputs

    if mode == "otf":
        outputs = run_otf_two_frame_inference(
            checkpoint, num_samples, seed, device, config
        )
        save_inference_outputs(outputs, out)
        return
    if mode == "two_frame":
        if not input_i1 or not input_i2:
            raise typer.BadParameter("two_frame requires --input-i1 and --input-i2")
        outputs = run_two_frame_inference(
            input_i1, input_i2, checkpoint, device, config
        )
        save_inference_outputs(outputs, out)
        return
    if mode != "intensity" or not input_path:
        raise typer.BadParameter("intensity mode requires --input")
    from phase_unwrap.analysis.inference import run_inference

    phi_pred, _, _ = run_inference(input_path, checkpoint, device, config)
    save_inference_outputs({"prediction": phi_pred[None]}, out)

