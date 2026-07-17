import typer
from typing import Optional

ExportApp = typer.Typer(help="Model export commands.")

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


