"""
Phase Unwrapping CLI - Flat Command Structure (Ponytail).
"""

from __future__ import annotations

import typer

from .cli_cmds.data import DataApp
from .cli_cmds.eval import EvalApp
from .cli_cmds.export import ExportApp
from .cli_cmds.infer import InferApp
from .cli_cmds.plot import PlotApp
from .cli_cmds.train import TrainApp

app = typer.Typer(
    name="phun",
    help="Physics-Constrained Latent Corrector Network (PCLCN) CLI.",
    add_completion=False,
)

# Flat command registration (zero nested sub-app boilerplate)
app.add_typer(DataApp)
app.add_typer(TrainApp)
app.add_typer(EvalApp)
app.add_typer(ExportApp)
app.add_typer(PlotApp)
app.add_typer(InferApp)


def main() -> None:
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    app()


if __name__ == "__main__":
    main()
