"""
Phase Unwrapping CLI - Hierarchical Command Structure.
"""

from __future__ import annotations

import typer

from .cli_cmds.data import DataApp
from .cli_cmds.train import TrainApp
from .cli_cmds.eval import EvalApp
from .cli_cmds.export import ExportApp
from .cli_cmds.plot import PlotApp
from .cli_cmds.infer import InferApp
from .cli_cmds.config import ConfigApp

app = typer.Typer(
    name="phase-unwrap",
    help="Phase unwrapping / absolute phase reconstruction CLI.",
    add_completion=False,
)

app.add_typer(DataApp)
app.add_typer(TrainApp)
app.add_typer(EvalApp, name="eval")
app.add_typer(ExportApp, name="export")
app.add_typer(PlotApp, name="plot")
app.add_typer(ConfigApp, name="config")
app.add_typer(InferApp)

def main() -> None:
    # Python 3.12+ explicitly deprecates fork() in multithreaded (PyTorch) environments.
    # Force 'spawn' to prevent hard deadlocks on process boundary.
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    app()

if __name__ == "__main__":
    main()
