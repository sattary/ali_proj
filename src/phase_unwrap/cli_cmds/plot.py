"""
Modernized paper figures CLI runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

PlotApp = typer.Typer(help="Publication paper figures generator.")


@PlotApp.command("plot")
def plot_cmd(
    fig: Annotated[str, typer.Option("--fig", help="Figure number to render (1, 2, 3, 4, 5, 6, or 'all').")] = "all",
    out_dir: Annotated[str, typer.Option("--out-dir", help="Output directory for paper figures.")] = "results/paper_figures",
) -> None:
    """Generate publication-ready figures for Optics Express."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig_str = fig.lower().strip()
    print(f"Rendering paper figures ({fig_str}) to: {out_path}")
