"""Publication paper figures CLI — renders figures from saved run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

PlotApp = typer.Typer(help="Publication paper figures generator.")


@PlotApp.command("plot")
def plot_cmd(
    run_dir: Annotated[str, typer.Option("--run-dir", help="Saved run dir (config.yaml + best.pth + result.json).")],
    fig: Annotated[str, typer.Option("--fig", help="Figure number (1,2,3,4,5,6, or 'all').")] = "all",
    out_dir: Annotated[str, typer.Option("--out-dir", help="Output directory for figures.")] = "results/paper_figures",
    n_samples: Annotated[int, typer.Option("--n-samples", help="Samples for qualitative figures.")] = 4,
    device: Annotated[str, typer.Option("--device", help="cpu | cuda | auto.")] = "auto",
    config: Annotated[str | None, typer.Option("--config", help="Override config path.")] = None,
) -> None:
    """Render publication figures from saved run artifacts (Kaggle-trained)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from ..cli_cmds._plot_runners import RUNNERS

    figs = ["1", "2", "3", "4", "5", "6"] if fig.lower().strip() == "all" else [fig.lower().strip()]
    for f in figs:
        runner = RUNNERS.get(f)
        if runner is None:
            print(f"Unknown fig {f}, skipping")
            continue
        print(f"Rendering fig{f}...")
        runner(run_dir=run_dir, out_dir=str(out_path), n_samples=n_samples, device=device, config=config)
    print(f"\nDone. Figures in {out_path}")
