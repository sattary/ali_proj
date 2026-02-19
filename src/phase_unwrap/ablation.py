"""
Automated ablation study runner.

Takes a mapping of ablation names to config overrides, runs each
configuration with multi-seed, and produces a comparison LaTeX table.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import TrainConfig, config_to_yaml
from .export_latex import comparison_to_latex
from .multiseed import run_multiseed
from .utils import ensure_dir


def _apply_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    """
    Apply flat key=value overrides to a config.

    Keys use dot notation: 'loss.w_grad', 'optim.lr', etc.
    """
    for key, val in overrides.items():
        parts = key.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return cfg


def run_ablation(
    base_cfg: TrainConfig,
    ablations: Dict[str, Dict[str, Any]],
    base_name: str = "ablation",
    seeds: Sequence[int] = (42, 1337, 7),
    metrics: Optional[Sequence[str]] = None,
    out_table: str = "results/tables/ablation.tex",
) -> str:
    """
    Run an ablation study.

    Args:
        base_cfg:   Base training configuration.
        ablations:  Mapping of {ablation_label: {config_key: value}}.
                    Example: {"no_grad": {"loss.w_grad": 0.0},
                              "high_grad": {"loss.w_grad": 1.0}}
        base_name:  Parent directory name under runs/.
        seeds:      List of seeds for multi-seed runs.
        metrics:    Which metrics to include in the table.
        out_table:  Output LaTeX file path.

    Returns:
        LaTeX table string.
    """
    ablation_dir = os.path.join(base_cfg.logging.runs_root, base_name)
    ensure_dir(ablation_dir)

    run_dirs: Dict[str, str] = {}

    for label, overrides in ablations.items():
        print(f"\n{'=' * 60}")
        print(f"Ablation: {label}")
        print(f"  Overrides: {overrides}")
        print(f"{'=' * 60}\n")

        cfg = deepcopy(base_cfg)
        _apply_overrides(cfg, overrides)

        group_name = f"{base_name}/{label}"

        run_multiseed(cfg, base_run_name=group_name, seeds=list(seeds))

        # use the aggregate from the multi-seed run
        run_dirs[label] = os.path.join(cfg.logging.runs_root, group_name)

    # generate comparison table
    table = comparison_to_latex(
        run_dirs=run_dirs,
        out_path=out_table,
        epoch=-1,
        metrics=metrics,
        caption=f"Ablation study results (mean over {len(seeds)} seeds).",
        label="tab:ablation",
        bold_best=True,
    )

    print(f"\nAblation complete. Table: {out_table}")
    return table
