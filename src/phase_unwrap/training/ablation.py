"""
Modernized PCLCN ablation study runner.
Saves raw JSON/CSV metrics for offline visualization and analysis.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence

from ..core.config import TrainConfig
from .multiseed import run_multiseed


def _apply_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    """
    Apply flat key=value overrides to a config using dot notation.

    Example: 'loss.w_curl' -> cfg.loss.w_curl = value
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
    base_name: str = "pclcn_ablation",
    seeds: Sequence[int] = (42, 1337, 7),
    use_amp: bool = False,
) -> Path:
    """
    Automates multi-seed ablation runs across PCLCN components.
    Exports raw numerical JSON summary for offline post-processing.
    """
    out_dir = Path(base_cfg.logging.runs_root) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}

    for label, overrides in ablations.items():
        print(f"\n{'=' * 60}")
        print(f"PCLCN Ablation: {label}")
        print(f"  Overrides: {overrides}")
        print(f"{'=' * 60}\n")

        cfg = deepcopy(base_cfg)
        _apply_overrides(cfg, overrides)

        group_name = f"{base_name}/{label}"
        run_multiseed(cfg, base_run_name=group_name, seeds=list(seeds), use_amp=use_amp)

        summary[label] = {
            "overrides": overrides,
            "run_dir": str(Path(cfg.logging.runs_root) / group_name),
            "aggregate_csv": str(Path(cfg.logging.runs_root) / group_name / "aggregate.csv"),
        }

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPCLCN ablation complete. Raw summary saved to: {summary_path}")
    return summary_path