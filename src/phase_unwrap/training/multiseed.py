"""
Multi-seed training runner.
Executes PCLCN training across multiple initialization seeds and aggregates raw CSV metrics.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np

from ..core.config import TrainConfig
from .train import CSV_COLUMNS, train


def _aggregate(run_dirs: List[str], out_path: Path) -> None:
    """Read metrics.csv from each run, compute mean+/-std per epoch."""
    all_data: list[list[dict[str, str]]] = []
    for rd in run_dirs:
        csv_path = Path(rd) / "metrics.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                all_data.append(list(csv.DictReader(f)))

    if not all_data:
        print("No metrics.csv files found to aggregate.")
        return

    numeric_cols = [c for c in CSV_COLUMNS if c != "epoch"]
    min_epochs = min(len(d) for d in all_data)

    agg_header = ["epoch"]
    for c in numeric_cols:
        agg_header.extend([f"{c}_mean", f"{c}_std"])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(agg_header)

        for i in range(min_epochs):
            epoch = all_data[0][i].get("epoch", str(i + 1))
            row = [epoch]
            for c in numeric_cols:
                vals = []
                for seed_data in all_data:
                    v = seed_data[i].get(c, "")
                    if v != "":
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                if vals:
                    row.append(f"{np.mean(vals):.6f}")
                    row.append(f"{np.std(vals):.6f}")
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    print(f"Aggregated {len(all_data)} runs x {min_epochs} epochs -> {out_path}")


def run_multiseed(
    cfg: TrainConfig,
    base_run_name: str,
    seeds: List[int],
    use_amp: bool = False,
) -> None:
    """
    Executes identical training configurations across multiple initialization seeds
    to measure statistical variance and output aggregated raw metrics.
    """
    base_dir = Path(cfg.logging.runs_root) / base_run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[str] = []

    for i, seed in enumerate(seeds):
        seed_cfg = deepcopy(cfg)
        seed_cfg.logging.seed = seed
        seed_cfg.logging.run_name = f"{base_run_name}/seed_{seed}"

        if use_amp:
            seed_cfg.model.use_amp = True

        print(f"\n{'=' * 60}")
        print(f"Multi-seed run {i + 1}/{len(seeds)} | seed={seed}")
        print(f"{'=' * 60}\n")

        train(seed_cfg)
        run_dirs.append(seed_cfg.logging.run_dir)

    agg_path = base_dir / "aggregate.csv"
    _aggregate(run_dirs, agg_path)

    print(f"\nAll {len(seeds)} seed runs complete.")
    print(f"Aggregate: {agg_path}")