"""
Multi-seed training runner.
"""

from __future__ import annotations

import csv
import os
from copy import deepcopy
from typing import List

import numpy as np

from ..core.config import TrainConfig
from ..core.utils import ensure_dir
from .train import CSV_COLUMNS, train


def _read_metrics_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _aggregate(run_dirs: List[str], out_path: str) -> None:
    """Read metrics.csv from each run, compute mean+/-std per epoch."""
    all_data: list[list[dict[str, str]]] = []
    for rd in run_dirs:
        csv_path = os.path.join(rd, "metrics.csv")
        if os.path.exists(csv_path):
            all_data.append(_read_metrics_csv(csv_path))

    if not all_data:
        print("No metrics.csv files found to aggregate.")
        return

    numeric_cols = [c for c in CSV_COLUMNS if c != "epoch"]
    min_epochs = min(len(d) for d in all_data)

    agg_header = ["epoch"]
    for c in numeric_cols:
        agg_header.extend([f"{c}_mean", f"{c}_std"])

    with open(out_path, "w", newline="") as f:
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
    multi_gpu: bool = False,
    use_amp: bool = False,
) -> None:
    """
    Rationale: Sequentially executes identical training configurations across diverse 
    initialization seeds to rigorously measure statistical variance. Hardware flags 
    (multi_gpu, use_amp) are explicitly cascaded to bypass redundant auto-detection 
    overhead in the inner loops, ensuring deterministic compute allocation.
    """
    base_dir = os.path.join(cfg.logging.runs_root, base_run_name)
    ensure_dir(base_dir)

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

        train(seed_cfg, multi_gpu=multi_gpu)
        run_dirs.append(seed_cfg.logging.run_dir)

    agg_path = os.path.join(base_dir, "aggregate.csv")
    _aggregate(run_dirs, agg_path)

    print(f"\nAll {len(seeds)} seed runs complete.")
    print(f"Aggregate: {agg_path}")