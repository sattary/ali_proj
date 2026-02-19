"""
Multi-seed comparison with seaborn boxen plots and swarm overlays.

For comparing model variants or ablation studies across multiple random seeds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .style import (
    DOUBLE_COL,
    annotate_significance,
    compute_statistical_test,
    create_nature_palette,
    nature_style,
    save_figure,
)


def plot_multiseed_comparison(
    run_dirs: dict[str, str],
    out_path: str = "results/figs/multiseed_comparison",
    metrics: list[str] | None = None,
    last_n_epochs: int = 10,
) -> None:
    """
    Compare multiple runs/ablations with multi-seed statistics.

    Args:
        run_dirs: Dict of {label: run_directory_path}
        out_path: Output path (without extension)
        metrics: Metrics to compare (default: val_mae, val_rmse, val_ssim)
        last_n_epochs: Use last N epochs for comparison
    """
    if metrics is None:
        metrics = ["val_mae", "val_rmse", "val_ssim"]

    # Collect data from each run
    all_data = {metric: {label: [] for label in run_dirs} for metric in metrics}

    for label, run_dir in run_dirs.items():
        agg_path = Path(run_dir) / "aggregate.csv"
        metrics_path = Path(run_dir) / "metrics.csv"
        csv_path = agg_path if agg_path.exists() else metrics_path

        if not csv_path.exists():
            print(f"Warning: No metrics found in {run_dir}")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Get last N epochs
        recent_rows = rows[-last_n_epochs:] if len(rows) > last_n_epochs else rows

        for metric in metrics:
            if agg_path.exists():
                # Multi-seed: use mean values
                values = [float(row.get(f"{metric}_mean", 0)) for row in recent_rows]
            else:
                # Single run
                values = [float(row.get(metric, 0)) for row in recent_rows]
            all_data[metric][label] = values

    with nature_style():
        palette = create_nature_palette(len(run_dirs))

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for seaborn
            plot_data = []
            labels = []
            for label in run_dirs:
                values = all_data[metric].get(label, [])
                if values:
                    plot_data.extend(values)
                    labels.extend([label] * len(values))

            if not plot_data:
                continue

            # Create DataFrame-like structure for seaborn
            import pandas as pd

            df = pd.DataFrame(
                {
                    "value": plot_data,
                    "method": labels,
                }
            )

            # Boxen plot (letter-value plot)
            sns.boxenplot(
                data=df, x="method", y="value", ax=ax, palette=palette[: len(run_dirs)]
            )

            # Overlay individual points
            sns.stripplot(
                data=df, x="method", y="value", ax=ax, color="black", alpha=0.3, size=3
            )

            # Statistical test between first two methods
            if len(run_dirs) >= 2:
                labels_list = list(run_dirs.keys())
                group1 = all_data[metric].get(labels_list[0], [])
                group2 = all_data[metric].get(labels_list[1], [])

                if group1 and group2:
                    _, pval = compute_statistical_test(
                        np.array(group1), np.array(group2), test="mannwhitney"
                    )
                    y_max = max(max(group1), max(group2))
                    annotate_significance(ax, 0, 1, y_max * 1.05, pval)

            # Format metric name
            metric_name = metric.replace("val_", "").upper()
            ax.set_ylabel(metric_name, fontsize=9)
            ax.set_xlabel("")
            ax.set_title(
                f"{metric_name} Distribution\n(last {last_n_epochs} epochs)", fontsize=9
            )
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Multi-Seed/Method Comparison", fontsize=11, y=1.02)
        plt.tight_layout()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved multi-seed comparison: {out_path}")


def plot_ablation_radar(
    results: dict[str, dict[str, float]],
    out_path: str = "results/figs/ablation_radar",
    baseline_name: str = "baseline",
) -> None:
    """
    Radar chart for ablation study visualization.

    Args:
        results: Dict of {ablation_name: {metric: value}}
        out_path: Output path
        baseline_name: Name of baseline configuration
    """
    import math

    metrics = list(list(results.values())[0].keys())
    n_metrics = len(metrics)

    with nature_style():
        palette = create_nature_palette(len(results))

        fig, ax = plt.subplots(
            figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.6),
            subplot_kw=dict(projection="polar"),
        )

        angles = [n / float(n_metrics) * 2 * math.pi for n in range(n_metrics)]
        angles += angles[:1]

        for idx, (name, values) in enumerate(results.items()):
            vals = [values.get(m, 0) for m in metrics]

            # Normalize to 0-1 scale (higher is better)
            baseline_vals = [results[baseline_name].get(m, 1) for m in metrics]
            normalized = []
            for v, bv in zip(vals, baseline_vals):
                if bv != 0:
                    normalized.append(min(v / bv, 2.0))  # Cap at 2x
                else:
                    normalized.append(0)

            normalized += normalized[:1]

            linewidth = 2.5 if name == baseline_name else 1.5
            alpha = 0.9 if name == baseline_name else 0.5

            ax.plot(
                angles,
                normalized,
                "o-",
                linewidth=linewidth,
                label=name,
                color=palette[idx],
                alpha=alpha,
            )
            ax.fill(angles, normalized, alpha=0.1 * alpha, color=palette[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 2.0)
        ax.set_title("Ablation Study\n(relative to baseline)", fontsize=10, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved ablation radar: {out_path}")
