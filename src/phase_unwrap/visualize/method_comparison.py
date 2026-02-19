"""
Method comparison radar chart and grouped bar plots.

Compares DL model against classical baselines (Itoh, Least-Squares)
across multiple metrics with statistical significance.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .style import (
    DOUBLE_COL,
    NATURE_PALETTE,
    annotate_significance,
    compute_statistical_test,
    create_nature_palette,
    nature_style,
    save_figure,
)


def plot_method_comparison(
    results: dict[str, dict[str, float]],
    out_path: str = "results/figs/method_comparison",
    metrics: list[str] | None = None,
) -> None:
    """
    Create radar chart comparing methods across metrics.

    Args:
        results: Dict of {method_name: {metric: value}}
                e.g., {"DL": {"MAE": 0.05, "RMSE": 0.08, ...},
                       "Itoh": {"MAE": 0.12, ...}}
        out_path: Output path (without extension)
        metrics: List of metrics to compare (default: MAE, RMSE, SSIM, PSNR)
    """
    if metrics is None:
        metrics = ["MAE", "RMSE", "SSIM", "PSNR"]

    methods = list(results.keys())
    n_metrics = len(metrics)

    with nature_style():
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))

        # Left: Radar chart
        ax1 = fig.add_subplot(121, projection="polar")

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        palette = create_nature_palette(len(methods))

        for idx, method in enumerate(methods):
            values = []
            for metric in metrics:
                val = results[method].get(metric, 0)
                # Normalize values for radar (0-1 scale)
                all_vals = [results[m].get(metric, 0) for m in methods]
                min_val, max_val = min(all_vals), max(all_vals)
                if max_val > min_val:
                    norm_val = 1 - (val - min_val) / (
                        max_val - min_val
                    )  # Invert so lower is better
                else:
                    norm_val = 0.5
                values.append(norm_val)

            values += values[:1]  # Complete the circle

            ax1.plot(
                angles, values, "o-", linewidth=2, label=method, color=palette[idx]
            )
            ax1.fill(angles, values, alpha=0.15, color=palette[idx])

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics, fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_title(
            "Performance Radar\n(Normalized, higher is better)", fontsize=10, pad=20
        )
        ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)

        # Right: Grouped bar chart with significance
        ax2 = fig.add_subplot(122)

        x = np.arange(len(metrics))
        width = 0.25

        for idx, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            offset = width * (idx - len(methods) / 2 + 0.5)
            bars = ax2.bar(
                x + offset,
                values,
                width,
                label=method,
                color=palette[idx],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        ax2.set_xlabel("Metric", fontsize=9)
        ax2.set_ylabel("Value", fontsize=9)
        ax2.set_title("Metric Comparison", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved method comparison: {out_path}")


def plot_method_comparison_bar(
    results: dict[str, dict[str, list[float]]],
    out_path: str = "results/figs/method_comparison_bar",
    metrics: list[str] | None = None,
) -> None:
    """
    Grouped bar chart with error bars and significance testing.

    Args:
        results: Dict of {method_name: {metric: [values]}}
                Each metric has a list of values (e.g., from multiple samples)
    """
    if metrics is None:
        metrics = ["MAE", "RMSE"]

    methods = list(results.keys())
    n_metrics = len(metrics)

    with nature_style():
        fig, axes = plt.subplots(1, n_metrics, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        if n_metrics == 1:
            axes = [axes]

        palette = create_nature_palette(len(methods))

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            means = []
            stds = []
            data_groups = []

            for method in methods:
                values = results[method].get(metric, [0])
                means.append(np.mean(values))
                stds.append(np.std(values))
                data_groups.append(values)

            x = np.arange(len(methods))
            bars = ax.bar(
                x,
                means,
                yerr=stds,
                capsize=3,
                color=palette[: len(methods)],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add swarm plot for individual points
            for idx, (method, values) in enumerate(zip(methods, data_groups)):
                y_jittered = np.array(values) + np.random.normal(0, 0.001, len(values))
                ax.scatter(
                    [idx] * len(values),
                    y_jittered,
                    color="black",
                    alpha=0.3,
                    s=5,
                    zorder=3,
                )

            # Statistical annotations
            if len(methods) >= 2:
                _, pval = compute_statistical_test(
                    np.array(data_groups[0]),
                    np.array(data_groups[1]),
                    test="mannwhitney",
                )
                annotate_significance(ax, 0, 1, max(means) * 1.1, pval)

            ax.set_ylabel(metric, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=8, rotation=15)
            ax.set_title(f"{metric} Comparison", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved method comparison bar: {out_path}")
