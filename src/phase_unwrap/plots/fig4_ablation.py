"""
Figure 4: PCLCN Fast Core Matrix Ablation Studies.

Stateless implementations for:
1. plot_f4_multiseed: Boxen plots of multi-seed / method comparisons.
2. plot_f4_radar: Radar chart for multi-metric ablations.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import (
    DOUBLE_COL,
    annotate_significance,
    compute_statistical_test,
    create_nature_palette,
    label_panels,
    publication_plot,
)


@publication_plot
def plot_f4_multiseed(
    df: pd.DataFrame,
    metrics: list[str],
    filepath: str | None = None,
) -> plt.Figure:
    """
    Boxen plot for multiseed metrics.
    
    Args:
        df: pd.DataFrame containing columns 'method', 'metric', 'value'
    """
    palette = create_nature_palette(len(df['method'].unique()))
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        plot_df = df[df['metric'] == metric]

        if plot_df.empty:
            continue

        sns.boxenplot(
            data=plot_df, x="method", y="value", ax=ax, palette=palette,
            hue="method", legend=False,
        )
        sns.stripplot(
            data=plot_df, x="method", y="value", ax=ax, color="black", alpha=0.3, size=3
        )

        methods = plot_df['method'].unique()
        if len(methods) >= 2:
            group1 = plot_df[plot_df['method'] == methods[0]]['value'].values
            group2 = plot_df[plot_df['method'] == methods[1]]['value'].values

            if len(group1) > 0 and len(group2) > 0:
                _, pval = compute_statistical_test(group1, group2, test="mannwhitney")
                y_max = max(np.max(group1), np.max(group2))
                annotate_significance(ax, 0, 1, y_max * 1.05, pval)

        metric_name = metric.replace("val_", "").upper()
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_xlabel("")
        ax.set_title(f"{metric_name} Distribution", fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Multi-Seed Ablation Comparison", fontsize=11, y=1.02)
    label_panels(axes)
    fig.tight_layout()
    return fig


@publication_plot
def plot_f4_radar(
    results: dict[str, dict[str, float]],
    baseline_name: str = "baseline",
    filepath: str | None = None,
) -> plt.Figure:
    """
    Radar chart for ablation study visualization.
    """
    metrics = list(list(results.values())[0].keys())
    n_metrics = len(metrics)

    palette = create_nature_palette(len(results))

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.6),
        subplot_kw=dict(projection="polar"),
    )

    angles = [n / float(n_metrics) * 2 * math.pi for n in range(n_metrics)]
    angles += angles[:1]

    for idx, (name, values) in enumerate(results.items()):
        vals = [values.get(m, 0) for m in metrics]

        baseline_vals = [results[baseline_name].get(m, 1) for m in metrics]
        normalized = []
        for v, bv in zip(vals, baseline_vals):
            if bv != 0:
                normalized.append(min(v / bv, 2.0))
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
    ax.set_title("PCLCN Ablation Matrix\n(relative to baseline)", fontsize=10, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.tight_layout()
    return fig
