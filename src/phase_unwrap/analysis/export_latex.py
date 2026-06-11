"""
LaTeX table exporter from metrics CSV.

Reads metrics.csv or aggregate.csv and outputs a publication-ready
LaTeX table with booktabs formatting and controlled significant figures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional, Sequence


def _fmt(val: float, decimals: int = 3) -> str:
    """Format a float or return '--' for NaN."""
    if math.isnan(val) or math.isinf(val):
        return "--"
    return f"{val:.{decimals}f}"


def _bold_best(values: list[str], lower_better: bool = True) -> list[str]:
    """Wrap the best value in \\textbf{}."""
    numerics = []
    for v in values:
        v_clean = v.replace("$", "").split("\\pm")[0].strip()
        try:
            numerics.append(float(v_clean))
        except ValueError:
            numerics.append(float("inf") if lower_better else float("-inf"))

    best_idx = (
        numerics.index(min(numerics)) if lower_better else numerics.index(max(numerics))
    )
    out = list(values)
    if out[best_idx] != "--":
        out[best_idx] = f"\\textbf{{{out[best_idx]}}}"
    return out


def metrics_to_latex(
    run_dir: str,
    out_path: Optional[str] = None,
    epoch: int = -1,
    metrics: Optional[Sequence[str]] = None,
    caption: str = "Training metrics summary.",
    label: str = "tab:metrics",
) -> str:
    """
    Convert metrics CSV to a LaTeX booktabs table.

    Args:
        run_dir:  Run directory containing metrics.csv or aggregate.csv.
        out_path: Write to file (None = return string only).
        epoch:    Which epoch row to export (-1 = last).
        metrics:  Which columns to include. Defaults to val metrics.
        caption:  LaTeX caption.
        label:    LaTeX label.

    Returns:
        LaTeX table source as a string.
    """
    run_path = Path(run_dir)
    agg_path = run_path / "aggregate.csv"
    metrics_path = run_path / "metrics.csv"
    csv_path = agg_path if agg_path.exists() else metrics_path

    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics CSV in {run_dir}")

    rows: list[dict[str, str]] = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    target = rows[epoch]  # -1 = last row

    if metrics is None:
        metrics = [
            "val_topo_mae",
            "val_abs_mae",
            "val_rmse",
            "val_ssim",
            "val_psnr",
            "val_max_err",
            "val_grad_mae",
        ]

    # header mapping for prettier column names
    header_map = {
        "val_topo_mae": "TopoMAE",
        "val_abs_mae": "AbsMAE",
        "val_rmse": "RMSE",
        "val_ssim": "SSIM",
        "val_psnr": "PSNR",
        "val_max_err": "MaxErr",
        "val_grad_mae": "GradMAE",
        "train_loss": "Train Loss",
        "train_mae": "Train MAE",
        "lr": "LR",
    }

    # decimal precision per metric
    precision_map = {
        "val_topo_mae": 4,
        "val_abs_mae": 4,
        "val_rmse": 4,
        "val_ssim": 4,
        "val_psnr": 2,
        "val_max_err": 3,
        "val_grad_mae": 4,
        "train_loss": 4,
        "train_mae": 4,
        "lr": 6,
    }

    headers = [header_map.get(m, m) for m in metrics]
    values = []
    for m in metrics:
        mean_key = f"{m}_mean"
        std_key = f"{m}_std"
        if mean_key in target and std_key in target:
            try:
                mean_val = float(target[mean_key])
                std_val = float(target[std_key])
                dec = precision_map.get(m, 3)
                values.append(f"${_fmt(mean_val, dec)} \\pm {_fmt(std_val, dec)}$")
            except (ValueError, TypeError):
                values.append("--")
        else:
            raw = target.get(m, "")
            try:
                values.append(_fmt(float(raw), precision_map.get(m, 3)))
            except (ValueError, TypeError):
                values.append("--")

    col_spec = "l" + "r" * len(metrics)

    epoch_val = target.get("epoch", "?")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    Epoch & " + " & ".join(headers) + r" \\",
        r"    \midrule",
        f"    {epoch_val} & " + " & ".join(values) + r" \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    table = "\n".join(lines) + "\n"

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(table)
        print(f"Saved LaTeX table: {out_path}")

    return table


def comparison_to_latex(
    run_dirs: dict[str, str],
    out_path: Optional[str] = None,
    epoch: int = -1,
    metrics: Optional[Sequence[str]] = None,
    caption: str = "Comparison of model configurations.",
    label: str = "tab:comparison",
    bold_best: bool = True,
) -> str:
    """
    Multi-row comparison table across multiple runs.

    Args:
        run_dirs: Mapping {row_label: run_dir_path}.
        out_path: Output .tex file.
        epoch:    Which epoch to compare (-1 = last).
        metrics:  Which metrics to include.
        caption:  LaTeX caption.
        label:    LaTeX label.
        bold_best: Bold the best value per column.

    Returns:
        LaTeX table string.
    """
    if metrics is None:
        metrics = ["val_topo_mae", "val_abs_mae", "val_rmse", "val_ssim", "val_psnr"]

    header_map = {
        "val_topo_mae": "TopoMAE",
        "val_abs_mae": "AbsMAE",
        "val_rmse": "RMSE",
        "val_ssim": "SSIM",
        "val_psnr": "PSNR",
        "val_max_err": "MaxErr",
        "val_grad_mae": "GradMAE",
    }

    # higher-better metrics
    higher_better = {"val_ssim", "val_psnr"}

    precision_map = {
        "val_topo_mae": 4,
        "val_abs_mae": 4,
        "val_rmse": 4,
        "val_ssim": 4,
        "val_psnr": 2,
        "val_max_err": 3,
        "val_grad_mae": 4,
    }

    headers = [header_map.get(m, m) for m in metrics]
    col_spec = "l" + "r" * len(metrics)

    # collect values per run
    all_values: dict[str, list[str]] = {}
    for name, rd in run_dirs.items():
        csv_path = Path(rd) / "metrics.csv"
        agg_path = Path(rd) / "aggregate.csv"
        path = agg_path if agg_path.exists() else csv_path
        rows: list[dict[str, str]] = []
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        target = rows[epoch]
        vals = []
        for m in metrics:
            mean_key = f"{m}_mean"
            std_key = f"{m}_std"
            if mean_key in target and std_key in target:
                try:
                    mean_val = float(target[mean_key])
                    std_val = float(target[std_key])
                    dec = precision_map.get(m, 3)
                    vals.append(f"${_fmt(mean_val, dec)} \\pm {_fmt(std_val, dec)}$")
                except (ValueError, TypeError):
                    vals.append("--")
            else:
                raw = target.get(m, "")
                try:
                    vals.append(_fmt(float(raw), precision_map.get(m, 3)))
                except (ValueError, TypeError):
                    vals.append("--")
        all_values[name] = vals

    # bold best per column
    if bold_best:
        for col_idx, m in enumerate(metrics):
            col_vals = [all_values[name][col_idx] for name in run_dirs]
            lower_better = m not in higher_better
            bolded = _bold_best(col_vals, lower_better=lower_better)
            for i, name in enumerate(run_dirs):
                all_values[name][col_idx] = bolded[i]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    Method & " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]

    for name in run_dirs:
        lines.append(f"    {name} & " + " & ".join(all_values[name]) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    table = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(table)
        print(f"Saved comparison table: {out_path}")
    return table