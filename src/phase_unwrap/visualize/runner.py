"""
Sequential visualization orchestrator for cloud payload extraction.

Rationale:
    ARCHITECTURE: The `cli.py` handler must be a lightweight parser interface.
    The heavy lifting of instantiating 8 different plotting modules, enforcing
    CPU extraction contexts, and managing cross-module exception states is a
    purely Domain logic concern. This file encapsulates the `plot-all-local`
    rendering sequence to uphold single-responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..core.config import load_train_config
from .baseline_comparison_grid import plot_baseline_comparison
from .convergence import plot_convergence
from .error_histogram import plot_error_histogram
from .loss_landscape import plot_loss_landscape
from .noise_degradation_grid import plot_noise_degradation
from .phase_profile import plot_phase_profile
from .qualitative_grid import plot_qualitative_grid
from .training_curve import plot_training_curve


def run_plot_all_local(
    run_dir: Path,
    data_dir: Optional[Path],
    n_samples: int,
) -> None:
    """Central sequence to render downloaded Kaggle/Colab payload locally on CPU."""
    from ..analysis.gradcam import plot_gradcam

    print("=" * 60)
    print(f"Starting plot-all-local on payload: {run_dir}")
    print("=" * 60)

    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        raise typer.Exit(1)

    cfg_path = run_dir / "config.yaml"
    cfg = load_train_config(str(cfg_path) if cfg_path.exists() else None)

    dataset_dir = str(data_dir) if data_dir else cfg.data.data_dir
    final_figs_dir = run_dir / "final_figures"
    final_figs_dir.mkdir(parents=True, exist_ok=True)

    print("Forcing strictly CPU extraction logic...")

    cfg_arg = str(cfg_path) if cfg_path.exists() else None
    ckpt_arg = str(ckpt_path)
    dir_arg = str(run_dir)

    try:
        print("\n[1/8] Generating Training Curves...")
        plot_training_curve(
            dir_arg, out_path=str(final_figs_dir / "training_curve.png")
        )
        plot_convergence(dir_arg, out_path=str(final_figs_dir / "convergence.png"))
    except Exception as e:
        print(f"Skipped metrics plots: {e}")

    print(f"\n[2/8] Generating Qualitative Grid (n={n_samples})...")
    plot_qualitative_grid(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "qualitative_grid.png"),
        n_samples=n_samples,
        config_path=cfg_arg,
    )

    print("\n[3/8] Generating Phase Profile...")
    plot_phase_profile(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "phase_profile.png"),
        sample_idx=0,
        config_path=cfg_arg,
    )

    print("\n[4/8] Generating Error Histogram...")
    plot_error_histogram(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "error_histogram.png"),
        config_path=cfg_arg,
    )

    print("\n[5/8] Generating Loss Landscape...")
    plot_loss_landscape(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "loss_landscape.png"),
        num_eval_samples=50,
        grid_size=11,
        config_path=cfg_arg,
    )

    print("\n[6/8] Generating GradCAM (enc5)...")
    plot_gradcam(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "gradcam.png"),
        n_samples=n_samples,
        target_layer_name="enc5",
        config_path=cfg_arg,
    )

    print("\n[7/8] Generating Baseline Superiority Matrix...")
    plot_baseline_comparison(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "baseline_comparison.png"),
        n_samples=n_samples,
        config_path=cfg_arg,
    )

    print("\n[8/8] Generating Noise Degradation Grid...")
    plot_noise_degradation(
        ckpt_arg,
        data_dir=dataset_dir,
        out_path=str(final_figs_dir / "noise_degradation.png"),
        sample_idx=0,
        config_path=cfg_arg,
    )

    print("\n" + "=" * 60)
    print(
        f"Done. All Nature-tier figures dynamically compiled and saved safely to CPU directory: {final_figs_dir}"
    )
