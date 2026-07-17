import typer
from typing import Optional

PlotApp = typer.Typer(help="Visualization commands.")

# ============================================================================
# PLOT COMMANDS
# ============================================================================


@PlotApp.command("training-curve")
def plot_training_curve_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
    no_lr: bool = typer.Option(False, "--no-lr", help="Omit learning rate subplot."),
) -> None:
    """Plot dual-axis training loss + validation MAE (supports multi-seed)."""
    from .visualize import plot_training_curve

    plot_training_curve(run_dir, out_path=out, show_lr=not no_lr)


@PlotApp.command("convergence")
def plot_convergence_cmd(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to run directory."),
    out: Optional[str] = typer.Option(None, "--out", help="Output file path."),
) -> None:
    """Plot loss components and learning rate schedule."""
    from .visualize import plot_convergence

    plot_convergence(run_dir, out_path=out)


@PlotApp.command("qualitative")
def plot_qualitative_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/qualitative_grid.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(4, "--n-samples", help="Number of rows in the grid."),
    show_noise: bool = typer.Option(
        True, "--show-noise/--no-show-noise", help="Show noisy input column."
    ),
    noise_level: float = typer.Option(
        1.0, "--noise-level", help="Noise level (0.0=clean, 1.0=max)."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot Clean | Noisy | GT | Wrapped | Pred | Error grid."""
    from .visualize import plot_qualitative_grid

    plot_qualitative_grid(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        n_samples=n_samples,
        config_path=config,
        show_noise=show_noise,
        noise_level=noise_level,
        subset=subset,
    )


@PlotApp.command("phase-profile")
def plot_phase_profile_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/phase_profile.png", "--out", help="Output file path."
    ),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Sample index to plot."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot 1D cross-section through center row/column."""
    from .visualize import plot_phase_profile

    plot_phase_profile(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        sample_idx=sample_idx,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("error-hist")
def plot_error_hist_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/error_histogram.png", "--out", help="Output file path."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot per-sample MAE histogram + CDF with percentiles."""
    from .visualize import plot_error_histogram

    plot_error_histogram(
        checkpoint, data_dir, out_path=out, config_path=config, subset=subset
    )


@PlotApp.command("loss-landscape")
def plot_loss_landscape_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/loss_landscape.png", "--out", help="Output file path."
    ),
    grid_size: int = typer.Option(31, "--grid-size", help="Resolution of 2D grid."),
    alpha_range: float = typer.Option(1.0, "--alpha-range", help="Perturbation range."),
    num_eval_samples: int = typer.Option(
        500, "--num-eval-samples", help="Samples per loss evaluation."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot 2D loss surface contour (Li et al., 2018 filter-normalized)."""
    from .visualize import plot_loss_landscape

    plot_loss_landscape(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        grid_size=grid_size,
        alpha_range=alpha_range,
        num_eval_samples=num_eval_samples,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("gradcam")
def plot_gradcam_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/gradcam.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    layer: str = typer.Option("enc5", "--layer", help="Target encoder layer."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot GradCAM attention overlay on interferograms."""
    from .analysis.gradcam import plot_gradcam

    plot_gradcam(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        n_samples=n_samples,
        target_layer_name=layer,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("baseline-comparison")
def plot_baseline_comparison_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/baseline_comparison.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot baseline superiority matrix."""
    from .visualize.baseline_comparison_grid import plot_baseline_comparison

    plot_baseline_comparison(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        n_samples=n_samples,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("noise-comparison")
def plot_noise_comparison_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/noise_comparison.png", "--out", help="Output file path."
    ),
    n_samples: int = typer.Option(
        4, "--n-samples", help="Number of samples to visualize."
    ),
    noise_level: float = typer.Option(
        1.0, "--noise-level", help="Noise level (0.0=clean, 1.0=max)."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot clean vs noisy inference comparison grid."""
    from .visualize import plot_noise_comparison_grid

    plot_noise_comparison_grid(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        n_samples=n_samples,
        noise_level=noise_level,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("noise-degradation")
def plot_noise_degradation_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/noise_degradation.png", "--out", help="Output file path."
    ),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Target dataset index."),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot iterative noise degradation evaluation."""
    from .visualize.noise_degradation_grid import plot_noise_degradation

    plot_noise_degradation(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        sample_idx=sample_idx,
        config_path=config,
        subset=subset,
    )


@PlotApp.command("curriculum-noise")
def plot_curriculum_noise_cmd(
    out: str = typer.Option(
        "results/figs/curriculum_noise_grid.png", "--out", help="Output file path."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    sample_idx: int = typer.Option(0, "--sample-idx", help="Target dataset index."),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
) -> None:
    """Plot dynamic curriculum noise progression over epochs."""
    from .visualize.curriculum_noise_grid import plot_curriculum_noise

    plot_curriculum_noise(
        data_dir, out_path=out, sample_idx=sample_idx, config_path=config
    )


@PlotApp.command("method-comparison")
def plot_method_comparison_cmd(
    results: str = typer.Option(
        ...,
        "--results",
        help="Comma-separated method:metric_value pairs (e.g., 'DL:0.05,Itoh:0.12').",
    ),
    out: str = typer.Option(
        "results/figs/method_comparison.png", "--out", help="Output file path."
    ),
    metrics: str = typer.Option(
        "MAE,RMSE,SSIM,PSNR", "--metrics", help="Comma-separated metrics."
    ),
) -> None:
    """Plot method comparison with statistical significance."""
    from .visualize import plot_method_comparison

    metric_list = [m.strip() for m in metrics.split(",")]
    results_dict: dict[str, dict[str, float]] = {}
    for pair in results.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            method, value = parts
            results_dict[method] = {metric_list[0]: float(value)}
    plot_method_comparison(results_dict, out_path=out, metrics=metric_list)


@PlotApp.command("multiseed-comparison")
def plot_multiseed_comparison_cmd(
    run_dirs: str = typer.Option(
        ...,
        "--run-dirs",
        help="Comma-separated 'label:path' pairs (e.g., 'run1:./runs/exp1,run2:./runs/exp2').",
    ),
    out: str = typer.Option(
        "results/figs/multiseed_comparison.png", "--out", help="Output file path."
    ),
) -> None:
    """Plot multi-seed comparison with confidence intervals."""
    from .visualize import plot_multiseed_comparison

    run_dir_dict: dict[str, str] = {}
    for pair in run_dirs.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            label, path = parts
            run_dir_dict[label] = path
    plot_multiseed_comparison(run_dir_dict, out_path=out)


@PlotApp.command("prediction-scatter")
def plot_prediction_scatter_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/prediction_scatter.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Maximum samples to scatter plot."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
    noise_level: Optional[float] = typer.Option(
        None,
        "--noise-level",
        help="Override curriculum noise level (0.0=clean, 1.0=max phase-7 destruction).",
    ),
    all_data: bool = typer.Option(
        False, "--all-data", help="Use entire dataset (disable val/test splits)."
    ),
) -> None:
    """Plot GT vs prediction scatter with regression line."""
    from .visualize import plot_prediction_scatter

    plot_prediction_scatter(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
        noise_level=noise_level,
        all_data=all_data,
    )


@PlotApp.command("residual-analysis")
def plot_residual_analysis_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/residual_analysis.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Maximum samples to analyze."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
    noise_level: Optional[float] = typer.Option(
        None, "--noise-level", help="Override curriculum noise level."
    ),
    all_data: bool = typer.Option(
        False, "--all-data", help="Use entire dataset (disable val/test splits)."
    ),
) -> None:
    """Plot residual analysis with spatial heatmaps."""
    from .visualize import plot_residual_analysis

    plot_residual_analysis(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
        noise_level=noise_level,
        all_data=all_data,
    )


@PlotApp.command("tta-benefit")
def plot_tta_benefit_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint."
    ),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override dataset directory."),
    out: str = typer.Option(
        "results/figs/tta_benefit.png", "--out", help="Output file path."
    ),
    max_samples: int = typer.Option(
        100, "--max-samples", help="Maximum samples to evaluate."
    ),
    subset: str = typer.Option(
        "val", "--subset", help="Dataset subset (train, val, test)."
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Config file."),
    all_data: bool = typer.Option(
        False, "--all-data", help="Use entire dataset (disable val/test splits)."
    ),
) -> None:
    """Plot TTA benefit analysis."""
    from .visualize import plot_tta_benefit

    plot_tta_benefit(
        checkpoint,
        data_dir=data_dir,
        out_path=out,
        max_samples=max_samples,
        config_path=config,
        subset=subset,
        all_data=all_data,
    )


