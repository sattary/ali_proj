"""
Visualize sub-package: Nature-level publication figures with seaborn.

Provides self-contained plot functions and a shared style module.
Each plot module handles its own data loading and model inference
where needed, requiring only paths to run directories or checkpoints.

New seaborn-enhanced plots:
- Statistical distributions (violin, KDE)
- Regression with confidence intervals
- Method comparisons with significance testing
- Multi-seed aggregation
"""

from .convergence import plot_convergence
from .error_histogram import plot_error_histogram
from .loss_landscape import plot_loss_landscape
from .method_comparison import plot_method_comparison, plot_method_comparison_bar
from .multiseed_comparison import plot_ablation_radar, plot_multiseed_comparison
from .phase_profile import plot_phase_profile
from .prediction_scatter import plot_prediction_scatter
from .qualitative_grid import plot_qualitative_grid
from .residual_analysis import plot_residual_analysis
from .training_curve import plot_training_curve
from .tta_benefit import plot_tta_benefit

# Backward-compat re-export for train.py epoch visuals
from ._epoch_visuals import save_epoch_visuals

__all__ = [
    # Enhanced existing plots
    "plot_convergence",
    "plot_error_histogram",
    "plot_loss_landscape",
    "plot_phase_profile",
    "plot_qualitative_grid",
    "plot_training_curve",
    # New seaborn plots
    "plot_method_comparison",
    "plot_method_comparison_bar",
    "plot_multiseed_comparison",
    "plot_ablation_radar",
    "plot_prediction_scatter",
    "plot_residual_analysis",
    "plot_tta_benefit",
    # Internal
    "save_epoch_visuals",
]