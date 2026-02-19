"""
Visualize sub-package: Nature-level publication figures.

Provides self-contained plot functions and a shared style module.
Each plot module handles its own data loading and model inference
where needed, requiring only paths to run directories or checkpoints.
"""

from .convergence import plot_convergence
from .error_histogram import plot_error_histogram
from .loss_landscape import plot_loss_landscape
from .phase_profile import plot_phase_profile
from .qualitative_grid import plot_qualitative_grid
from .training_curve import plot_training_curve

# Backward-compat re-export for train.py epoch visuals
from ._epoch_visuals import save_epoch_visuals

__all__ = [
    "plot_convergence",
    "plot_error_histogram",
    "plot_loss_landscape",
    "plot_phase_profile",
    "plot_qualitative_grid",
    "plot_training_curve",
    "save_epoch_visuals",
]
