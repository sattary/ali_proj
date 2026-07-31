"""
Visualize sub-package: Publication-ready figures aligned with Optics Express.

Stateless plotting functions taking pure data arrays.
"""

from .fig1_architecture import plot_f1_architecture
from .fig2_baseline_comparison import plot_f2_baseline_comparison
from .fig3_noise_robustness import plot_f3_noise_grid, plot_f3_noise_sweep, plot_f3_dual_panel
from .fig4_ablation import plot_f4_multiseed, plot_f4_radar
from .fig5_diagnostics import plot_f5_diagnostics
from .fig6_physics_zernike import plot_f6_physics_zernike
from ._epoch_visuals import save_epoch_visuals

__all__ = [
    "plot_f1_architecture",
    "plot_f2_baseline_comparison",
    "plot_f3_noise_sweep",
    "plot_f3_noise_grid",
    "plot_f3_dual_panel",
    "plot_f4_multiseed",
    "plot_f4_radar",
    "plot_f5_diagnostics",
    "plot_f6_physics_zernike",
    "save_epoch_visuals",
]
