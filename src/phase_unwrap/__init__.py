"""
Phase unwrapping: UNetRes2-based absolute phase reconstruction.

Sub-package layout:
    core/      - config, ops, losses, utils
    data/      - HDF5 dataset, synthetic generator
    model/     - UNetRes2 architecture, EMA
    training/  - training loop, multi-seed, HPO, ablation
    analysis/  - baselines, noise sweep, GradCAM, TTA, export, LaTeX tables
    plots/ - publication figures
"""

import os

# Fix matplotlib backend for Kaggle/Colab environments
# Must be set before any matplotlib imports
if os.environ.get("MPLBACKEND") == "module://matplotlib_inline.backend_inline":
    os.environ["MPLBACKEND"] = "Agg"

from .core.config import TrainConfig, load_train_config
from .training.train import train

__version__ = "0.1.0"
__all__ = ["TrainConfig", "load_train_config", "train", "__version__"]