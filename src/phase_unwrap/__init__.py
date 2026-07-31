"""
Phase unwrapping: UNetRes2-based absolute phase reconstruction.

Sub-package layout:
    core/      - config, ops, losses, utils
    data/      - HDF5 dataset, synthetic generator
    model/     - UNetRes2 architecture, EMA
    training/  - training loop, multi-seed, ablation
    analysis/  - baselines, noise sweep, GradCAM, TTA, export, LaTeX tables
    plots/ - publication figures
"""

import os

import matplotlib
matplotlib.use("Agg")

from .core.config import TrainConfig, load_train_config
from .training.train import train

__version__ = "1.0.0"
__all__ = ["TrainConfig", "load_train_config", "train", "__version__"]
