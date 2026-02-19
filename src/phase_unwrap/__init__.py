"""
Phase unwrapping and absolute phase reconstruction package.

This package exposes a modular implementation of the UNetRes2-based
absolute phase reconstruction pipeline.
"""

from .config import TrainConfig, load_train_config
from .train import train

__all__ = ["TrainConfig", "load_train_config", "train"]
