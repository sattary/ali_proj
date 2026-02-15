"""
Phase unwrapping and absolute phase reconstruction package.

This package exposes a modular implementation of the UNetRes2-based
absolute phase reconstruction pipeline originally developed in
`src/try.py`. The main programmatic entrypoint for training is
`phase_unwrap.train.train`, which operates on a structured
configuration object.
"""

from .config import TrainConfig, load_train_config
from .pipelines.trainer import train

__all__ = ["train"]
