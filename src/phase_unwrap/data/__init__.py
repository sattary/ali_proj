"""
Data sub-package: HDF5 dataset and synthetic data generator.
"""

from .dataset import (
    H5ShardDataset,
    build_dataloaders,
    discover_h5_shards,
    smart_split,
)
from .generate import generate_sample, generate_to_h5

__all__ = [
    "H5ShardDataset",
    "build_dataloaders",
    "discover_h5_shards",
    "smart_split",
    "generate_sample",
    "generate_to_h5",
]