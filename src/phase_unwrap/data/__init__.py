"""
Data sub-package: HDF5 dataset and synthetic data generator.
"""

from .dataset import (
    H5ShardDataset,
    build_dataloaders,
    discover_h5_shards,
    smart_split,
    build_otf_loaders,
)
from .generate import generate_sample, generate_to_h5
from .simulator import MatlabSimulator, SimulationBatch
from .augmentation import normalize_intensity, NoiseAug

__all__ = [
    "H5ShardDataset",
    "build_dataloaders",
    "discover_h5_shards",
    "smart_split",
    "build_otf_loaders",
    "generate_sample",
    "generate_to_h5",
    "MatlabSimulator",
    "SimulationBatch",
    "normalize_intensity",
    "NoiseAug",
]
