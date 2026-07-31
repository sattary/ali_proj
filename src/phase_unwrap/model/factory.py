"""
Model factory for constructing PCLCN networks from config.
"""

from __future__ import annotations

from ..core.config import ModelConfig
from .pclcn import PCLCNModel


def build_model(cfg: ModelConfig) -> PCLCNModel:
    """Construct PCLCNModel from configuration."""
    return PCLCNModel(
        height=128,
        width=128,
        base=cfg.base,
        zero_reference_prior=(
            getattr(cfg, "reference_mode", "reference_free") == "reference_free"
        )
        or getattr(cfg, "zero_reference_prior", False),
        unmasked_zernike=getattr(cfg, "unmasked_zernike", False),
    )
