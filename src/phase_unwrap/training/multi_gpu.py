"""
Multi-GPU training support using DataParallel.

Provides utilities for wrapping models in DataParallel and handling
checkpoint saving/loading across single and multi-GPU configurations.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def setup_multi_gpu(model: nn.Module, gpu_ids: Optional[list[int]] = None) -> nn.Module:
    """
    Wrap model in DataParallel if multiple GPUs are available.

    Args:
        model: The model to wrap
        gpu_ids: List of GPU IDs to use. If None, use all available.

    Returns:
        Model wrapped in DataParallel (if multi-GPU) or original model
    """
    if not torch.cuda.is_available():
        return model

    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        return model

    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))

    # Validate GPU IDs
    available_gpus = set(range(num_gpus))
    requested_gpus = set(gpu_ids)

    if not requested_gpus.issubset(available_gpus):
        invalid = requested_gpus - available_gpus
        raise ValueError(
            f"Invalid GPU IDs: {invalid}. Available GPUs: {list(available_gpus)}"
        )

    # Wrap in DataParallel
    model = nn.DataParallel(model, device_ids=gpu_ids)

    print(
        f"[multi-gpu] Using {len(gpu_ids)} GPUs: {gpu_ids}\n"
        f"[multi-gpu] Devices: {[torch.cuda.get_device_name(i) for i in gpu_ids]}"
    )

    return model


def get_model_state_dict(model: nn.Module) -> dict:
    """
    Get state dict from model, handling DataParallel wrapper.

    Args:
        model: Model (possibly wrapped in DataParallel)

    Returns:
        State dict (unwrapped if necessary)
    """
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    """
    Load state dict into model, handling DataParallel wrapper.

    Args:
        model: Model (possibly wrapped in DataParallel)
        state_dict: State dict to load
    """
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def detect_kaggle_multi_gpu() -> bool:
    """
    Detect if running on Kaggle with multiple GPUs.

    Returns:
        True if Kaggle environment with 2+ GPUs detected
    """
    is_kaggle = (
        os.path.exists("/kaggle")
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    )

    has_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2

    if is_kaggle and has_multi_gpu:
        print(
            f"[kaggle] Detected {torch.cuda.device_count()}x "
            f"{torch.cuda.get_device_name(0)}"
        )
        return True

    return False


def calculate_total_batch_size(batch_size_per_gpu: int, num_gpus: int) -> int:
    """
    Calculate total batch size based on per-GPU batch size.

    Args:
        batch_size_per_gpu: Batch size per GPU
        num_gpus: Number of GPUs

    Returns:
        Total batch size
    """
    return batch_size_per_gpu * num_gpus


def print_gpu_info():
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        print("[gpu] No GPUs available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"[gpu] {num_gpus} GPU(s) detected:")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(
            f"  GPU {i}: {props.name}\n"
            f"    Memory: {memory_gb:.2f} GB\n"
            f"    Compute: {props.major}.{props.minor}"
        )