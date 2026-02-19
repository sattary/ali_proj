"""
Training sub-package: training loop, multi-seed, HPO, and ablation.
"""

from .ablation import run_ablation
from .multi_gpu import (
    calculate_total_batch_size,
    detect_kaggle_multi_gpu,
    get_model_state_dict,
    load_model_state_dict,
    print_gpu_info,
    setup_multi_gpu,
)
from .multiseed import run_multiseed
from .train import train
from .tune import run_tuning

__all__ = [
    "train",
    "run_multiseed",
    "run_tuning",
    "run_ablation",
    "setup_multi_gpu",
    "get_model_state_dict",
    "load_model_state_dict",
    "detect_kaggle_multi_gpu",
    "calculate_total_batch_size",
    "print_gpu_info",
]
