"""
Analysis sub-package: baselines, noise robustness, GradCAM, TTA, export, tables.
"""

from .baselines import evaluate_baselines, evaluate_dl_baseline
from .export import benchmark_inference, export_onnx, export_torchscript
from .export_latex import comparison_to_latex, metrics_to_latex
from .gradcam import GradCAM
from .noise_sweep import compute_noise_robustness
from .tta import evaluate_tta, predict_tta

__all__ = [
    "evaluate_baselines",
    "evaluate_dl_baseline",
    "export_onnx",
    "export_torchscript",
    "benchmark_inference",
    "metrics_to_latex",
    "comparison_to_latex",
    "GradCAM",
    "compute_noise_robustness",
    "evaluate_tta",
    "predict_tta",
]