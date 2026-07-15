"""Import smoke for analysis entrypoints fixed by plan 014."""

from __future__ import annotations

import inspect

from phase_unwrap.analysis.export import (
    benchmark_inference,
    export_onnx,
    export_torchscript,
)
from phase_unwrap.analysis.gradcam import plot_gradcam
from phase_unwrap.analysis.tta import evaluate_tta, predict_tta


def test_plot_gradcam_signature() -> None:
    sig = inspect.signature(plot_gradcam)
    assert "checkpoint_path" in sig.parameters
    assert "data_dir" in sig.parameters


def test_export_and_tta_importable() -> None:
    assert callable(export_torchscript)
    assert callable(benchmark_inference)
    assert callable(export_onnx)
    assert callable(evaluate_tta)
    assert callable(predict_tta)
