"""
ONNX and TorchScript model export.

Exports a trained checkpoint to portable inference formats
with shape validation and optional simplification.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig, load_train_config
from .model import build_model


def export_onnx(
    checkpoint_path: str,
    out_path: str = "results/model.onnx",
    opset: int = 17,
    config_path: str | None = None,
) -> None:
    """
    Export model to ONNX format.

    Args:
        checkpoint_path: Path to trained .pth checkpoint.
        out_path:        Output ONNX file path.
        opset:           ONNX opset version (17 recommended for PyTorch 2.x).
        config_path:     Optional config override for model architecture.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_file = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_file if Path(cfg_file).exists() else None)

    model = build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # prefer EMA weights
    state_key = "model_ema" if "model_ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    dummy = torch.randn(1, 2, 128, 128)  # (B, C_in, H, W)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=opset,
        input_names=["interferogram"],
        output_names=["phi_raw", "k_offset"],
        dynamic_axes={
            "interferogram": {0: "batch", 2: "height", 3: "width"},
            "phi_raw": {0: "batch", 2: "height", 3: "width"},
            "k_offset": {0: "batch"},
        },
    )

    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Exported ONNX: {out_path} ({size_mb:.1f} MB, opset {opset})")

    # validation roundtrip
    try:
        import onnx

        m = onnx.load(out_path)
        onnx.checker.check_model(m)
        print("  ONNX model validation passed.")
    except ImportError:
        print("  Install 'onnx' package for model validation.")
    except Exception as e:
        print(f"  ONNX validation warning: {e}")


def export_torchscript(
    checkpoint_path: str,
    out_path: str = "results/model.pt",
    config_path: str | None = None,
) -> None:
    """
    Export model to TorchScript (traced) format.

    Args:
        checkpoint_path: Path to trained .pth checkpoint.
        out_path:        Output .pt file path.
        config_path:     Optional config override.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_file = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_file if Path(cfg_file).exists() else None)

    model = build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "model_ema" if "model_ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    dummy = torch.randn(1, 2, 128, 128)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(out_path)

    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Exported TorchScript: {out_path} ({size_mb:.1f} MB)")

    # verify shapes
    with torch.no_grad():
        phi_raw, k_off = traced(dummy)
    print(f"  phi_raw shape: {list(phi_raw.shape)}")
    print(f"  k_off shape:   {list(k_off.shape)}")


def benchmark_inference(
    checkpoint_path: str,
    n_warmup: int = 10,
    n_runs: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
    config_path: str | None = None,
) -> dict[str, float]:
    """
    Benchmark inference latency.

    Returns:
        Dict with mean_ms, std_ms, throughput_fps.
    """
    import time

    from .utils import pick_device

    run_dir = str(Path(checkpoint_path).parent)
    cfg_file = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_file if Path(cfg_file).exists() else None)

    dev = pick_device(device)
    model = build_model(cfg.model).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state_key = "model_ema" if "model_ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    dummy = torch.randn(batch_size, 2, 128, 128, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # timed runs
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    import numpy as np

    times_arr = np.array(times)
    mean_ms = float(times_arr.mean())
    std_ms = float(times_arr.std())
    fps = batch_size * 1000.0 / mean_ms

    print(f"Inference benchmark ({dev}, batch={batch_size}, n={n_runs}):")
    print(f"  Latency: {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Throughput: {fps:.1f} samples/s")

    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput_fps": fps}
