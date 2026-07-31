"""
ONNX and TorchScript model export + inference benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..core import _torch_compat  # noqa: F401


def export_onnx(
    checkpoint_path: str,
    out_path: str = "results/model.onnx",
    opset: int = 17,
    config_path: str | None = None,
) -> None:
    """Export PCLCN model to ONNX."""
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir="", config_path=config_path)

    I_raw = torch.randn(1, 1, 128, 128, device=device)
    grad_phi2 = torch.randn(1, 2, 128, 128, device=device)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (I_raw, grad_phi2),
        out_path,
        opset_version=opset,
        input_names=["I_raw", "grad_phi2"],
        output_names=["phi_final", "gx_tilde", "gy_tilde", "c_zernike", "phi_zernike"],
    )

    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Exported PCLCN ONNX: {out_path} ({size_mb:.1f} MB, opset {opset})")


def export_torchscript(
    checkpoint_path: str,
    out_path: str = "results/model.pt",
    config_path: str | None = None,
) -> None:
    """Export PCLCN model to TorchScript (traced) format."""
    from ..core.config import load_train_config
    from ..model import build_model

    run_dir = str(Path(checkpoint_path).parent)
    cfg_file = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_file if Path(cfg_file).exists() else None)

    model = build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    I_raw = torch.randn(1, 1, 128, 128)
    grad_phi2 = torch.randn(1, 2, 128, 128)
    with torch.no_grad():
        traced = torch.jit.trace(model, (I_raw, grad_phi2))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(out_path)

    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Exported PCLCN TorchScript: {out_path} ({size_mb:.1f} MB)")


def benchmark_inference(
    checkpoint_path: str,
    n_warmup: int = 10,
    n_runs: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
    config_path: str | None = None,
) -> dict[str, float]:
    """Benchmark mean latency and throughput."""
    from ..core.config import load_train_config
    from ..core.utils import pick_device
    from ..model import build_model

    run_dir = str(Path(checkpoint_path).parent)
    cfg_file = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_file if Path(cfg_file).exists() else None)

    dev = pick_device(device)
    model = build_model(cfg.model).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    I_raw = torch.randn(batch_size, 1, 128, 128, device=dev)
    grad_phi2 = torch.randn(batch_size, 2, 128, 128, device=dev)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(I_raw, grad_phi2)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(I_raw, grad_phi2)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    mean_ms = float(times_arr.mean())
    std_ms = float(times_arr.std())
    fps = batch_size * 1000.0 / mean_ms

    print(f"Inference benchmark ({dev}, batch={batch_size}, n={n_runs}):")
    print(f"  Latency:    {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Throughput: {fps:.1f} samples/s")

    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput_fps": fps}
