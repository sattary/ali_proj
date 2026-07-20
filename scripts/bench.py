"""
Pure-compute training-step microbenchmark.

Isolates the model's forward+backward+optimizer cost from the data pipeline,
augmentation, and metrics. Runs on synthetic tensors so the only variable is
GPU compute. Compare the reported it/s against the it/s you see in a real
`phase-unwrap train` run:

  - bench it/s ~= real it/s   -> the model architecture is the wall (compute-bound).
                                 The Res2 depthwise-separable convs are memory-bound
                                 and pin consumer GPUs at 100% util for low throughput.
                                 Lever: reduce model.base, or change the block design.
  - bench it/s >> real it/s   -> the model is fine; overhead lives in the training
                                 loop (data loading, augmentation, per-step .item()
                                 host syncs, or eager-mode kernel launch overhead).

Usage (on the GPU box):
    uv run python scripts/bench.py -c my_windows_config.yaml
    uv run python scripts/bench.py -c my_windows_config.yaml --sweep-base 8 16 24 32
"""

from __future__ import annotations

import argparse
import time

import torch

from phase_unwrap.core.config import load_train_config
from phase_unwrap.core.losses import MAEGradLoss
from phase_unwrap.model import build_model


def bench_once(base: int, batch_size: int, use_amp: bool, cfg, n_iter: int = 50) -> float:
    """Return iterations/sec for a forward+backward+step on synthetic data."""
    dev = torch.device("cuda")
    cfg.model.base = base
    model = build_model(cfg.model).to(dev, memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Real loss: handles the multi-scale output list exactly like train.py does,
    # so we never have to guess the model's output resolution.
    loss_fn = MAEGradLoss(
        w_mae=cfg.loss.w_mae,
        w_grad=cfg.loss.w_grad,
        intensity_weighted=cfg.loss.int_wgrad,
    )

    # Input is [B, 2, 128, 128]: normalized intensity + phi_hint channel.
    x = torch.randn(batch_size, 2, 128, 128, device=dev).contiguous(
        memory_format=torch.channels_last
    )
    tgt = torch.randn(batch_size, 1, 128, 128, device=dev)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            phi_raw, k_off = model(x)
            if isinstance(phi_raw, list):
                phi_abs = [p + k_off for p in phi_raw]
            else:
                phi_abs = phi_raw + k_off
            loss, _ = loss_fn(phi_abs, tgt, None)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(10):  # warmup: cuDNN autotune + allocator warmup
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    print(
        f"base={base:>3} bs={batch_size:>3} amp={int(use_amp)} | "
        f"{n_iter / dt:6.2f} it/s | {dt / n_iter * 1000:6.1f} ms/it | "
        f"params={n_params/1e6:5.2f}M | peak={peak_mb:6.0f}MB"
    )
    return n_iter / dt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True, help="Path to training YAML.")
    p.add_argument(
        "--sweep-base",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of model.base values to sweep, e.g. --sweep-base 8 16 32.",
    )
    p.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP to match `train --use-amp` (config leaves use_amp unset).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark requires a GPU.")

    cfg = load_train_config(args.config)
    use_amp = bool(cfg.model.use_amp) or args.amp
    bs = cfg.optim.batch_size

    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"config base={cfg.model.base} batch_size={bs} use_amp={use_amp}\n")

    bases = args.sweep_base if args.sweep_base else [cfg.model.base]
    for base in bases:
        bench_once(base, bs, use_amp, cfg, n_iter=args.iters)


if __name__ == "__main__":
    main()
