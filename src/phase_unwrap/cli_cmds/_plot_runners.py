"""Plot runners: load saved run artifacts and render the 6 paper figures.

Sample-driven figures (1,2,5,6) rebuild OTF samples from the saved config
seeds (deterministic, no dataset copy) and re-run PCLCN inference from best.pth.
Metric figures (3,4) read saved JSON/CSV offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..analysis.baselines import _unwrap_itoh, _unwrap_skimage
from ..core.config import load_train_config
from ..core.ops import piston_align
from ..core.utils import pick_device
from ..data import build_otf_loaders
from ..data.augmentation import NoiseAug, normalize_intensity
from ..model import build_model
from ..plots.fig1_architecture import plot_f1_architecture
from ..plots.fig2_baseline_comparison import plot_f2_baseline_comparison
from ..plots.fig3_noise_robustness import plot_f3_dual_panel
from ..plots.fig4_ablation import plot_f4_multiseed, plot_f4_radar
from ..plots.fig5_diagnostics import plot_f5_diagnostics
from ..plots.fig6_physics_zernike import plot_f6_physics_zernike

_FIG_METRICS = ["AbsMAE", "RMSE", "SSIM"]


def _load_model(run_dir: str, device: torch.device, config: str | None = None):
    cfg = load_train_config(config or str(Path(run_dir) / "config.yaml"))
    model = build_model(cfg.model).to(device).eval()
    ckpt = torch.load(str(Path(run_dir) / "best.pth"), map_location=device, weights_only=True)
    sd = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    return model, cfg


@torch.no_grad()
def _otf_batch(cfg, device, n: int = 4, split: str = "test"):
    _, val_loader, test_loader = build_otf_loaders(cfg, device)
    loader = test_loader if split == "test" else val_loader
    I, phi, g, _ = next(iter(loader))
    return I[:n].to(device), phi[:n].to(device), g[:n].to(device)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def run_f1(run_dir: str, out_dir: str, n_samples: int = 4, device: str = "auto", config: str | None = None, **_):
    dev = pick_device(device)
    model, cfg = _load_model(run_dir, dev, config)
    I_raw, phi_gt, grad_phi2 = _otf_batch(cfg, dev, n=n_samples)
    phi_abs, *_ = model(normalize_intensity(I_raw), grad_phi2)
    phi_aligned, _ = piston_align(phi_abs, phi_gt)
    aug = NoiseAug()
    gen = torch.Generator(device=dev).manual_seed(0)
    I_noisy, _ = aug(I_raw, 1.0, generator=gen)
    plot_f1_architecture(
        raw_clean_np=_np(I_raw),
        raw_noisy_np=_np(I_noisy),
        gt_np=_np(phi_gt),
        pred_np=_np(phi_aligned),
        filepath=str(Path(out_dir) / "fig1_architecture.png"),
    )


@torch.no_grad()
def run_f2(run_dir: str, out_dir: str, n_samples: int = 3, device: str = "auto", config: str | None = None, **_):
    dev = pick_device(device)
    model, cfg = _load_model(run_dir, dev, config)
    I_raw, phi_gt, grad_phi2 = _otf_batch(cfg, dev, n=n_samples)
    phi_abs, *_ = model(normalize_intensity(I_raw), grad_phi2)
    phi_aligned, _ = piston_align(phi_abs, phi_gt)
    raw_np, gt_np, pred_np = _np(I_raw), _np(phi_gt), _np(phi_aligned)
    itoh_imgs, lsq_imgs = [], []
    for row in range(len(I_raw)):
        I_img, gt_img = raw_np[row, 0], gt_np[row, 0]
        for fn, lst in ((_unwrap_itoh, itoh_imgs), (_unwrap_skimage, lsq_imgs)):
            try:
                p = fn(I_img)
                al, _ = piston_align(
                    torch.from_numpy(p)[None, None].float(),
                    torch.from_numpy(gt_img)[None, None].float(),
                )
                lst.append(al.numpy()[0])
            except Exception:  # noqa: BLE001 — baseline failure → zeros fallback
                lst.append(np.zeros_like(gt_img)[None])
    plot_f2_baseline_comparison(
        raw_i_np=raw_np,
        gt_np=gt_np,
        itoh_np=np.stack(itoh_imgs),
        lsq_np=np.stack(lsq_imgs),
        unet_np=pred_np,
        filepath=str(Path(out_dir) / "fig2_baseline_comparison.png"),
    )


def run_f3(run_dir: str, out_dir: str, **_):
    rj = json.loads(Path(run_dir, "result.json").read_text())
    sev = rj.get("test_by_severity", {})
    snr = rj.get("test_by_snr", {})
    if not sev:
        print("fig3: no test_by_severity in result.json, skipping")
        return
    if not snr:
        print("fig3: no test_by_snr in result.json, skipping")
        return

    def _to_results(data):
        items = sorted((float(k), v) for k, v in data.items())
        return {
            "snr_db": [k for k, _ in items],
            "mae": [v.get("TopoMAE", float("nan")) for _, v in items],
            "std": [0.0] * len(items),
        }

    severity_results = _to_results(sev)
    snr_results = _to_results(snr)

    plot_f3_dual_panel(
        severity_results,
        snr_results,
        filepath=str(Path(out_dir) / "fig3_noise_robustness.png"),
    )


def run_f4(run_dir: str, out_dir: str, **_):
    base = Path(run_dir)
    out = Path(out_dir)
    rows: list[dict] = []
    radar: dict[str, dict[str, float]] = {}
    summary_path = base / "ablation_summary.json"

    def _per_seed(label: str, group_dir: Path) -> None:
        for sd in sorted(group_dir.glob("seed_*")):
            tmf = sd / "test_metrics.csv"
            if tmf.exists():
                tm = pd.read_csv(tmf).iloc[0]
                for m in _FIG_METRICS:
                    rows.append({"method": label, "metric": m, "value": float(tm[m])})

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for label, info in summary.items():
            agg = info.get("aggregate_test_csv")
            if agg and Path(agg).exists():
                row = pd.read_csv(agg).iloc[0]
                radar[label] = {m: float(row.get(f"{m}_mean")) for m in _FIG_METRICS}
            _per_seed(label, Path(info["run_dir"]))
        if radar:
            baseline = "pclcn_full" if "pclcn_full" in radar else next(iter(radar))
            plot_f4_radar(radar, baseline_name=baseline, filepath=str(out / "fig4_ablation_radar.png"))
    else:
        _per_seed(base.name, base)

    if rows:
        plot_f4_multiseed(pd.DataFrame(rows), _FIG_METRICS, filepath=str(out / "fig4_ablation_boxen.png"))
    if not radar and not rows:
        print("fig4: no ablation/multiseed data in run_dir, skipping")


@torch.no_grad()
def run_f5(run_dir: str, out_dir: str, n_samples: int = 1, device: str = "auto", config: str | None = None, **_):
    dev = pick_device(device)
    model, cfg = _load_model(run_dir, dev, config)
    I_raw, phi_gt, grad_phi2 = _otf_batch(cfg, dev, n=1)
    phi_abs, *_ = model(normalize_intensity(I_raw), grad_phi2)
    phi_aligned, _ = piston_align(phi_abs, phi_gt)
    plot_f5_diagnostics(gt_np=_np(phi_gt), pred_np=_np(phi_aligned), filepath=str(Path(out_dir) / "fig5_diagnostics.png"))


@torch.no_grad()
def run_f6(run_dir: str, out_dir: str, n_samples: int = 1, device: str = "auto", config: str | None = None, **_):
    dev = pick_device(device)
    model, cfg = _load_model(run_dir, dev, config)
    I_raw, phi_gt, grad_phi2 = _otf_batch(cfg, dev, n=1)
    I_input = normalize_intensity(I_raw)
    phi_abs, gx_t, gy_t, c_pred, _ = model(I_input, grad_phi2)
    wrapped, _ = model.stem(I_input)
    gx, gy = model.grad_op(wrapped)
    curl_raw = (torch.diff(gy, dim=-1, prepend=gy[..., :1]) - torch.diff(gx, dim=-2, prepend=gx[..., :1, :])).abs()
    curl_corr = (torch.diff(gy_t, dim=-1, prepend=gy_t[..., :1]) - torch.diff(gx_t, dim=-2, prepend=gx_t[..., :1, :])).abs()
    c_gt, _ = model.zernike_proj(phi_gt)
    phi_aligned, _ = piston_align(phi_abs, phi_gt)
    plot_f6_physics_zernike(
        curl_raw_np=_np(curl_raw),
        curl_corr_np=_np(curl_corr),
        c_gt_np=_np(c_gt),
        c_pred_np=_np(c_pred),
        gt_2d_np=_np(phi_gt),
        pred_2d_np=_np(phi_aligned),
        filepath=str(Path(out_dir) / "fig6_physics_zernike.png"),
    )


RUNNERS = {"1": run_f1, "2": run_f2, "3": run_f3, "4": run_f4, "5": run_f5, "6": run_f6}
