"""Self-check: render all 6 figures from a fake run dir (untrained model + synthetic metrics).

Proves the plot wiring end-to-end without a real training run.
    uv run python scripts/smoke_plot.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

from phase_unwrap.cli_cmds._plot_runners import (
    run_f1,
    run_f2,
    run_f3,
    run_f4,
    run_f5,
    run_f6,
)
from phase_unwrap.core.config import TrainConfig, config_to_yaml
from phase_unwrap.model import build_model


def _make_run(tmp: Path) -> Path:
    cfg = TrainConfig()
    cfg.model.base = 8
    run = tmp / "run"
    run.mkdir()
    model = build_model(cfg.model)
    torch.save({"model": model.state_dict(), "model_ema": model.state_dict()}, run / "best.pth")
    (run / "config.yaml").write_text(config_to_yaml(cfg))
    (run / "result.json").write_text(json.dumps({
        "test_by_severity": {
            "0.0": {"TopoMAE": 0.10}, "0.25": {"TopoMAE": 0.14},
            "0.5": {"TopoMAE": 0.21}, "0.75": {"TopoMAE": 0.30}, "1.0": {"TopoMAE": 0.42},
        },
        "test_by_snr": {
            "5.0": {"TopoMAE": 0.85}, "10.0": {"TopoMAE": 0.55},
            "15.0": {"TopoMAE": 0.35}, "20.0": {"TopoMAE": 0.22},
            "25.0": {"TopoMAE": 0.16}, "30.0": {"TopoMAE": 0.13},
            "35.0": {"TopoMAE": 0.11}, "40.0": {"TopoMAE": 0.10},
            "inf": {"TopoMAE": 0.09},
        },
    }))
    return run


def _make_ablation(tmp: Path) -> Path:
    abl = tmp / "ablation"
    summary = {}
    for label in ("pclcn_full", "no_curl_loss", "unmasked_zernike"):
        grp = abl / label
        grp.mkdir(parents=True)
        seed_vals = []
        for seed in (1, 2):
            sd = grp / f"seed_{seed}"
            sd.mkdir()
            row = {"AbsMAE": 0.02 + 0.01 * seed, "RMSE": 0.03 + 0.01 * seed,
                   "SSIM": 0.95 - 0.01 * seed, "PSNR": 30.0, "MaxErr": 0.1, "GradMAE": 0.05}
            seed_vals.append(row)
            pd.DataFrame([row]).to_csv(sd / "test_metrics.csv", index=False)
        agg = pd.DataFrame([{
            "AbsMAE_mean": sum(r["AbsMAE"] for r in seed_vals) / len(seed_vals),
            "AbsMAE_std": 0.01,
            "RMSE_mean": sum(r["RMSE"] for r in seed_vals) / len(seed_vals),
            "RMSE_std": 0.01,
            "SSIM_mean": sum(r["SSIM"] for r in seed_vals) / len(seed_vals),
            "SSIM_std": 0.01,
        }])
        agg.to_csv(grp / "aggregate_test.csv", index=False)
        summary[label] = {"run_dir": str(grp), "aggregate_test_csv": str(grp / "aggregate_test.csv")}
    (abl / "ablation_summary.json").write_text(json.dumps(summary))
    return abl


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "figs"
    out.mkdir()
    run = _make_run(tmp)
    abl = _make_ablation(tmp)

    kwargs = {"run_dir": str(run), "out_dir": str(out), "n_samples": 2, "device": "cpu"}
    run_f1(**kwargs); run_f2(**kwargs); run_f3(run_dir=str(run), out_dir=str(out))
    run_f5(**kwargs); run_f6(**kwargs)
    run_f4(run_dir=str(abl), out_dir=str(out))

    expected = [
        "fig1_architecture.png", "fig2_baseline_comparison.png", "fig3_noise_robustness.png",
        "fig4_ablation_radar.png", "fig4_ablation_boxen.png", "fig5_diagnostics.png",
        "fig6_physics_zernike.png",
    ]
    missing = [n for n in expected if not (out / n).exists()]
    if missing:
        print(f"FAIL: missing {missing}")
        return 1
    print(f"OK: {len(expected)} figures rendered to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
