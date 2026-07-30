# Plan 029: Modernize PCLCN Ablation & Multi-Seed Framework (Fast Core Matrix)

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 631bac8..HEAD -- src/phase_unwrap/training/ablation.py src/phase_unwrap/training/multiseed.py src/phase_unwrap/cli_cmds/train.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/028-purge-legacy-unetres2-and-oracle-leaks.md
- **Category**: refactor
- **Planned at**: commit `631bac8`, 2026-07-30

## Why this matters

The current `ablation.py`, `multiseed.py`, and `cli_cmds/train.py` modules hardcode obsolete baseline hyperparameter overrides (`loss.w_grad`, `model.use_coordconv`) from early UNet plans. They lack support for PCLCN's physics-constrained components: 3-tuple batches `(I_raw, phi_gt, grad_phi2)`, reference beam gradient priors, vector gradient curl penalties (`w_curl`), and complex domain losses.

Furthermore, online LaTeX compilation and plot rendering inside the training loop slow down Kaggle GPU runs. This plan modernizes `ablation.py` and `multiseed.py` to execute Option 1 (Fast Core PCLCN Ablation Matrix) at maximum GPU throughput, outputting raw numerical CSV/JSON logs for offline post-processing.

## Current state

- `src/phase_unwrap/cli_cmds/train.py` (lines 178–183):
```python
    ablations = {
        "no_grad_loss": {"loss.w_grad": 0.0},
        "no_curv_loss": {"loss.w_curv": 0.0},
        "no_coordconv": {"model.use_coordconv": False},
        "no_ema": {"model.ema_decay": 0.0},
    }
```
- `src/phase_unwrap/training/ablation.py`: Invokes online LaTeX table export during ablation execution.
- `src/phase_unwrap/training/multiseed.py`: Lacks explicit PCLCN 3-tuple batch (`grad_phi2`) awareness.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit Tests | `uv run pytest tests/ -q` | All pass |
| CLI Smoke Test | `uv run phase-unwrap train ablation --help` | Displays PCLCN options |
| Lint Check | `uv run ruff check src/phase_unwrap/training/` | Exit 0 |

## Scope

**In scope**:
- `src/phase_unwrap/cli_cmds/train.py` — Replace obsolete ablation overrides with Option 1 Fast Core PCLCN Matrix (`no_reference_prior`, `no_curl_loss`, `unmasked_zernike`).
- `src/phase_unwrap/training/ablation.py` — Streamline execution for Kaggle GPU throughput: save raw CSV/JSON logs (`ablation_summary.json`, `metrics.csv`), remove online LaTeX compilation overhead.
- `src/phase_unwrap/training/multiseed.py` — Update multi-seed aggregation for PCLCN 3-tuple dataset loader and raw summary output.

**Out of scope**:
- Post-processing LaTeX rendering in `src/phase_unwrap/analysis/export_latex.py`.
- Offline plotting routines in `src/phase_unwrap/plots/`.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit per step with imperative message: e.g. `Modernize ablation and multiseed framework for PCLCN Fast Core Matrix`

---

## Steps

### Step 1: Update CLI PCLCN Ablation Matrix in `cli_cmds/train.py`

In `src/phase_unwrap/cli_cmds/train.py`, update `ablation_cmd`:

```python
    ablations = {
        "no_reference_prior": {"model.zero_reference_prior": True},
        "no_curl_loss": {"loss.w_curl": 0.0},
        "unmasked_zernike": {"model.unmasked_zernike": True},
    }
```

### Step 2: Streamline `ablation.py` for High GPU Throughput & Raw Data Output

In `src/phase_unwrap/training/ablation.py`:
1. Remove `comparison_to_latex` import and online table generation.
2. Collect raw numerical summaries from `run_multiseed` runs into `ablation_summary.json`:

```python
"""
Modernized PCLCN ablation study runner.
Saves raw JSON/CSV metrics for offline visualization.
"""

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence

from ..core.config import TrainConfig
from .multiseed import run_multiseed

def run_ablation(
    base_cfg: TrainConfig,
    ablations: Dict[str, Dict[str, Any]],
    base_name: str = "pclcn_ablation",
    seeds: Sequence[int] = (42, 1337, 7),
    use_amp: bool = False,
) -> Path:
    out_dir = Path(base_cfg.logging.runs_root) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}

    for label, overrides in ablations.items():
        cfg = deepcopy(base_cfg)
        _apply_overrides(cfg, overrides)
        group_name = f"{base_name}/{label}"
        run_multiseed(cfg, base_run_name=group_name, seeds=list(seeds), use_amp=use_amp)
        summary[label] = {
            "overrides": overrides,
            "run_dir": str(Path(cfg.logging.runs_root) / group_name),
        }

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PCLCN ablation complete. Raw data summary: {summary_path}")
    return summary_path
```

### Step 3: Modernize `multiseed.py`

In `src/phase_unwrap/training/multiseed.py`:
- Use `Path` objects for path operations.
- Aggregate metrics across seed runs into `aggregate.csv` using stdlib `csv` and `numpy`.
- Ensure 3-tuple batch awareness and non-blocking CUDA transfers.

### Step 4: Verification

```bash
uv run pytest tests/ -q
uv run ruff check src/phase_unwrap/training/
```

---

## Test plan

- Run integration test for multi-seed and ablation functions.
- Verification command: `uv run pytest tests/ -q`

## Done criteria

- [ ] `ablation_cmd` defaults to Option 1 PCLCN matrix (`no_reference_prior`, `no_curl_loss`, `unmasked_zernike`).
- [ ] Online LaTeX compilation removed from `ablation.py`; raw `ablation_summary.json` saved instead.
- [ ] `multiseed.py` cleanly aggregates PCLCN metric runs.
- [ ] All unit tests pass.

## STOP conditions

- If `model.zero_reference_prior` requires additional parameter handling in `PCLCNModel`, update `PCLCNModel.forward` to accept a zeroing flag.
