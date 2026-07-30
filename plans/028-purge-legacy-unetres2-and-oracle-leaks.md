# Plan 028: Purge Legacy `UNetRes2` Models & Oracle Data Leaks from `src/`

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 631bac8..HEAD -- src/phase_unwrap/model/ src/phase_unwrap/core/config.py src/phase_unwrap/data/augmentation.py src/phase_unwrap/training/train.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: cleanup
- **Planned at**: commit `631bac8`, 2026-07-30

## Why this matters

The codebase has evolved to the Physics-Constrained Latent Corrector Network (**PCLCN**). Retaining legacy baseline architectures (`UNetRes2_AbsPhase`), obsolete data hints (`gt_center` oracle leak), and deprecated facade shims in `src/` adds code bloat, slows down maintenance, and creates reviewer confusion. 

Per version-control best practices, previous git branches store historical baselines. `src/phase_unwrap/` should exclusively contain the active state-of-the-art architecture: **PCLCN**.

## Current state

- `src/phase_unwrap/model/unetres2.py` — Contains legacy `UNetRes2_AbsPhase`.
- `src/phase_unwrap/model/unet.py` — Legacy facade shim re-exporting `UNetRes2_AbsPhase`.
- `src/phase_unwrap/model/blocks.py` — Contains `UpBlockRes2` (only used by `UNetRes2`).
- `src/phase_unwrap/core/config.py` — `ModelConfig.arch` defaults to `"unetres2"`; `AugmentationConfig.hint_mode` supports `"gt_center"`.
- `src/phase_unwrap/training/train.py` — Contains dual branching logic for `PCLCNModel` vs `UNetRes2_AbsPhase`.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit Tests | `uv run pytest tests/ -q` | All remaining unit tests pass |
| Lint Check | `uv run ruff check src/phase_unwrap/` | Exit 0, no errors |
| Type Check | `uv run mypy src/phase_unwrap/` | Exit 0, no errors |

## Scope

**In scope**:
- Delete `src/phase_unwrap/model/unetres2.py` and `src/phase_unwrap/model/unet.py`.
- Remove `UpBlockRes2` from `src/phase_unwrap/model/blocks.py`.
- Simplify `src/phase_unwrap/model/factory.py` so `build_model()` directly constructs `PCLCNModel`.
- Update `src/phase_unwrap/model/__init__.py` to export PCLCN primitives only.
- Remove `gt_center` oracle hint mode from `src/phase_unwrap/core/config.py` and `src/phase_unwrap/data/augmentation.py`.
- Remove `UNetRes2` branching in `src/phase_unwrap/training/train.py` and `src/phase_unwrap/analysis/`.
- Update `tests/` to remove tests targeting deleted `UNetRes2` modules.

**Out of scope**:
- PCLCN operators in `src/phase_unwrap/core/ops.py`.
- Loss functions in `src/phase_unwrap/core/losses.py`.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit per step with imperative message: e.g. `Purge legacy UNetRes2 architecture and gt_center oracle leak`

---

## Steps

### Step 1: Remove `UNetRes2` from `model/` package

1. Delete `src/phase_unwrap/model/unetres2.py`.
2. Delete `src/phase_unwrap/model/unet.py`.
3. Remove `UpBlockRes2` from `src/phase_unwrap/model/blocks.py`.
4. Update `src/phase_unwrap/model/factory.py`:

```python
"""Model factory for PCLCN architecture."""

from __future__ import annotations
import torch.nn as nn
from ..core.config import ModelConfig
from .pclcn import PCLCNModel

def build_model(cfg: ModelConfig) -> PCLCNModel:
    """Construct PCLCNModel from configuration."""
    return PCLCNModel(height=128, width=128, base=cfg.base)
```

5. Update `src/phase_unwrap/model/__init__.py`:

```python
"""PCLCN model sub-package."""

from .blocks import EMA, AddCoords, Res2_DS_Block
from .factory import build_model
from .pclcn import OrthogonalResidualCNN, PCLCNModel, ReferenceConditionedGradientCorrector

__all__ = [
    "PCLCNModel",
    "ReferenceConditionedGradientCorrector",
    "OrthogonalResidualCNN",
    "Res2_DS_Block",
    "AddCoords",
    "EMA",
    "build_model",
]
```

### Step 2: Purge `gt_center` oracle leak from `config.py` and `augmentation.py`

In `src/phase_unwrap/core/config.py`:
- Change `ModelConfig.arch` default to `"pclcn"`.
- Remove `"gt_center"` from `AugmentationConfig.hint_mode` documentation and validation.

In `src/phase_unwrap/data/augmentation.py`:
- Remove `gt_center` calculation branch in `prepare_batch()`.

### Step 3: Streamline `train.py` for PCLCN only

In `src/phase_unwrap/training/train.py`:
- Remove `if hasattr(model, "corrector"):` and `else:` branches for `UNetRes2`.
- Direct execution path: `phi_abs, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)`.

### Step 4: Update tests and verify

Delete or update legacy test cases in `tests/test_model.py` that reference `UNetRes2_AbsPhase` or `unet.py`.

```bash
uv run pytest tests/ -q
uv run ruff check src/phase_unwrap/
```

---

## Test plan

- Execute Pytest suite verifying `PCLCNModel` construction, 3-tuple data loader execution, and training step.
- Verification command: `uv run pytest tests/ -q`

## Done criteria

- [ ] `UNetRes2_AbsPhase`, `unetres2.py`, and `unet.py` completely purged.
- [ ] `gt_center` oracle leak removed from config and augmentation.
- [ ] `build_model()` returns `PCLCNModel` directly.
- [ ] Pytest suite passes 100%.

## STOP conditions

- If any non-legacy file requires `UNetRes2_AbsPhase`, update caller to `PCLCNModel` before proceeding.
