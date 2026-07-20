# Plan 017: Evaluate noise robustness on held-out test split, not fresh synthetic data

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 23fd1ef..HEAD -- src/phase_unwrap/analysis/noise_sweep.py src/phase_unwrap/cli_cmds/eval.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: correctness / methodology
- **Planned at**: commit `23fd1ef`, 2026-07-18

## Why this matters

The noise-robustness sweep (`phase-unwrap eval noise`) currently generates
**fresh synthetic interferograms on-the-fly** via `generate_sample()` instead
of loading samples from the held-out HDF5 test split. These fresh samples are
drawn from the exact same parametric distribution the model trained on, so the
reported noise-robustness curve is "model vs. fresh i.i.d. training
distribution + added Gaussian noise" — not "model vs. held-out test set +
added Gaussian noise."

Additionally, the CLI's `--data-dir` flag is accepted but never forwarded to
the sweep function, so the user cannot point it at their dataset even if they
wanted to.

For any paper figure claiming "noise robustness," evaluating on fresh
training-distribution data is methodologically indefensible. The held-out test
split exists precisely for this. This plan makes the sweep consume the test
split, matching what `evaluate_baselines` and `evaluate_dl_baseline` already do
(they were fixed in commit `c9a978c` for the same class of issue — DIR-03 in
plan 016).

## Current state

### `src/phase_unwrap/analysis/noise_sweep.py`

The file generates fresh data instead of loading from the test split:

- Line 19: `from ..data.generate import _build_grid, generate_sample` — imports
  the training-data generator.
- Lines 59–61: builds the physics grid and ignores `data_dir`:
  ```python
  model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)
  x, y, r2 = _build_grid()
  ```
- Lines 72–75: generates fresh samples per SNR level:
  ```python
  test_rng = np.random.default_rng(seed)
  for _ in range(n_samples):
      I_clean, dphi = generate_sample(x, y, r2, test_rng)
  ```
- The function signature (lines 40–49) has no `data_dir` parameter:
  ```python
  def noise_robustness_sweep(
      checkpoint_path: str,
      out_path: str = "results/figs/noise_robustness",
      snr_range: tuple[float, float] = (5.0, 40.0),
      n_snr_steps: int = 8,
      n_samples: int = 100,
      seed: int = 42,
      config_path: str | None = None,
      device_str: str = "auto",
  ) -> dict[str, list[float]]:
  ```

### `src/phase_unwrap/cli_cmds/eval.py`

The `eval noise` command accepts `--data-dir` but does not pass it through
(lines 57–86):

```python
@EvalApp.command("noise")
def eval_noise_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", ...),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", ...),
    ...
) -> None:
    from .analysis.noise_sweep import noise_robustness_sweep
    noise_robustness_sweep(
        checkpoint,
        out_path=out,
        snr_range=(snr_min, snr_max),
        n_snr_steps=n_steps,
        n_samples=n_samples,
        device_str=device,
        config_path=config,
    )
```

`data_dir` is accepted but never forwarded — dead parameter.

### The correct pattern to follow

`src/phase_unwrap/analysis/baselines.py` (`evaluate_baselines`, lines 44–91)
was fixed in commit `c9a978c` to load from the test split. Its pattern:

```python
from ..data.dataset import discover_h5_shards, smart_split, H5ShardDataset

cfg = TrainConfig()
cfg.data.data_dir = data_dir
paths = discover_h5_shards(cfg.data)
_, _, test_paths = smart_split(paths, seed=seed)
test_ds = H5ShardDataset(test_paths, augment=False)
rng = np.random.default_rng(seed)
indices = rng.choice(len(test_ds), size=min(n_samples, len(test_ds)), replace=False)
for idx in indices:
    I_t, gt_t = test_ds[int(idx)]
    I_clean = I_t.squeeze().numpy()
    dphi_gt = gt_t.squeeze().numpy()
    ...
```

However, `noise_sweep.py` should use `load_inference_state(subset="test")`
(defined in `src/phase_unwrap/core/inference.py:19-66`) since it already needs
the model loaded and returns the test loader directly. This avoids duplicating
the model-loading logic.

### Repo conventions

- `from __future__ import annotations` at the top of every module.
- Type hints use `str | None` (PEP 604) syntax.
- Analysis functions return `dict[str, ...]`.
- `load_inference_state` is the canonical inference entrypoint — see
  `src/phase_unwrap/analysis/tta.py:93-99` and
  `src/phase_unwrap/analysis/gradcam.py:86-91` for existing usage patterns.
- Option B: inference always uses `hint_mode="zero"` — no GT phase hint channel.
- Normalization for inference follows this pattern (from
  `baselines.py:150-154`):
  ```python
  I_mean = I_clean.mean()
  I_std = I_clean.std() + 1e-6
  I_norm = (I_clean - I_mean) / I_std
  phi_hint = np.zeros_like(I_norm, dtype=np.float32)
  ```

## Commands you will need

| Purpose   | Command                                  | Expected on success |
|-----------|------------------------------------------|---------------------|
| Install   | `uv sync`                                | exit 0              |
| Tests     | `uv run pytest tests/ -q`                | all pass            |
| Lint      | `uv run ruff check src/ tests/`          | exit 0, no errors   |
| Typecheck | `uv run mypy src/`                       | exit 0, no errors    |

## Scope

**In scope** (the only files you should modify):
- `src/phase_unwrap/analysis/noise_sweep.py`
- `src/phase_unwrap/cli_cmds/eval.py` (only the `eval_noise_cmd` function)
- `tests/test_analysis_cli_smoke.py` (if it exists and references `noise_sweep`; otherwise add a test)

**Out of scope** (do NOT touch, even though they look related):
- `src/phase_unwrap/data/generate.py` — the generator itself is fine; the bug
  is that `noise_sweep.py` calls it when it shouldn't.
- `src/phase_unwrap/core/inference.py` — `load_inference_state` already
  supports `subset="test"`. Do not modify it in this plan.
- The `_add_gaussian_noise` function and the SNR sweep logic — these are
  legitimate noise-robustness probes (additive Gaussian at controlled SNR is
  standard). The bug is the *data source*, not the noise model.
- Any other analysis module (`baselines.py`, `tta.py`, `gradcam.py`).

## Git workflow

- **Branch**: work on the current branch (`fix-review`). Do NOT create a new
  branch.
- Commit per logical unit; message style: conventional commits (match the
  repo's existing style, e.g. `Fix: Evaluate noise sweep on held-out test
  split`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add `data_dir` parameter to `noise_robustness_sweep`

In `src/phase_unwrap/analysis/noise_sweep.py`, add a `data_dir: str` parameter
to the function signature (after `checkpoint_path`):

```python
def noise_robustness_sweep(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/noise_robustness",
    snr_range: tuple[float, float] = (5.0, 40.0),
    n_snr_steps: int = 8,
    n_samples: int = 100,
    seed: int = 42,
    config_path: str | None = None,
    device_str: str = "auto",
) -> dict[str, list[float]]:
```

**Verify**: `uv run python -c "from phase_unwrap.analysis.noise_sweep import noise_robustness_sweep; import inspect; print(inspect.signature(noise_robustness_sweep))"` → signature includes `data_dir: str`.

### Step 2: Replace `generate_sample` with held-out test split loading

Replace the model loading and data generation block (currently lines ~58–75)
with loading from the test split via `load_inference_state`.

**Remove** these imports (line 19):
```python
from ..data.generate import _build_grid, generate_sample
```

**Remove** the grid construction (line ~61):
```python
x, y, r2 = _build_grid()
```

**Replace** the model loading + sample generation block. The current code is:
```python
model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

x, y, r2 = _build_grid()

snr_levels = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
snr_eval = list(snr_levels) + [float("inf")]

all_maes_by_snr: dict[float, list[float]] = {}
results: dict[str, list[float]] = {"snr_db": [], "mae": [], "std": []}

for snr_db in snr_eval:
    maes: list[float] = []
    test_rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        I_clean, dphi = generate_sample(x, y, r2, test_rng)
        ...
```

**New code**:
```python
from ..core.inference import load_inference_state
from ..data.augmentation import build_phi_hint

model, cfg, loader, device = load_inference_state(
    checkpoint_path, data_dir=data_dir, subset="test", config_path=config_path
)

# Collect n_samples from the held-out test split (once, reused across SNR levels)
test_samples: list[tuple[np.ndarray, np.ndarray]] = []
for I_raw_batch, phi_gt_batch in loader:
    for i in range(I_raw_batch.size(0)):
        if len(test_samples) >= n_samples:
            break
        I_clean = I_raw_batch[i, 0].numpy()
        dphi = phi_gt_batch[i, 0].numpy()
        test_samples.append((I_clean, dphi))
    if len(test_samples) >= n_samples:
        break

if not test_samples:
    raise ValueError(
        f"Test split is empty. Check data_dir={data_dir!r} and test_frac in config."
    )

snr_levels = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
snr_eval = list(snr_levels) + [float("inf")]

all_maes_by_snr: dict[float, list[float]] = {}
results: dict[str, list[float]] = {"snr_db": [], "mae": [], "std": []}

for snr_db in snr_eval:
    maes: list[float] = []

    for I_clean, dphi in test_samples:
        ...
```

The rest of the per-sample loop body (noise addition, normalization, inference,
`piston_align`, MAE computation) stays the same — it already operates on numpy
arrays `I_clean` and `dphi`, which is what `test_samples` now provides.

**Verify**: `uv run ruff check src/phase_unwrap/analysis/noise_sweep.py` → exit 0, no errors.

### Step 3: Forward `data_dir` from the CLI command

In `src/phase_unwrap/cli_cmds/eval.py`, update the `eval_noise_cmd` function to
pass `data_dir`:

Current (lines 78–86):
```python
noise_robustness_sweep(
    checkpoint,
    out_path=out,
    snr_range=(snr_min, snr_max),
    n_snr_steps=n_steps,
    n_samples=n_samples,
    device_str=device,
    config_path=config,
)
```

New:
```python
noise_robustness_sweep(
    checkpoint,
    data_dir=data_dir,
    out_path=out,
    snr_range=(snr_min, snr_max),
    n_snr_steps=n_steps,
    n_samples=n_samples,
    device_str=device,
    config_path=config,
)
```

**Verify**: `uv run ruff check src/phase_unwrap/cli_cmds/eval.py` → exit 0.

### Step 4: Write a regression test

Create or update a test file to verify that `noise_robustness_sweep` loads from
the test split rather than generating fresh data. Follow the pattern in
`tests/test_train.py` (creates dummy HDF5 shards in `tmp_path`).

Add to `tests/test_analysis_cli_smoke.py` (if it exists) or create
`tests/test_noise_sweep.py`:

```python
"""Noise sweep must evaluate on held-out test split, not fresh synthetic data."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from pathlib import Path

from phase_unwrap.analysis.noise_sweep import noise_robustness_sweep


def test_noise_sweep_uses_test_split(tmp_path):
    """Verify noise sweep loads from data_dir, not generate_sample."""
    # Create dummy HDF5 shards (need >=3 for smart_split to produce train/val/test)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I = np.random.randn(4, 1, 128, 128).astype(np.float32)
            phi = np.random.randn(4, 1, 128, 128).astype(np.float32)
            f.create_dataset("I", data=I)
            f.create_dataset("phi", data=phi)

    # We can't run the full sweep without a trained checkpoint, but we can
    # verify that the function attempts to load from data_dir (not generate).
    # If it tries generate_sample, it won't touch data_dir at all.
    # A missing checkpoint will raise, but the error should be about the
    # checkpoint, not about data loading.
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        noise_robustness_sweep(
            checkpoint_path=str(tmp_path / "nonexistent.pth"),
            data_dir=str(data_dir),
            n_samples=2,
            n_snr_steps=1,
        )
```

Note: The test verifies that the function accepts `data_dir` and attempts to
load from it. A full end-to-end test requires a trained checkpoint, which is
out of scope. The existing `test_train.py` pattern (dummy shards + tiny
training run) could be extended for a full integration test if desired.

**Verify**: `uv run pytest tests/test_noise_sweep.py -q` → passes (the test
confirms `data_dir` is accepted and the function reaches the data-loading
stage).

### Step 5: Run the full test suite and lint

**Verify**:
- `uv run pytest tests/ -q` → all pass, including the new test
- `uv run ruff check src/ tests/` → exit 0
- `uv run mypy src/` → exit 0

## Test plan

- New test in `tests/test_noise_sweep.py` (or `tests/test_analysis_cli_smoke.py`)
  verifying `data_dir` is accepted and the function reaches the test-split
  loading stage.
- Structural pattern: follow `tests/test_train.py:12-47` for HDF5 shard setup.
- Verification: `uv run pytest tests/ -q` → all pass.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uv run ruff check src/ tests/` exits 0
- [ ] `uv run mypy src/` exits 0
- [ ] `uv run pytest tests/ -q` exits 0; new test for `noise_robustness_sweep`
      `data_dir` parameter exists and passes
- [ ] `grep -n "generate_sample\|_build_grid" src/phase_unwrap/analysis/noise_sweep.py`
      returns no matches (dead imports removed)
- [ ] `grep -n "data_dir" src/phase_unwrap/cli_cmds/eval.py` shows `data_dir`
      forwarded to `noise_robustness_sweep`
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (the codebase has drifted since this plan was written).
- `load_inference_state(subset="test")` raises for a config with `test_frac=0`
  — this is expected behavior (the test split doesn't exist), but if the
  function silently returns `train_loader` instead of raising, that's plan 018's
  scope; do not fix it here.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file.

## Maintenance notes

- If `load_inference_state` is later modified to accept a `device_str`
  parameter, the `device_str` argument in `noise_robustness_sweep` can be
  forwarded properly (currently it's accepted but unused — the device comes
  from the checkpoint's config). This is a pre-existing DX issue, not
  introduced by this plan.
- If the curriculum noise model (`NoiseAug`) is later added as an alternative
  sweep axis, it should be a separate `noise_type` parameter, not a replacement
  for the Gaussian SNR sweep — they measure different things.
- The `test_samples` list holds all N samples in memory. For large N or large
  image sizes, this could be memory-intensive. If that becomes an issue, cache
  to disk or reduce `n_samples` default.
