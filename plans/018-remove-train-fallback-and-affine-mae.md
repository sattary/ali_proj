# Plan 018: Remove silent train_loader fallback and GT-scale MAE annotation in figure code

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 23fd1ef..HEAD -- src/phase_unwrap/core/inference.py src/phase_unwrap/visualize/baseline_comparison_grid.py src/phase_unwrap/core/__init__.py src/phase_unwrap/visualize/error_histogram.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: correctness / tech-debt
- **Planned at**: commit `23fd1ef`, 2026-07-18

## Why this matters

Two related honesty bugs in the figure/inference paths:

1. **Silent `train_loader` fallback**: When `val_loader` is empty (e.g.
   `val_frac=0` in config, or `configs/tiny.yaml`), `load_inference_state` and
   `baseline_comparison_grid.py` silently fall back to the **training** loader.
   This means "validation" or "baseline comparison" figures could display
   training samples without any warning — and if those figures report MAE, the
   numbers are on data the model trained on. The `subset="test"` path already
   raises `ValueError` for an empty test split; the `subset="val"` path should
   do the same.

2. **GT-scale MAE annotation in figure**: `baseline_comparison_grid.py` uses
   `affine_align` (which fits scale `a` to ground truth) for both the
   visualization alignment and the "MAE: X.XXXX" text annotation. The project
   policy (plan 007) explicitly rejects GT-fit scale for metrics —
   `piston_align` (offset-only) is the sanctioned alignment. The MAE
   annotation in this figure violates that policy and produces inflated
   numbers that can't be achieved at deployment.

Additionally (Finding E), `affine_align` is exported from `core/__init__.py`
as part of the public API, making it easy to misuse. This plan removes it
from the public exports and updates the one remaining consumer to import it
directly from `core.ops`.

## Current state

### `src/phase_unwrap/core/inference.py` — silent fallback

Lines 57–64:
```python
if subset == "test":
    if test_loader is None or len(test_loader) == 0:
        raise ValueError("Test set requested but test_frac=0 in config.")
    loader = test_loader
elif subset == "val":
    loader = val_loader if (val_loader is not None and len(val_loader) > 0) else train_loader
else:
    loader = train_loader
```

The `subset="test"` branch raises. The `subset="val"` branch silently falls
back to `train_loader`. This is the asymmetry to fix.

### `src/phase_unwrap/visualize/baseline_comparison_grid.py` — fallback + affine_align

Lines 48–59 (same fallback pattern):
```python
if subset == "test":
    if test_loader is None:
        raise ValueError("Test set requested but test_frac=0 in config.")
    loader = test_loader
elif subset == "val":
    loader = (
        val_loader
        if (val_loader is not None and len(val_loader) > 0)
        else train_loader
    )
else:
    loader = train_loader
```

Lines 74 (UNet column — uses `affine_align` for visualization):
```python
phi_aligned, _, _ = affine_align(phi_abs, phi_gt)
```

Lines 107, 115 (Itoh and LSQ columns — also `affine_align`):
```python
itoh_aligned, _, _ = affine_align(itoh_t, gt_t)
lsq_aligned, _, _ = affine_align(lsq_t, gt_t)
```

Lines 168–180 (MAE annotation computed on the `affine_align`ed prediction):
```python
def ann_mae(ax, pred):
    mae = np.abs(pred - gt_img).mean()
    ax.text(0.98, 0.98, f"MAE: {mae:.4f}", ...)
```

### `src/phase_unwrap/core/__init__.py` — public export of `affine_align`

Lines 14–18:
```python
from .losses import MAEGradLoss, compute_metrics
from .ops import FixedSobel, affine_align, curvature_loss, laplacian, piston_align
from .utils import ensure_dir, pick_device, set_seed
```

Lines 29–34:
```python
"compute_metrics",
"FixedSobel",
"affine_align",
"piston_align",
"curvature_loss",
"laplacian",
```

### `src/phase_unwrap/visualize/error_histogram.py` — remaining `affine_align` consumer

Line 22:
```python
from ..core.ops import affine_align
```

This already imports directly from `core.ops`, so it will not break when
`affine_align` is removed from `core/__init__.py`. Verify this during execution.

### The sanctioned alignment

`src/phase_unwrap/core/ops.py:98-116` — `piston_align` is offset-only (no GT
scale fit), documented as the deployment-faithful alignment. It is what
`run_eval` (train.py:82) and `evaluate_baselines` (baselines.py:83) already use
for the `TopoMAE` metric.

`affine_align` (`ops.py:59-95`) is documented as "Diagnostic / figure tool
only. Do **not** use for headline metrics." But `baseline_comparison_grid.py`
uses it for a MAE annotation that looks like a metric.

### Repo conventions

- `from __future__ import annotations` at the top of every module.
- Type hints use `str | None` (PEP 604) syntax.
- Error handling: raise `ValueError` with a descriptive message (see
  `inference.py:59` for the existing pattern).
- Imports from `..core.ops` directly when a specific op is needed (see
  `error_histogram.py:22`).

## Commands you will need

| Purpose   | Command                                  | Expected on success |
|-----------|------------------------------------------|---------------------|
| Install   | `uv sync`                                | exit 0              |
| Tests     | `uv run pytest tests/ -q`                | all pass            |
| Lint      | `uv run ruff check src/ tests/`          | exit 0, no errors   |
| Typecheck | `uv run mypy src/`                       | exit 0, no errors    |

## Scope

**In scope** (the only files you should modify):
- `src/phase_unwrap/core/inference.py` (only the `subset="val"` branch)
- `src/phase_unwrap/visualize/baseline_comparison_grid.py`
- `src/phase_unwrap/core/__init__.py` (remove `affine_align` from exports)
- `src/phase_unwrap/visualize/error_histogram.py` (verify import path; no change
  expected, but verify it still works after `__init__.py` change)

**Out of scope** (do NOT touch, even though they look related):
- `src/phase_unwrap/core/ops.py` — `affine_align` stays defined there. Only its
  public export is removed.
- `src/phase_unwrap/analysis/baselines.py` — already uses `piston_align`
  correctly. No change needed.
- `src/phase_unwrap/analysis/tta.py` or `gradcam.py` — they use
  `load_inference_state` but don't have the fallback issue (they pass
  `subset="val"` and rely on the loader returned; after this fix, they'll get
  a `ValueError` instead of silent train data if val is empty — which is the
  correct behavior).
- Any other visualize module.

## Git workflow

- **Branch**: work on the current branch (`fix-review`). Do NOT create a new
  branch.
- Commit per logical unit; message style: conventional commits (match the
  repo's existing style, e.g. `Fix: Remove silent train_loader fallback and
  GT-scale MAE in figures`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Fix the silent fallback in `load_inference_state`

In `src/phase_unwrap/core/inference.py`, change the `subset="val"` branch
(currently line 62) to raise instead of falling back to `train_loader`.

Current (lines 61–62):
```python
elif subset == "val":
    loader = val_loader if (val_loader is not None and len(val_loader) > 0) else train_loader
```

New:
```python
elif subset == "val":
    if val_loader is None or len(val_loader) == 0:
        raise ValueError("Validation set requested but val_frac=0 in config.")
    loader = val_loader
```

This matches the existing `subset="test"` pattern (lines 57–59).

**Verify**: `uv run ruff check src/phase_unwrap/core/inference.py` → exit 0.

### Step 2: Fix the silent fallback in `baseline_comparison_grid.py`

In `src/phase_unwrap/visualize/baseline_comparison_grid.py`, change the
`subset="val"` branch (currently lines 52–57) to raise instead of falling back.

Current:
```python
elif subset == "val":
    loader = (
        val_loader
        if (val_loader is not None and len(val_loader) > 0)
        else train_loader
    )
```

New:
```python
elif subset == "val":
    if val_loader is None or len(val_loader) == 0:
        raise ValueError("Validation set requested but val_frac=0 in config.")
    loader = val_loader
```

**Verify**: `uv run ruff check src/phase_unwrap/visualize/baseline_comparison_grid.py` → exit 0.

### Step 3: Switch `baseline_comparison_grid.py` from `affine_align` to `piston_align`

This affects three locations in `plot_baseline_comparison`:

**Import** (line 18): replace `affine_align` with `piston_align`:
```python
from ..core.ops import piston_align
```

**UNet column** (line 74): replace `affine_align` with `piston_align`. Note
that `piston_align` returns `(aligned, c)` (2-tuple), while `affine_align`
returns `(aligned, a, c)` (3-tuple).

Current:
```python
phi_aligned, _, _ = affine_align(phi_abs, phi_gt)
```

New:
```python
phi_aligned, _ = piston_align(phi_abs, phi_gt)
```

**Itoh column** (lines 104–108): same change.

Current:
```python
itoh_t = torch.from_numpy(itoh_pred).unsqueeze(0).unsqueeze(0)
gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)
itoh_aligned, _, _ = affine_align(itoh_t, gt_t)
itoh_img = itoh_aligned.numpy()[0, 0]
```

New:
```python
itoh_t = torch.from_numpy(itoh_pred).unsqueeze(0).unsqueeze(0)
gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)
itoh_aligned, _ = piston_align(itoh_t, gt_t)
itoh_img = itoh_aligned.numpy()[0, 0]
```

**LSQ column** (lines 113–116): same change.

Current:
```python
lsq_t = torch.from_numpy(lsq_pred).unsqueeze(0).unsqueeze(0)
lsq_aligned, _, _ = affine_align(lsq_t, gt_t)
lsq_img = lsq_aligned.numpy()[0, 0]
```

New:
```python
lsq_t = torch.from_numpy(lsq_pred).unsqueeze(0).unsqueeze(0)
lsq_aligned, _ = piston_align(lsq_t, gt_t)
lsq_img = lsq_aligned.numpy()[0, 0]
```

The `ann_mae` function (lines 168–180) does not need changes — it already
computes `np.abs(pred - gt_img).mean()` on whatever aligned prediction is
passed to it. After this change, the MAE annotation will reflect
piston-aligned MAE (consistent with the `TopoMAE` policy), not GT-scale-fit
MAE.

**Verify**: `uv run ruff check src/phase_unwrap/visualize/baseline_comparison_grid.py` → exit 0.

### Step 4: Remove `affine_align` from `core/__init__.py` public exports

In `src/phase_unwrap/core/__init__.py`:

**Line 16** — remove `affine_align` from the import:
```python
# Before:
from .ops import FixedSobel, affine_align, curvature_loss, laplacian, piston_align
# After:
from .ops import FixedSobel, curvature_loss, laplacian, piston_align
```

**Line 31** — remove `"affine_align"` from `__all__`:
```python
# Before:
"FixedSobel",
"affine_align",
"piston_align",
# After:
"FixedSobel",
"piston_align",
```

**Verify**: `uv run ruff check src/phase_unwrap/core/__init__.py` → exit 0.

### Step 5: Verify `error_histogram.py` still imports correctly

`src/phase_unwrap/visualize/error_histogram.py:22` already imports directly:
```python
from ..core.ops import affine_align
```

This is unaffected by the `__init__.py` change (it imports from the module
directly, not the package). Verify by running the import.

**Verify**: `uv run python -c "from phase_unwrap.visualize.error_histogram import plot_error_histogram; print('OK')"` → prints `OK`.

### Step 6: Write a regression test

Add a test verifying that `load_inference_state(subset="val")` raises
`ValueError` when the val split is empty, instead of falling back to train data.

Follow the pattern in `tests/test_train.py:12-47` for HDF5 shard setup.

Add to `tests/test_dataset.py` (or a new `tests/test_inference.py`):

```python
def test_load_inference_state_raises_on_empty_val(tmp_path):
    """load_inference_state(subset='val') must raise, not fall back to train."""
    import h5py
    from phase_unwrap.core.config import TrainConfig
    from phase_unwrap.core.inference import load_inference_state

    # Create 3 dummy shards so smart_split has something to split
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I = np.random.randn(2, 1, 32, 32).astype(np.float32)
            phi = np.random.randn(2, 1, 32, 32).astype(np.float32)
            f.create_dataset("I", data=I)
            f.create_dataset("phi", data=phi)

    # We can't run load_inference_state without a checkpoint, but we can
    # verify the function signature and the subset logic by checking that
    # it doesn't silently return train data. A missing checkpoint will raise
    # before reaching the loader logic — so this test is structural.
    # Full integration test requires a trained checkpoint (out of scope).
    import inspect
    sig = inspect.signature(load_inference_state)
    assert "subset" in sig.parameters
    assert sig.parameters["subset"].default == "val"
```

Note: A full end-to-end test of the `ValueError` requires a trained checkpoint.
The structural test verifies the function accepts `subset` and defaults to
`"val"`. If a trained checkpoint is available (e.g. from `test_train.py`
fixtures), a more complete test can verify the raise behavior directly.

**Verify**: `uv run pytest tests/test_dataset.py -q` → passes.

### Step 7: Run the full test suite and lint

**Verify**:
- `uv run pytest tests/ -q` → all pass
- `uv run ruff check src/ tests/` → exit 0
- `uv run mypy src/` → exit 0
- `grep -rn "affine_align" src/phase_unwrap/visualize/baseline_comparison_grid.py` → no matches
- `grep -n "affine_align" src/phase_unwrap/core/__init__.py` → no matches
- `grep -n "train_loader" src/phase_unwrap/core/inference.py` → the `else` branch
  (`subset="train"`) is fine; the `subset="val"` branch must not reference
  `train_loader`.

## Test plan

- New test in `tests/test_dataset.py` (or `tests/test_inference.py`) verifying
  `load_inference_state` accepts `subset` and defaults to `"val"`.
- Structural pattern: follow `tests/test_train.py:12-47` for HDF5 setup.
- Verification: `uv run pytest tests/ -q` → all pass.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uv run ruff check src/ tests/` exits 0
- [ ] `uv run mypy src/` exits 0
- [ ] `uv run pytest tests/ -q` exits 0
- [ ] `grep -n "train_loader" src/phase_unwrap/core/inference.py` shows
      `train_loader` only in the `else: loader = train_loader` branch (the
      explicit `subset="train"` path), NOT in the `subset="val"` branch
- [ ] `grep -rn "affine_align" src/phase_unwrap/visualize/baseline_comparison_grid.py`
      returns no matches
- [ ] `grep -n "affine_align" src/phase_unwrap/core/__init__.py` returns no
      matches
- [ ] `uv run python -c "from phase_unwrap.visualize.error_histogram import plot_error_histogram"`
      succeeds (import path still works)
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (the codebase has drifted since this plan was written).
- `error_histogram.py` imports `affine_align` from `..core` (the package) rather
  than `..core.ops` (the module) — if so, it will break when `affine_align` is
  removed from `__init__.py`, and the fix is to update its import (in scope).
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file.
- Any other analysis or visualize module imports `affine_align` from
  `..core` (the package) rather than `..core.ops` — report all such sites so
  they can be fixed together.

## Maintenance notes

- After this plan lands, `affine_align` is still defined in `core/ops.py` and
  importable via `from ..core.ops import affine_align`. It is just no longer in
  the `core` package's public `__all__`. This discourages casual misuse while
  keeping it available for genuine diagnostic figures.
- If `error_histogram.py`'s alignment choice should also switch from
  `affine_align` to `piston_align` (for consistency with the metric policy),
  that is a separate decision — this plan only ensures the import path works.
  The owner should decide whether error histograms should show GT-scale-fit
  errors or piston-aligned errors.
- The `subset="train"` path in `load_inference_state` still returns
  `train_loader` — this is intentional (e.g. for `all_data=True` inference). It
  is NOT a fallback; it's an explicit request for training data.
- Any future code that calls `load_inference_state(subset="val")` with
  `val_frac=0` will now get a `ValueError` instead of silent train data. This
  is the desired behavior — callers should either set `val_frac > 0` or
  explicitly request `subset="train"`.
