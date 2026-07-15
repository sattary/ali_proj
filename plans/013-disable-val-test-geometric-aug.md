# Plan 013: Disable geometric augmentation on val/test (honor DataConfig.augment)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/data/dataset.py src/phase_unwrap/core/config.py src/phase_unwrap/core/inference.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P0
- **Effort**: S
- **Risk**: MED (reported val/test metrics will change vs historical runs)
- **Depends on**: none (can land before train integration tests; add a unit test in this plan)
- **Category**: bug / methodology
- **Planned at**: commit `967403f`, 2026-07-15

## Why this matters

`H5ShardDataset.__getitem__` always applies random horizontal/vertical flips and
`rot90` with no train/val flag. Validation and test loaders share the same class, so
every epoch’s “held-out” metrics are stochastic geometric transforms of the data.
`DataConfig.augment` exists and `load_inference_state` sets `cfg.data.augment = False`,
but **nothing reads that flag**. Paper-facing metrics and checkpoint selection are
therefore not fixed-orientation evaluations. This must land before re-running honest
metrics (plans 007/010) or the numbers remain irreproducible.

## Current state

- `src/phase_unwrap/data/dataset.py:29-34` — constructor has no `augment` argument:

  ```python
  def __init__(
      self,
      shard_paths: Sequence[str],
      I_key: str = "I",
      phi_key: str = "phi",
  ) -> None:
  ```

- `src/phase_unwrap/data/dataset.py:80-92` — geometric aug always runs:

  ```python
  # Phase topology geometric augmentation
  if random.random() > 0.5:
      I_raw_t = I_raw_t.flip(-1)
      ...
  k = random.randint(0, 3)
  if k > 0:
      I_raw_t = torch.rot90(I_raw_t, k, dims=(-2, -1))
      phi_gt_t = torch.rot90(phi_gt_t, k, dims=(-2, -1))
  return I_raw_t, phi_gt_t
  ```

- `src/phase_unwrap/data/dataset.py:172-189` — train/val/test all construct
  `H5ShardDataset(...)` with no augment flag.

- `src/phase_unwrap/core/config.py:34` — `augment: bool = True` on `DataConfig`.

- `src/phase_unwrap/core/inference.py:35` — sets `cfg.data.augment = False` (currently a no-op).

**Conventions**: Match existing dataset style (type hints, short docstrings on why).
Return type annotation on `__getitem__` currently says four tensors but returns two —
do **not** “fix” that annotation in this plan (out of scope); leave as-is unless your
edit forces a touch, then set it to the true two-tensor return.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Unit tests | `uv run pytest tests/test_dataset_augment.py -q` | all pass |
| Full suite | `uv run pytest tests/ -q` | all pass |
| Grep flag wired | `rg -n "augment" src/phase_unwrap/data/dataset.py` | constructor + `__getitem__` + build_dataloaders |

## Scope

**In scope**:
- `src/phase_unwrap/data/dataset.py`
- `tests/test_dataset_augment.py` (create)

**Out of scope**:
- Curriculum/optical noise (`NoiseAug` / `prepare_batch`) — different subsystem.
- Plan 009 hint leakage.
- Changing default `DataConfig.augment` value (keep `True` for training).
- `visualize/` modules.
- Return-type cleanup of `__getitem__` unless required by typing tools.

## Git workflow

- Branch: `advisor/013-disable-val-test-geometric-aug`
- Commit style (from `git log`): `Fix: ...` prefixes. Example:
  `Fix: honor DataConfig.augment and disable geometric aug on val/test`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add `augment: bool` to `H5ShardDataset`

In `src/phase_unwrap/data/dataset.py`:

1. Add constructor arg `augment: bool = True` and store `self.augment = augment`.
2. Wrap the geometric block (lines 80–90) in `if self.augment:`.

**Verify**: `rg -n "self\.augment" src/phase_unwrap/data/dataset.py` → at least two hits
(constructor assign + guard).

### Step 2: Wire `build_dataloaders`

In `build_dataloaders` in the same file:

- Train dataset: `augment=cfg.data.augment` (so global disable still works).
- Val dataset: `augment=False` always.
- Test dataset: `augment=False` always.

```python
train_ds = H5ShardDataset(
    train_paths,
    I_key=cfg.data.I_key,
    phi_key=cfg.data.phi_key,
    augment=cfg.data.augment,
)
val_ds = (
    H5ShardDataset(
        val_paths,
        I_key=cfg.data.I_key,
        phi_key=cfg.data.phi_key,
        augment=False,
    )
    if val_paths
    else None
)
# same pattern for test_ds with augment=False
```

**Verify**: `rg -n "augment=" src/phase_unwrap/data/dataset.py` → train uses
`cfg.data.augment`; val/test use `False`.

### Step 3: Unit tests

Create `tests/test_dataset_augment.py` modeled after `tests/test_generate.py`
(simple classes, no fixtures required beyond temp files if needed).

Minimum cases:

1. **With augment=False, orientation is deterministic**: build a tiny in-memory path
   is hard without HDF5; instead unit-test by constructing a minimal temporary `.h5`
   shard (see `tests/test_generate.py::TestGenerateToH5` for HDF5 write pattern) with
   one sample that is **not** rotationally symmetric (e.g. a ramp). Call `__getitem__(0)`
   twice with `augment=False` and assert tensors are equal.

2. **With augment=True, transforms can change data**: call `__getitem__(0)` many times
   (e.g. 40) with `augment=True` and assert **not all** outputs are equal (probabilistic;
   if all equal, fail). Seed is not required to prove “sometimes transforms.”

3. **build_dataloaders wiring** (optional if heavy): if creating shards is expensive,
   cases 1–2 on `H5ShardDataset` alone are enough; still grep-verify step 2.

**Verify**: `uv run pytest tests/test_dataset_augment.py -q` → pass.

### Step 4: Full suite

**Verify**: `uv run pytest tests/ -q` → all pass.

## Test plan

- New file: `tests/test_dataset_augment.py`.
- Pattern: `tests/test_generate.py` (temp dirs, h5py, assert shapes/equality).
- Regression: val/test path cannot apply random flips when `augment=False`.

## Done criteria

- [ ] `H5ShardDataset` accepts `augment` and guards geometric transforms
- [ ] Val/test loaders always `augment=False`; train uses `cfg.data.augment`
- [ ] `uv run pytest tests/test_dataset_augment.py -q` passes
- [ ] `uv run pytest tests/ -q` passes
- [ ] No files outside scope modified (`git status`)
- [ ] `plans/README.md` status row set to DONE

## STOP conditions

- Excerpts no longer match live `dataset.py`.
- No way to create a temp HDF5 shard in tests (report; do not mock the entire Dataset away).
- You believe train geometric aug must also be removed — that is **out of scope**; report, do not change train default.

## Maintenance notes

- After this lands, any historical `metrics.csv` val numbers are **not** comparable 1:1
  (they included random D4). Re-eval or retrain for paper tables.
- Reviewer: confirm val/test never pass `augment=True`.
- Follow-up: plans 007/010 assume fixed eval orientation; run them after this.
- Curriculum optical noise is independent and stays on the train path via `prepare_batch`.
