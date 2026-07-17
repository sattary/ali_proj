# Plan 006: Allow zero-length validation and test splits

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/data/dataset.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-train-integration-tests.md
- **Category**: bug
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The dataset split logic explicitly forces a minimum count of 1 for validation and test splits, even if the user configures a fraction of 0.0. This silently steals shards away from the training pool, preventing users from utilizing 100% of their dataset for training.

## Current state

- `src/phase_unwrap/data/dataset.py` — shard splitting logic.
Lines 123-124:
```python
    val_count = max(1, int(round(val_frac * n)))
    test_count = max(1, int(round(test_frac * n)))
```

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Tests     | `uv run pytest tests/test_dataset.py`| all pass            |

## Scope

**In scope**:
- `src/phase_unwrap/data/dataset.py`

## Steps

### Step 1: Conditionally apply max(1, ...)
Modify lines 123-124 to only enforce a minimum of 1 if the fraction is greater than 0:
```python
    val_count = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    test_count = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
```

### Step 2: Handle edge cases in fallback
Lines 126-132 contain a fallback for very small datasets. Ensure this fallback gracefully handles `val_count == 0` and `test_count == 0` appropriately. (e.g. if `val_frac == 0`, do not forcefully allocate `n // 3` to `val_count`).

**Verify**: `uv run pytest tests/test_dataset.py` → all pass.

## Test plan

- Add a test case in `tests/test_dataset.py` (created in Plan 001) calling `get_shard_splits` with `val_frac=0.0` and `test_frac=0.0`, asserting that `val_paths` and `test_paths` are empty.

## Done criteria

- [x] `uv run pytest tests/test_dataset.py` exits 0; zero-split tests pass
- [x] No files outside the in-scope list are modified
- [x] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- Modifying the split logic causes division-by-zero errors downstream in the `train.py` when it attempts to initialize `val_loader`. (Note: `train.py` already checks `if val_loader is not None:`, so it should be fine).
