# Plan 001: Write integration tests for core training loop

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- tests/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The critical paths of the codebase—including the dynamic CUDA OOM recovery loop, HDF5 sharding splits, and training logic—have zero automated verification. Adding an end-to-end integration test running `train.py` and verifying `dataset.py` logic prevents silent regressions during upcoming performance refactoring.

## Current state

- `tests/test_model.py` — an existing test file you should pattern-match for PyTorch test setups.
- `tests/test_dataset.py` — does not exist.
- `tests/test_train.py` — does not exist.

Conventions:
Use `pytest`. Match the style in `tests/test_model.py`. Use temporary directories via `pytest`'s `tmp_path` fixture for generating dummy HDF5 data if needed.

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Tests     | `uv run pytest tests/ -q`| all pass            |

## Scope

**In scope** (the only files you should modify):
- `tests/test_dataset.py` (create)
- `tests/test_train.py` (create)

**Out of scope**:
- Modifying any application code in `src/`.

## Steps

### Step 1: Create `tests/test_dataset.py`
Create a test file for `src/phase_unwrap/data/dataset.py`.
Write tests that:
- Generate a minimal HDF5 file with `h5py` in a `tmp_path`.
- Initialize `H5ShardDataset` pointing to it.
- Assert `__len__` and `__getitem__` work correctly without errors.

**Verify**: `uv run pytest tests/test_dataset.py` → 1+ tests pass.

### Step 2: Create `tests/test_train.py`
Create a test file for `src/phase_unwrap/training/train.py`.
Write tests that:
- Mock or setup a minimal `TrainConfig`.
- Create a minimal dummy dataset to feed into the training loop.
- Invoke `train_cmd` or the underlying `train()` function for 1 epoch to ensure it doesn't crash.

**Verify**: `uv run pytest tests/test_train.py` → 1+ tests pass.

## Test plan

This plan *is* the test plan. You are implementing the tests.

## Done criteria

- [ ] `uv run pytest tests/test_dataset.py` exits 0; new tests pass
- [ ] `uv run pytest tests/test_train.py` exits 0; new tests pass
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:
- `train()` requires dependencies that cannot be easily mocked or bypassed.
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

These integration tests serve as a verification baseline for the upcoming `dataset.py` HDF5 I/O optimizations and `train.py` bug fixes. Reviewers should ensure the tests don't take too long (use minimal dimensions/epochs).
