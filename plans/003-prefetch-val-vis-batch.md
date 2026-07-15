# Plan 003: Prefetch validation visualization batch

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/training/train.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-train-integration-tests.md
- **Category**: perf
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The validation loop dynamically extracts one batch from `train_loader` purely for visualization. Doing `next(iter(train_loader))` spawns and immediately destroys a full set of PyTorch multiprocessing dataloader workers. This injects seconds of process-spawning latency and memory thrashing into every validation step.

## Current state

- `src/phase_unwrap/training/train.py` — core training loop.
Lines 509-514:
```python
            try:
                # Get visualization data from train_loader to show actual noisy data
                I_raw_v, phi_gt_v = next(iter(train_loader))
                I_raw_v = I_raw_v.to(device, non_blocking=True)
                phi_gt_v = phi_gt_v.to(device, non_blocking=True)
```

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Tests     | `uv run pytest tests/test_train.py`| all pass            |

## Scope

**In scope**:
- `src/phase_unwrap/training/train.py`

## Steps

### Step 1: Pre-fetch visualization batch
In `src/phase_unwrap/training/train.py`, before the outer epoch loop begins (e.g., right before `for epoch in range(start_epoch, cfg.optim.epochs):`), add code to extract a static visualization batch:
```python
    vis_batch = None
    if train_loader is not None:
        vis_batch = next(iter(train_loader))
```

### Step 2: Use pre-fetched batch in validation
Replace the `next(iter(train_loader))` call in the validation block with the cached batch:
```python
            try:
                # Get visualization data from cached vis_batch
                if vis_batch is not None:
                    I_raw_v, phi_gt_v = vis_batch
                else:
                    I_raw_v, phi_gt_v = next(iter(train_loader))
```

**Verify**: `uv run pytest tests/test_train.py` → all pass.

## Test plan

- Rely on integration tests from Plan 001 to ensure the visualization pipeline still executes without failing on `vis_batch`.

## Done criteria

- [ ] `uv run pytest tests/test_train.py` exits 0
- [ ] No files outside the in-scope list are modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- `vis_batch` cannot be safely cached without memory leaks (should be fine as it's just two CPU tensors).
- Tests fail.
