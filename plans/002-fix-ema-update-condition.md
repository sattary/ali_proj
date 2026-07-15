# Plan 002: Fix EMA update inside optimizer step check

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
- **Category**: bug
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

When PyTorch AMP catches NaN/Inf gradients, it skips the optimizer step, but currently the EMA update still executes. This drags the EMA weights incorrectly toward the un-updated model during gradient spikes, breaking the exponential moving average invariant.

## Current state

- `src/phase_unwrap/training/train.py` — core training loop.
Lines 415-421:
```python
                # Only step the scheduler if the scaler didn't reduce the scale
                # (which indicates it skipped the opt.step due to nan/inf grads).
                if scale_after >= scale_before:
                    sched.step()
                    global_step += 1

                ema.update(model)
```

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Tests     | `uv run pytest tests/test_train.py`| all pass            |

## Scope

**In scope**:
- `src/phase_unwrap/training/train.py`

**Out of scope**:
- Any other training logic.

## Steps

### Step 1: Move EMA update inside the conditional
In `src/phase_unwrap/training/train.py`, indent `ema.update(model)` so it sits inside the `if scale_after >= scale_before:` conditional block alongside `sched.step()`.

```python
                if scale_after >= scale_before:
                    sched.step()
                    global_step += 1
                    ema.update(model)
```

**Verify**: `uv run pytest tests/test_train.py` → all pass.

## Test plan

- Rely on the integration tests from Plan 001 to ensure the training loop still runs without syntax or indentation errors.

## Done criteria

- [ ] `uv run pytest tests/test_train.py` exits 0
- [ ] No files outside the in-scope list are modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- The code at the locations in "Current state" doesn't match the excerpts.
- Tests fail.

## Maintenance notes
Reviewers should just confirm the indentation change.
