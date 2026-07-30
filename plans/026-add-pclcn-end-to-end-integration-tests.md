# Plan 026: Add PCLCN End-to-End Integration Tests

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat ab82cfc..HEAD -- tests/test_train.py tests/test_model.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/023-integrate-pclcn-architecture-config-dispatch.md, plans/024-unpack-grad-phi2-batch-in-trainer.md
- **Category**: tests
- **Planned at**: commit `ab82cfc`, 2026-07-30

## Why this matters

Existing tests in `tests/test_model.py` and `tests/test_losses.py` only verify `PCLCNModel` layer output shapes and `PCLCNLoss` dictionary outputs in isolation. Zero integration tests exist verifying that a full epoch of training, validation evaluation, and checkpoint saving/loading works with `PCLCNModel` inside `train.py`. This plan adds comprehensive end-to-end tests for PCLCN execution.

## Current state

- `tests/test_train.py` — Currently tests smoke training loop using `UNetRes2` defaults only.
- `tests/test_model.py` — Has `test_pclcn_model_forward_shape` verifying layer shapes.

## Commands you will need

| Purpose   | Command | Expected on success |
|-----------|---------|---------------------|
| Unit Tests | `uv run pytest tests/test_pclcn_integration.py -v` | All pass |
| Full Test Suite | `uv run pytest tests/ -q` | All pass |

## Scope

**In scope**:
- `tests/test_pclcn_integration.py` (NEW) — Create end-to-end integration tests for PCLCN architecture running strictly on CPU (`device = torch.device("cpu")` and `use_amp = False`).

**Hardware Constraint & Execution Boundary**:
- Local laptop has no compatible GPU. All integration tests MUST run on CPU without requiring CUDA or throwing PyTorch CUDA capability warnings.

**Out of scope**:
- GPU-only CUDA kernel tests.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit message: `Add end-to-end integration tests for PCLCN training loop`

## Steps

### Step 1: Create `tests/test_pclcn_integration.py`

Create `tests/test_pclcn_integration.py` with the following test cases:

1. `test_pclcn_build_model()` — Verifies `build_model(ModelConfig(arch="pclcn"))` returns `PCLCNModel`.
2. `test_pclcn_train_epoch()` — Runs 1 training epoch on synthetic 3-tuple shards (`I_raw`, `phi_gt`, `grad_phi2`) using `PCLCNModel` and `PCLCNLoss`, asserting loss decreases and no NaNs occur.
3. `test_pclcn_checkpoint_roundtrip()` — Verifies `CheckpointManager.save()` and `load()` preserve `PCLCNModel` weights and optimizer states.

**Verify**: `uv run pytest tests/test_pclcn_integration.py -v` → All 3 tests pass.

## Done criteria

- [ ] `tests/test_pclcn_integration.py` exists with 3 passing integration test functions.
- [ ] PCLCN forward/backward pass, loss computation, and checkpointing verified end-to-end.
- [ ] `uv run pytest tests/ -q` passes without errors.

## STOP conditions

- If NaN loss occurs during PCLCN forward/backward pass, stop and inspect `ComplexDomainLoss` and gradient scaling.
