# Plan 024: Unpack `grad_phi2` Batch Tuple in Training & Evaluation Loops

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat ab82cfc..HEAD -- src/phase_unwrap/training/train.py src/phase_unwrap/data/dataset.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/023-integrate-pclcn-architecture-config-dispatch.md
- **Category**: bug
- **Planned at**: commit `ab82cfc`, 2026-07-30

## Why this matters

The dataset generator (`generate.py`) and HDF5 dataset loader (`H5ShardDataset` in `dataset.py`) return 3-tuples: `(I_raw, phi_gt, grad_phi2)`. However, `train.py` currently unpacks only 2 items (`I_raw, phi_gt = batch[0], batch[1]`) in both `run_eval()` and the training loop. Without forwarding `grad_phi2` to GPU device memory and into `model(I_raw, grad_phi2)`, `PCLCNModel` cannot access the analytical reference beam gradient prior required by `CNN 1`.

## Current state

- `src/phase_unwrap/data/dataset.py` — `H5ShardDataset.__getitem__` (lines 130–145) returns `(I, phi, grad_phi2)`.
- `src/phase_unwrap/training/train.py` — `run_eval()` (lines 66–68) and `train()` epoch loop (lines 296–298) unpack `batch[0], batch[1]` and ignore `batch[2]`.
- **Hardware Constraint**: Local laptop has no GPU (`device="cpu"`), whereas Kaggle runs on GPU (`device="cuda"`). `_unpack_batch()` must dynamically bind to whatever `torch.device` is active (`cpu` or `cuda`) without CUDA assumptions.

## Commands you will need

| Purpose   | Command | Expected on success |
|-----------|---------|---------------------|
| Unit Tests | `uv run pytest tests/test_train.py -q` | All pass |
| Full Test Suite | `uv run pytest tests/ -q` | All 57+ pass |

## Scope

**In scope**:
- `src/phase_unwrap/training/train.py` — Safely handle both 2-tuple `(I, phi)` and 3-tuple `(I, phi, grad_phi2)` batches during training and validation. Forward `grad_phi2` to device memory when present.

**Out of scope**:
- `src/phase_unwrap/data/generate.py` — HDF5 shard generation format is already complete and verified.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit message: `Unpack 3-tuple batches and pass grad_phi2 to PCLCN forward call`

## Steps

### Step 1: Update batch unpacking helper in `train.py`

In `src/phase_unwrap/training/train.py`, implement robust batch tuple unpacking that handles legacy 2-tuples and PCLCN 3-tuples:

```python
def _unpack_batch(batch: tuple, device: torch.device):
    I_raw = batch[0].to(device, non_blocking=True)
    phi_gt = batch[1].to(device, non_blocking=True)
    grad_phi2 = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
    return I_raw, phi_gt, grad_phi2
```

### Step 2: Update model execution logic for `PCLCNModel` vs `UNetRes2`

In `train.py` (both inside `run_eval()` and the training loop):

```python
if getattr(cfg.model, "arch", "unetres2").lower() == "pclcn":
    if grad_phi2 is None:
        raise ValueError("PCLCN model requires grad_phi2 in dataset batches.")
    phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)
    loss, loss_dict = loss_fn(phi_final, phi_gt, gx_tilde, gy_tilde)
else:
    phi_raw, k_off = model(I_input)
    phi_abs = phi_raw + k_off
    loss, loss_dict = loss_fn(phi_abs, phi_gt)
```

**Verify**: `uv run pytest tests/test_train.py -q` → passes.

## Test plan

- Test training step with 3-tuple dummy dataset in `tests/test_train.py`.
- Verification: `uv run pytest tests/test_train.py -q` → passes.

## Done criteria

- [ ] Batch unpacker handles 2-tuples and 3-tuples cleanly.
- [ ] `grad_phi2` is transferred to GPU and passed into `PCLCNModel`.
- [ ] `uv run pytest tests/ -q` passes without errors.

## STOP conditions

- If dataset shards fail to load `grad_phi2`, stop and inspect HDF5 keys in `dataset.py`.
