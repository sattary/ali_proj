# Plan 023: Integrate PCLCN Architecture Config & Model Dispatch

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat ab82cfc..HEAD -- src/phase_unwrap/core/config.py src/phase_unwrap/model/unet.py src/phase_unwrap/training/train.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/022-pclcn-optics-express.md
- **Category**: bug
- **Planned at**: commit `ab82cfc`, 2026-07-30

## Why this matters

Although `PCLCNModel` and `PCLCNLoss` were implemented in `unet.py` and `losses.py` for Plan 022, they are completely disconnected from the runtime configuration and training engine. Currently, `ModelConfig` lacks an architecture selector (`arch`), `build_model()` hardcodes `UNetRes2`, and `train.py` hardcodes `MAEGradLoss`. Consequently, `phase-unwrap train` physically trains the old baseline model. This plan adds schema support, dynamic model/loss dispatch, and CLI selection for PCLCN.

## Current state

- `src/phase_unwrap/core/config.py` — Defines `ModelConfig` (lines 39–50) and `LossConfig` (lines 66–75); currently missing `arch` and PCLCN loss weight fields (`w_curl`, `w_phase_complex`).
- `src/phase_unwrap/model/unet.py` — Contains `build_model()` (lines 380–424) which instantiates `UNetRes2` regardless of config settings. `PCLCNModel` exists at lines 350–423.
- `src/phase_unwrap/training/train.py` — Hardcodes `loss_fn = MAEGradLoss(...)` (lines 184–188) and `phi_raw, k_off = model(I_input)` (lines 80, 313).
- **Repo convention & Hardware constraint**: Dataclasses loaded via `TrainConfig.from_dict()` or YAML files. Local test execution runs on CPU (`device="cpu"`), while Kaggle GPU execution uses `device="cuda"`. `pick_device("auto")` MUST resolve to CPU locally without throwing CUDA capability errors.

## Commands you will need

| Purpose   | Command | Expected on success |
|-----------|---------|---------------------|
| Unit Tests | `uv run pytest tests/ -q` | 57+ passed |
| Typecheck | `uv run mypy src/phase_unwrap/core/config.py src/phase_unwrap/model/unet.py` | exit 0, no errors |
| Lint | `uv run ruff check src/phase_unwrap/core/config.py src/phase_unwrap/model/unet.py` | exit 0, no errors |

## Scope

**In scope**:
- `src/phase_unwrap/core/config.py` — Add `arch: str = "unetres2"` to `ModelConfig` and PCLCN loss weights to `LossConfig`.
- `src/phase_unwrap/model/unet.py` — Update `build_model(cfg: ModelConfig)` to construct `PCLCNModel` when `cfg.arch == "pclcn"`.
- `src/phase_unwrap/training/train.py` — Dispatch loss function (`PCLCNLoss` vs `MAEGradLoss`) based on configuration.

**Out of scope**:
- `src/phase_unwrap/core/ops.py` — Signal processing math is tested and stable; do not alter.
- Public dataset layout or HDF5 shard structure.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit per step with imperative message: e.g. `Add arch selection to ModelConfig and update build_model dispatch`

## Steps

### Step 1: Update `ModelConfig` and `LossConfig` in `config.py`

In `src/phase_unwrap/core/config.py`, update `ModelConfig` to include `arch: str = "unetres2"` (valid values: `"unetres2"`, `"pclcn"`). Update `LossConfig` to include PCLCN weights: `w_curl: float = 0.1` and `w_complex: float = 1.0`.

```python
@dataclass
class ModelConfig:
    arch: str = "unetres2"  # "unetres2" or "pclcn"
    base: int = 32
    activation: str = "silu"
    final_dropout: float = 0.3
    ema_decay: float = 0.999
    device: str = "auto"
    use_amp: bool = False
    use_coordconv: bool = True
```

**Verify**: `uv run mypy src/phase_unwrap/core/config.py` → Success: no issues found.

### Step 2: Implement dynamic dispatch in `build_model()` in `unet.py`

In `src/phase_unwrap/model/unet.py`, update `build_model(cfg: ModelConfig) -> nn.Module`:

```python
def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch.lower() == "pclcn":
        return PCLCNModel(base=cfg.base)
    return UNetRes2(
        in_ch=1,
        base=cfg.base,
        act_name=cfg.activation,
        final_dropout=cfg.final_dropout,
        use_coordconv=cfg.use_coordconv,
    )
```

**Verify**: `uv run pytest tests/test_model.py -q` → All tests pass.

### Step 3: Wire loss function dispatch in `train.py`

In `src/phase_unwrap/training/train.py`, update model initialization and loss function construction:

```python
from ..core.losses import MAEGradLoss, PCLCNLoss, compute_metrics

if cfg.model.arch.lower() == "pclcn":
    loss_fn = PCLCNLoss(w_phase=cfg.loss.w_mae, w_curl=cfg.loss.w_curl)
else:
    loss_fn = MAEGradLoss(
        w_mae=cfg.loss.w_mae,
        w_grad=cfg.loss.w_grad,
        intensity_weighted=cfg.loss.int_wgrad,
    )
```

**Verify**: `uv run pytest tests/test_train.py -q` → All tests pass.

## Test plan

- Add unit test in `tests/test_model.py` verifying that `build_model(ModelConfig(arch="pclcn"))` returns an instance of `PCLCNModel`.
- Verification: `uv run pytest tests/test_model.py -q` → passes.

## Done criteria

- [ ] `ModelConfig` accepts `arch="pclcn"`.
- [ ] `build_model()` instantiates `PCLCNModel` when requested.
- [ ] `train.py` constructs `PCLCNLoss` for `pclcn` architecture.
- [ ] `uv run pytest tests/ -q` passes without errors.

## STOP conditions

- If `PCLCNModel` constructor signature in `unet.py` mismatches `ModelConfig` fields, stop and report.
- If existing baseline `UNetRes2` tests fail after changes, stop and fix regression.

## Maintenance notes

- Future additions of neural architectures must update `ModelConfig.arch` validation and `build_model` dispatch.
