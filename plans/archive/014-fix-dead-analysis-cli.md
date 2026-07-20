# Plan 014: Repair dead GradCAM, TTA, TorchScript, and benchmark CLI paths

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/analysis/gradcam.py src/phase_unwrap/analysis/tta.py src/phase_unwrap/analysis/export.py src/phase_unwrap/core/inference.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none (works without 013, but eval metrics still inherit other methodology issues)
- **Category**: bug
- **Planned at**: commit `967403f`, 2026-07-15

## Why this matters

README and CLI advertise GradCAM plots, TTA evaluation, TorchScript export, and
inference benchmarks. Several entry points crash with `NameError` or wrong batch
unpacking before producing output. Paper tooling and cloud notebooks that call these
commands fail silently from a user perspective. Fix the crash paths so advertised
commands run; do **not** redesign metrics or hint policy here.

## Current state

### GradCAM — undefined symbols

`src/phase_unwrap/analysis/gradcam.py:83-90`:

```python
from ..core.inference import load_inference_state
model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

I_input, phi_gt, I_raw, _ = next(iter(loader))  # loader undefined
...
cam, phi_abs = cam_extractor(I_input)  # cam_extractor undefined
```

Also: `data_dir` argument is ignored (`data_dir=''`); dataset returns **two** tensors
`(I_raw, phi_gt)` not four (`dataset.py:92`).

### TTA eval — missing import + wrong unpack

`src/phase_unwrap/analysis/tta.py:90-95,126`:

```python
model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

train_loader, val_loader, test_loader = build_dataloaders(  # not imported; also redundant
    cfg, device, seed=cfg.logging.seed
)
...
for I_input, phi_gt, _, _ in loader:  # loader yields (I_raw, phi_gt) only
```

`load_inference_state` already returns `(model, cfg, loader, device)` with the correct
subset (`src/phase_unwrap/core/inference.py:25-64`). Prefer that loader; do not rebuild
unless subset selection is broken.

### TorchScript + benchmark — missing imports

`src/phase_unwrap/analysis/export.py` top-level imports only `time`, `Path`, `numpy`,
`torch`. But:

```python
# export_torchscript ~58-60
cfg = load_train_config(...)   # NameError
model = build_model(cfg.model) # NameError

# benchmark_inference ~94-97
cfg = load_train_config(...)
dev = pick_device(device)
model = build_model(cfg.model).to(dev)
```

`export_onnx` works because it local-imports `load_inference_state`.

**Conventions**: Prefer local imports inside functions that already use
`load_inference_state` (see `export_onnx`). Match existing `prepare_batch` usage in
visualize modules for turning `(I_raw, phi_gt)` into model input.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Syntax/import smoke | `uv run python -c "from phase_unwrap.analysis.export import export_torchscript, benchmark_inference; from phase_unwrap.analysis.gradcam import plot_gradcam; from phase_unwrap.analysis.tta import evaluate_tta"` | exit 0 |
| Full tests | `uv run pytest tests/ -q` | all pass |
| Optional runtime | only if a small checkpoint + data exist in the workspace | see steps |

## Scope

**In scope**:
- `src/phase_unwrap/analysis/gradcam.py`
- `src/phase_unwrap/analysis/tta.py`
- `src/phase_unwrap/analysis/export.py`
- `tests/test_analysis_cli_smoke.py` (create; import/structure smoke only)

**Out of scope**:
- Changing `affine_align` metric policy (plans 007/010).
- Changing `phi_hint` construction policy (plan 009) beyond what is required to
  build a valid 2-channel input for GradCAM/TTA (use existing `prepare_batch`).
- README rewrites (plan 011).
- Making GradCAM numerically “correct” for paper claims — only make it **run**.

## Git workflow

- Branch: `advisor/014-fix-dead-analysis-cli`
- Commit example: `Fix: repair GradCAM, TTA, TorchScript, and benchmark entrypoints`
- Do NOT push unless instructed.

## Steps

### Step 1: Fix `plot_gradcam`

In `src/phase_unwrap/analysis/gradcam.py` `plot_gradcam`:

1. Call:
   ```python
   model, cfg, loader, device = load_inference_state(
       checkpoint_path, data_dir=data_dir, config_path=config_path, subset=subset
   )
   ```
   (Pass through `subset` if `load_inference_state` supports it — it does at
   `core/inference.py` signature; check live signature and use it.)

2. Resolve target layer:
   ```python
   if not hasattr(model, target_layer_name):
       raise ValueError(f"Unknown layer {target_layer_name}")
   target_layer = getattr(model, target_layer_name)
   # If layer is Sequential, GradCAM hooks on the module itself are OK for enc*
   cam_extractor = _GradCAM(model, target_layer if not hasattr(target_layer, '__iter__') else target_layer)
   ```
   Simpler acceptable approach: if `getattr(model, name)` is `nn.Sequential`, hook the
   **last** submodule: `list(target_layer.children())[-1]` when children exist, else the
   module itself. Document the choice in a one-line comment.

3. Fetch batch:
   ```python
   from ..data.augmentation import prepare_batch
   I_raw, phi_gt = next(iter(loader))
   I_raw = I_raw[:n_samples]
   phi_gt = phi_gt[:n_samples]
   I_input, phi_gt, I_raw_n, I_raw_c = prepare_batch(I_raw.to(device), phi_gt.to(device), noise_aug=None)
   I_input = I_input.requires_grad_(True)
   ```
   Use `I_raw_c` (clean) or `I_raw_n` for display — prefer clean for visualization.

4. `cam, phi_abs = cam_extractor(I_input)` then keep existing plotting code, adapting
   numpy conversion if tensor is already on GPU (`.detach().cpu()`).

**Verify**: `uv run python -c "import ast; ast.parse(open('src/phase_unwrap/analysis/gradcam.py').read())"` → exit 0  
and `rg -n "loader|cam_extractor" src/phase_unwrap/analysis/gradcam.py` shows definitions before use.

### Step 2: Fix `evaluate_tta`

In `src/phase_unwrap/analysis/tta.py` `evaluate_tta`:

1. Use the loader from `load_inference_state` with the real `data_dir` and `subset`:
   ```python
   model, cfg, loader, device = load_inference_state(
       checkpoint_path,
       data_dir=data_dir,
       config_path=config_path,
       subset=subset,
       all_data=all_data,
   )
   ```
   Delete the broken `build_dataloaders` call unless you need all three splits — you do not.

2. Loop:
   ```python
   from ..data.augmentation import prepare_batch
   for I_raw, phi_gt in loader:
       I_input, phi_gt, _, _ = prepare_batch(I_raw.to(device), phi_gt.to(device), noise_aug=None)
       ...
   ```
   Keep existing MAE / affine_align comparison logic after you have `phi_noaug` / TTA preds.

**Verify**: `rg -n "build_dataloaders" src/phase_unwrap/analysis/tta.py` → no matches  
and `rg -n "for .* in loader" src/phase_unwrap/analysis/tta.py` → unpacks two values from dataset path.

### Step 3: Fix `export_torchscript` and `benchmark_inference`

In `src/phase_unwrap/analysis/export.py`, either:

**Preferred (minimal):** add local imports inside each function:

```python
from ..core.config import load_train_config
from ..core.utils import pick_device  # verify pick_device lives in core/utils.py
from ..model.unet import build_model
```

Confirm real symbols before importing:
- `rg -n "def load_train_config|def pick_device|def build_model" src/phase_unwrap`

Alternatively route both through `load_inference_state` like `export_onnx` (also fine).

**Verify**:
```bash
uv run python -c "from phase_unwrap.analysis.export import export_torchscript, benchmark_inference, export_onnx"
```
→ exit 0

### Step 4: Smoke tests

Create `tests/test_analysis_cli_smoke.py`:

- Import `plot_gradcam`, `evaluate_tta`, `export_torchscript`, `benchmark_inference`.
- Optionally `inspect.signature` checks that `plot_gradcam` parameters still include
  `checkpoint_path`, `data_dir`.
- Do **not** require a real checkpoint unless one is guaranteed present; keep tests offline.

**Verify**: `uv run pytest tests/test_analysis_cli_smoke.py -q` → pass  
`uv run pytest tests/ -q` → pass

## Test plan

- Smoke imports only (no GPU, no checkpoint).
- Pattern: lightweight tests like `tests/test_model.py` structure if needed; imports-only is OK.

## Done criteria

- [ ] `plot_gradcam` defines `loader` and `cam_extractor` before use; passes `data_dir`
- [ ] `evaluate_tta` uses `load_inference_state` loader + `prepare_batch`; no bare `build_dataloaders`
- [ ] `export_torchscript` / `benchmark_inference` import all symbols they use
- [ ] `uv run pytest tests/ -q` green
- [ ] No files outside scope modified
- [ ] `plans/README.md` status updated

## STOP conditions

- `load_inference_state` signature does not accept `subset` / `all_data` as assumed —
  open `core/inference.py` and adapt; do not invent a parallel loader stack.
- GradCAM hooks fail on Sequential modules after a reasonable hook-target choice —
  report with the error; do not rewrite the whole model.
- Fix appears to require metric policy changes — defer to 007/009.

## Maintenance notes

- After plan 009 changes hint construction, GradCAM/TTA will automatically pick up
  `prepare_batch` behavior — good.
- After plan 007, TTA MAE still uses `affine_align` in `tta.py`; leave it unless 007
  explicitly expands scope (currently it does not). Note in review if paper cites TTA MAE.
- Reviewer: run one manual GradCAM/TTA command when a checkpoint + data exist.
