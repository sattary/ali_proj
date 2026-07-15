# Plan 001: Establish a reproducible verification baseline (CI, test command, worker/RNG determinism)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- README.md src/phase_unwrap/core/utils.py src/phase_unwrap/data/dataset.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests / dx / repro
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

There is no automated signal that the code works, and the one documented test
command (`README.md`) references `tests/test_git_automation.py`, a module deleted
in commit `9968a53`. Meanwhile the training pipeline is **not reproducible on GPU**:
`pick_device` sets `torch.backends.cudnn.benchmark = True` (nondeterministic kernel
autotuning), `set_seed` never enables deterministic algorithms, and the DataLoader
has no `worker_init_fn`, so the `random`-driven geometric augmentation in
`H5ShardDataset.__getitem__` runs from an unseeded per-worker state. This means the
multi-seed "mean ± std" reported in the proposal conflates seed variance with kernel
and worker-RNG nondeterminism. This plan is finding #1: every other risky plan
(002–006) needs a green test suite and reproducible runs as its safety net.

## Current state

- `src/phase_unwrap/core/utils.py` — `set_seed` and `pick_device`:
  ```python
  def set_seed(seed: int = 1337) -> None:
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)

  def pick_device(device_str: str) -> torch.device:
      if torch.cuda.is_available():
          torch.backends.cudnn.benchmark = True   # <-- nondeterministic
      if device_str == "auto":
          return torch.device("cuda" if torch.cuda.is_available() else "cpu")
      return torch.device(device_str)
  ```
- `src/phase_unwrap/data/dataset.py` — `__getitem__` uses the global `random`
  module for flips/rot90 (lines ~80-90); `build_dataloaders` (lines ~147-231)
  constructs `DataLoader(...)` with **no** `worker_init_fn` and **no** `generator=`.
- `README.md` (lines ~750-761) documents:
  ```
  uv run pytest tests/test_git_automation.py -v
  ...
  31 tests covering data generation, model architecture, loss functions, ops, metrics, and git automation.
  ```
  `tests/test_git_automation.py` does not exist. Real suite: `tests/test_losses_ops.py`,
  `tests/test_generate.py`, `tests/test_model.py`, `tests/conftest.py`.
- No `.github/workflows/` directory exists.
- A stray empty `test.bak` (0 bytes) sits in the repo root.

Repo conventions: package under `src/phase_unwrap/`, `uv` for env management,
pytest for tests, Typer CLI. Match the existing test style in
`tests/test_losses_ops.py` (plain `def test_*` functions, `conftest.py` fixtures).

## Commands you will need

| Purpose   | Command                                   | Expected on success |
|-----------|-------------------------------------------|---------------------|
| Sync env  | `uv sync`                                 | exit 0 (installs torch; may be slow, several min) |
| Tests     | `uv run pytest tests/ -q`                 | all pass, exit 0    |
| One file  | `uv run pytest tests/test_model.py -q`    | pass                |
| Import    | `uv run python -c "import phase_unwrap"`  | exit 0              |

> NOTE: `uv sync` downloads torch 2.10 + CUDA wheels and can take several minutes
> and significant disk. If `uv sync` cannot complete in this environment, that is
> itself a STOP condition — report it; do not skip the test gate.

## Scope

**In scope**:
- `src/phase_unwrap/core/utils.py`
- `src/phase_unwrap/data/dataset.py`
- `tests/test_repro.py` (create)
- `README.md` (only the test-command block near lines 750-761)
- `.github/workflows/ci.yml` (create)
- delete `test.bak` (repo root, 0 bytes)

**Out of scope** (do NOT touch):
- `src/phase_unwrap/training/train.py` — determinism at call sites is plan 006's concern; here only fix the primitives in `utils.py`/`dataset.py`.
- The `.bak` files under `src/` — those are plan 008.
- Any change to augmentation *behavior* (the transforms themselves) — only their seeding.

## Git workflow

- Branch: `advisor/001-verification-baseline`
- Commit per step; message style matches repo (`git log` shows `Fix: ...` /
  imperative prefixes, e.g. `Fix: Add missing data-dir ... flags`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add opt-in determinism to `set_seed` and stop forcing `cudnn.benchmark`

In `src/phase_unwrap/core/utils.py`, change `set_seed` to accept a `deterministic`
flag and, when set, enable deterministic algorithms; change `pick_device` to not
unconditionally enable `cudnn.benchmark`.

Target shape:
```python
def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for reproducibility.

    When ``deterministic`` is True, also enables deterministic cuDNN/cuBLAS
    algorithms. This makes GPU runs reproducible at some throughput cost and
    may raise if an op lacks a deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
```
And in `pick_device`, delete the two lines that set `torch.backends.cudnn.benchmark = True`.
(`benchmark` is a throughput optimization that is nondeterministic; leaving it off by
default is the reproducible choice. A performance-minded run can still set it explicitly.)

Ensure `import os` is present at the top of the file.

**Verify**: `uv run python -c "from phase_unwrap.core.utils import set_seed; set_seed(1, deterministic=True); print('ok')"` → prints `ok`, exit 0.

### Step 2: Seed DataLoader workers

In `src/phase_unwrap/data/dataset.py`, add a module-level `worker_init_fn` and wire
it (plus a seeded `generator`) into every `DataLoader` built in `build_dataloaders`.

Target shape:
```python
def _seed_worker(worker_id: int) -> None:
    # Derive a per-worker seed from torch's base seed so each worker's
    # random/numpy streams are distinct and reproducible.
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
```
In `build_dataloaders`, create `g = torch.Generator(); g.manual_seed(seed)` and pass
`worker_init_fn=_seed_worker, generator=g` to the **train** `DataLoader` (the one with
`shuffle=True`). Add `worker_init_fn=_seed_worker` to the val/test loaders too (they
don't shuffle but their datasets still call `random` for augmentation — note val/test
augmentation is itself a separate concern; here only make it reproducible). Ensure
`import numpy as np` is available in the file (it already imports numpy).

**Verify**: `uv run python -c "from phase_unwrap.data.dataset import _seed_worker; print('ok')"` → `ok`.

### Step 3: Write a reproducibility characterization test

Create `tests/test_repro.py` modeled on `tests/test_losses_ops.py` structure. Cover:
- `set_seed(123)` then two draws of `torch.rand(4)` / `np.random.rand(4)` / `random.random()`
  reset by `set_seed(123)` again produce identical values.
- `set_seed(123, deterministic=True)` runs without raising and sets
  `torch.backends.cudnn.benchmark is False`.
- `_seed_worker(0)` and `_seed_worker(1)` produce different `random.random()` values
  (workers are distinct), and calling `_seed_worker(0)` twice reproduces the same value.

**Verify**: `uv run pytest tests/test_repro.py -q` → all pass.

### Step 4: Fix the README test block

In `README.md` (the block near lines 750-761): remove the
`uv run pytest tests/test_git_automation.py -v` line, replace the
`tests/test_model.py` example if you like but keep it valid, and change the
"31 tests covering ... and git automation" sentence to describe the real suite,
e.g. "Tests cover data generation, model architecture, loss functions, ops, and
metrics (`tests/test_generate.py`, `tests/test_model.py`, `tests/test_losses_ops.py`,
`tests/test_repro.py`)." Do not invent a test count — either omit it or compute it
from `uv run pytest tests/ --collect-only -q`.

**Verify**: `grep -n "git_automation" README.md` → no matches.

### Step 5: Add a minimal CI workflow

Create `.github/workflows/ci.yml` that installs `uv`, runs `uv sync`, and runs
`uv run pytest tests/ -q` on push/PR. Use the `astral-sh/setup-uv` action. Keep it to
a single Ubuntu job, Python 3.12 (matches `pyproject.toml` `requires-python = ">=3.12"`).
Mark the torch install as potentially slow but do not add GPU runners (tests are CPU-only).

**Verify**: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid yaml')"` → `valid yaml`.

### Step 6: Delete the stray empty `test.bak`

Confirm it is 0 bytes and unreferenced, then `git rm test.bak`.

**Verify**: `test ! -e test.bak && echo gone` → `gone`.

## Test plan

- New file `tests/test_repro.py` with the three cases in Step 3.
- Model structural pattern: `tests/test_losses_ops.py`.
- Verification: `uv run pytest tests/ -q` → all pass, including the new repro tests.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uv run pytest tests/ -q` exits 0; `tests/test_repro.py` exists and passes
- [ ] `grep -n "benchmark = True" src/phase_unwrap/core/utils.py` returns no matches
- [ ] `grep -n "worker_init_fn" src/phase_unwrap/data/dataset.py` returns ≥1 match
- [ ] `grep -n "git_automation" README.md` returns no matches
- [ ] `.github/workflows/ci.yml` exists and is valid YAML
- [ ] `test.bak` no longer exists
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `uv sync` cannot complete in this environment (torch unavailable) — the test
  gate cannot be established; report this so the operator can run it elsewhere.
- `torch.use_deterministic_algorithms(True, warn_only=True)` raises on import/setup
  (older torch) — report the torch version.
- The code at the locations in "Current state" doesn't match the excerpts.
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

- Enabling `deterministic=True` by default in `train.py` is deliberately **not** done
  here (it has a throughput cost); plan 006 decides call-site wiring. This plan only
  makes determinism *available* and makes workers seeded.
- A reviewer should confirm the CI torch install actually resolves on GitHub runners;
  if it times out, pin a CPU-only torch wheel index in the workflow.
