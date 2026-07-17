# Deep Audit Findings

This document contains the consolidated and prioritized findings from the `/improve deep` codebase audit across 9 categories. Findings are ordered by leverage (impact ÷ effort) with security and validity-blocking issues floated to the top.

## Priority 1: Critical Security & Scientific Validity

### [SECURITY-01] Leaked Kaggle Proxy JWT Token
- **Evidence**: `notebooks/run_on_cloud.ipynb:1104` — contains a raw JWT token leaked in the committed notebook's output block.
- **Impact**: Anyone with read access to the repo could potentially hijack the Kaggle Jupyter session or steal the proxy identity.
- **Effort**: S (hours)
- **Risk**: LOW (clearing notebook outputs does not affect code execution)
- **Confidence**: HIGH
- **Fix sketch**: Clear all cell outputs from `run_on_cloud.ipynb`, commit the clean notebook, and recommend rotating the compromised Kaggle API/proxy token.

### [DIR-03] Baselines evaluated on fresh clean data instead of the noisy test set
- **Evidence**: `src/phase_unwrap/analysis/baselines.py:62` and `124` — both `evaluate_baselines` and `evaluate_dl_baseline` ignore `--data-dir` and instead call `generate_sample()` directly in a loop to generate fresh, zero-noise synthetic grids on the fly.
- **Impact**: Unfair/invalid evaluation methodology. The DL model is trained to be robust to curriculum noise (`AugmentationConfig`), but the baseline comparison evaluates it against classical methods (Itoh/skimage) on perfectly clean, un-augmented samples rather than the actual hold-out HDF5 test set.
- **Effort**: M (a day-ish)
- **Risk**: MED — it will likely significantly degrade the reported classical baseline metrics, changing the claims made about relative performance.
- **Confidence**: HIGH
- **Fix sketch**: Remove the `generate_sample` loops. Run `_unwrap_skimage`, `_unwrap_itoh`, and the DL model directly on batches loaded from the `--data-dir` test split.

### [PERF-01] N+1 I/O Amplification in Dataloader (w/ Naive Cache Eviction)
- **Evidence**: `src/phase_unwrap/data/dataset.py:93` — `H5ShardDataset` maintains a tiny 4-block cache of size 128 items and naive `clear()` eviction. `src/phase_unwrap/training/train.py:241` — `DataLoader` uses `shuffle=True`.
- **Impact**: Global uniform random sampling causes near 100% cache misses, leading to entire 128-item HDF5 blocks being read from disk repeatedly to serve just 1 random item. This yields 128x I/O amplification and crippling epoch times.
- **Effort**: M (a day-ish)
- **Risk**: LOW - Solves the data loading bottleneck without affecting the actual inputs.
- **Confidence**: HIGH
- **Fix sketch**: Implement a custom `BatchSampler` that shuffles HDF5 block indices and yields shuffled sequential indices within each block, or redesign as an `IterableDataset`. Replace the naive `dict.clear()` with an `OrderedDict` LRU cache.

## Priority 2: Performance & Security Quick Wins

### [SEC-02] Arbitrary Code Execution in Checkpoint Loading
- **Evidence**: `src/phase_unwrap/training/train.py:112`, `src/phase_unwrap/analysis/export.py:65`, `src/phase_unwrap/analysis/inference.py:58` — `torch.load` is explicitly called with `weights_only=False`.
- **Impact**: Loading an untrusted or downloaded checkpoint (`.pth` file) will unpickle and execute arbitrary Python code.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Change `weights_only=False` to `weights_only=True` in all `torch.load` calls.

### [PERF-02] Synchronous CPU-to-GPU Transfer in Loss Function
- **Evidence**: `src/phase_unwrap/training/train.py:275` — `loss_fn = MAEGradLoss(...)` is instantiated without `.to(device)`. `src/phase_unwrap/core/ops.py:29` — `FixedSobel.forward()` dynamically moves `self.gx` and `self.gy` to device.
- **Impact**: A host-to-device synchronization block occurs on every step because CPU tensors are dynamically pushed to the GPU inside the backward pass, severely starving GPU throughput.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Call `loss_fn = loss_fn.to(device)` directly after instantiating it, ensuring `FixedSobel` buffers exist on the GPU persistently.

## Priority 3: Architecture & Tech Debt

### [ARCH-01] God Function / Layering Violation in Training Loop
- **Evidence**: `src/phase_unwrap/training/train.py:234` — The `train()` function is massive (over 450 lines) and directly manages OOM recovery, scheduler math, metric logging, checkpointing, CSV writing, and launches a background thread for visualization plotting.
- **Impact**: Makes the training loop brittle and hard to test in isolation.
- **Effort**: M (a day-ish)
- **Risk**: MED 
- **Confidence**: HIGH
- **Fix sketch**: Extract checkpointing, metrics CSV logging, and visualization dispatch into separate callback or manager classes.

### [ARCH-02] Surface asymmetry: w_curv / Laplacian loss calculation outside Loss class
- **Evidence**: `src/phase_unwrap/core/losses.py:32` — `MAEGradLoss` computes MAE and Gradient loss, but omits the curvature loss entirely. `train.py:416` and `tune.py:182` manually add it.
- **Impact**: Architectural friction. The loss function class doesn't encapsulate the actual full equation promised by the README.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Move the `curvature_loss` invocation and `w_curv` weighting directly into `MAEGradLoss._single_scale_loss()`.

### [ARCH-03] Global State Modification in Library Code
- **Evidence**: `src/phase_unwrap/data/dataset.py:192` — `mp.set_start_method("spawn", force=True)` explicitly forces the process start method.
- **Impact**: Modifying global multiprocessing state deep inside a library function can break caller applications.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Move the multiprocessing start method configuration out of the data layer and into the main CLI entrypoint (`cli.py`).

## Priority 4: DX, Tooling & Documentation

### [DX-01] Align README CLI documentation with nested implementation
- **Evidence**: `README.md:25` — explicitly claims "CLI is flat (not nested)", but `src/phase_unwrap/cli.py:22` registers subcommands using `app.add_typer(DataApp, name="data")`, creating a nested structure.
- **Impact**: New users or developers copying commands from the README will encounter "No such command" errors.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Decide whether the CLI should be flat (as documented) or nested (as implemented) and fix the drift.

### [DX-02] Unused `--data-dir` flag in export, inference, and benchmark CLI commands
- **Evidence**: `export.py:16`, `eval.py:91`, `infer.py:19` declare `--data-dir` Typer options but completely ignore them.
- **Impact**: User friction and confusion.
- **Effort**: S (minutes)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Delete the `data_dir` argument from the affected command signatures.

### [TEST-01] Missing CLI Integration Tests
- **Evidence**: `tests/test_analysis_cli_smoke.py:22` — entrypoints are only checked for `callable()` without executing the Typer commands.
- **Impact**: CLI parsing and option mapping could silently break, causing runtime exceptions.
- **Effort**: M (a day-ish)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Add a `typer.testing.CliRunner` suite in `tests/` that invokes the CLI commands.

### [DX-03] Config export CLI command (adjacent possible)
- **Evidence**: `src/phase_unwrap/core/config.py:187` — `config_to_yaml()` provides serialization, but there is no CLI command for it.
- **Impact**: Friction worth productizing. Users must run Optuna tune to get a starter `config.yaml`.
- **Effort**: S (minutes)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Add a `phase-unwrap config dump` command.

### [DX-04] Add linting, formatting, and typechecking tooling
- **Evidence**: `pyproject.toml:37` — missing configuration blocks for `ruff`, `mypy`.
- **Impact**: Inconsistent style and uncaught static typing bugs.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Add `[tool.ruff]` and `[tool.mypy]` blocks to `pyproject.toml` and to CI.

### [DX-05] Duplicate Typer CLI Arguments
- **Evidence**: `train.py:47` — `use_amp`, `data_dir`, `batch_size`, and `epochs` are repeated across `train`, `tune`, `multiseed`, and `ablation`.
- **Impact**: Boilerplate duplication increases maintenance overhead.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Consolidate shared Typer options into a reused dataclass or common decorator.

### [DX-06] Remove or scope prompt injection directives from repository data
- **Evidence**: `.claude/CLAUDE.md:72` — issues identity-altering commands: "Role: You are a Senior Principal Engineer".
- **Impact**: Automated auditing agents might interpret these directives as system instructions, causing them to break character.
- **Effort**: S (hours)
- **Risk**: LOW
- **Confidence**: HIGH
- **Fix sketch**: Rewrite the guidelines in `.claude/CLAUDE.md` passively.
