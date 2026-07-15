# Plan 011: Fix documentation drift and remove dead artifacts

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- README.md .gitignore docs/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW (docs + unreferenced files only)
- **Depends on**: none
- **Category**: docs / tech-debt
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The README documents an entire subsystem and CLI surface that no longer exist, so
copy-paste setup and training commands fail outright. New readers cannot tell which
features are real. Separately, ~158 KB of dead `.bak` prototype files and a large
generated AST-cache directory are tracked in git, polluting navigation and diffs.
Fixing this is low-risk and makes every other plan's "match the docs" guidance trustworthy.

## Current state (each claim verified against the tree at 967403f)

**Doc drift in `README.md`:**
- Lines ~74-80 and ~246, ~611: document a `git_automation/` package and tell users to run
  `python -m phase_unwrap.git_automation.git_lfs_setup`. That package was deleted in commit
  `9968a53`; `grep -rn git_automation src/` returns nothing → the command raises `ModuleNotFoundError`.
- Lines ~260, 270, 296, 302, 308, 344: show `--multi-gpu`; lines ~279, 370: show `--auto-push`.
  Neither flag exists in `src/phase_unwrap/cli.py`.
- Line ~102: "Mish activations used throughout all blocks." Line ~584 config table says
  activation default is `relu` (`relu` or `silu`). Actual default: `config.py:42` `activation = "silu"`;
  `unet.py:76-81` supports relu/mish/else→SiLU. Three-way contradiction; Mish is not the default anywhere.
- Lines ~750-761: documents `uv run pytest tests/test_git_automation.py -v` and "31 tests
  covering ... git automation." That file does not exist; real suite is 3 files.
- Config-defaults table (lines ~580-590) omits real fields: `model.use_coordconv`,
  `optim.compile`, `loss.w_data`, `loss.int_wgrad`, `data.test_frac`, and the entire `aug` group.

**Doc drift in `docs/md/absolute_phase_unet_training_report.md`:**
- Lines 1, 5, 307 describe "the training script `src/try.py`" reading MATLAB `.mat` files.
  No `src/try.py` exists (only `src/try.py.bak`); the real pipeline reads `.h5`
  (`config.py:27` `pattern = "*.h5"`). The report describes a pre-package prototype.

**Dead artifacts:**
- `src/pinn_phase.py.bak`, `src/pinn_phase_modified.py.bak`, `src/test_for_latest.py.bak`,
  `src/try.py.bak` (~158 KB total), and `test.bak` (0 bytes, repo root) — all git-tracked,
  zero references from any `.py`.
- `graphify-out/` — 72 tracked files (AST cache `cache/ast/*.json`, `GRAPH_REPORT.md`,
  `.graphify_*`). `.gitignore` has no entry for it. It is a regenerable tool cache.
- `status_handoff.md` — stale agent "COGNITIVE STATE" doc (already gitignored; delete from tree).

`configs/default.yaml:2` claims "All values match the built-in defaults" but sets
`data.val_frac: 0.3`, `model.base: 16`, `optim.epochs: 10`, `optim.batch_size: 8`,
`loss.w_curv: 0.003` — none of which match `config.py`. It is a smoke-test config mislabeled.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Confirm no code refs the .bak | `grep -rn "\.bak\|git_automation\|src/try\.py\|--multi-gpu\|--auto-push" src/` | only doc/no matches in src |
| CLI help (verify real flags) | `uv run phase-unwrap --help` and `uv run phase-unwrap train --help` | lists actual commands/flags |
| Import still works after deletes | `uv run python -c "import phase_unwrap"` | exit 0 |

## Scope

**In scope**:
- `README.md` — correct the four drift areas above.
- `docs/md/absolute_phase_unet_training_report.md` — mark historical or correct to `.h5`/package layout.
- `.gitignore` — add `graphify-out/`, `*.bak`.
- Remove tracked files: the four `src/*.bak`, `test.bak`, `status_handoff.md`, and untrack `graphify-out/`.
- `configs/default.yaml` — fix the false "matches defaults" header comment (or rename to `tiny`/`smoke`).

**Out of scope**:
- Do NOT change any `src/` code behavior — this is docs + file hygiene only.
- Do NOT resolve the activation *default* question in code (that is a config/model decision);
  only make the README describe what the code actually does today (`silu` default, relu/mish/silu supported).
- Do NOT delete `configs/default.yaml` itself; it is referenced by workflows/README.

## Steps

### Step 1: Remove dead files and update `.gitignore`
```
git rm src/pinn_phase.py.bak src/pinn_phase_modified.py.bak src/test_for_latest.py.bak src/try.py.bak test.bak
git rm --cached -r graphify-out
git rm status_handoff.md   # if tracked; otherwise plain rm
```
Append to `.gitignore`: `graphify-out/` and `*.bak`.

**Verify**: `git status --short` shows the deletions; `uv run python -c "import phase_unwrap"` exits 0;
`grep -rn "\.bak" src/` returns nothing.

### Step 2: Fix README `git_automation` / CLI-flag drift
Remove the `git_automation/` section and every `--multi-gpu` / `--auto-push` /
`git_lfs_setup` reference. Cross-check every remaining command against
`uv run phase-unwrap --help` output and replace stale invocations with real ones.

**Verify**: `grep -n "git_automation\|--multi-gpu\|--auto-push\|git_lfs_setup" README.md` returns nothing.

### Step 3: Fix README activation + test claims + config table
- State the activation default as `silu` (options: relu, mish, silu), matching `config.py:42`/`unet.py`.
- Replace the `test_git_automation.py` line and "31 tests ... git automation" with the real
  suite (`test_losses_ops.py`, `test_generate.py`, `test_model.py`) and an accurate count.
- Add the missing config fields to the defaults table.

**Verify**: `grep -n "test_git_automation\|Mish activations used throughout" README.md` returns nothing.

### Step 4: Fix the stale training report and config header
- In `docs/md/absolute_phase_unet_training_report.md`, either prepend a clear
  "HISTORICAL — describes the pre-package `src/try.py` prototype" banner or update the data-format
  and script references to the current `.h5` / `phase_unwrap` pipeline.
- In `configs/default.yaml`, correct or remove the line-2 claim that values match built-in defaults.

**Verify**: `grep -n "src/try.py" docs/md/absolute_phase_unet_training_report.md` — either gone or under a HISTORICAL banner.

## Done criteria

- [ ] `grep -rn "git_automation\|--multi-gpu\|--auto-push" src/ README.md` → no matches
- [ ] `.bak` files, `test.bak`, `status_handoff.md` removed; `graphify-out/` untracked and gitignored
- [ ] `uv run python -c "import phase_unwrap"` exits 0
- [ ] README activation/test/config claims match the code
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

- `uv run phase-unwrap --help` fails to run (cannot verify real CLI surface) → stop, report.
- Any `.bak` file turns out to be imported by live code (`grep` finds a reference) → stop, report.
- `git rm --cached -r graphify-out` would remove something also referenced by a live tool config → stop, report.

## Maintenance notes

Keep README command examples in sync with `cli.py` going forward; a doctest or a CI step
that runs `phase-unwrap --help` and greps for documented commands would prevent regression.
