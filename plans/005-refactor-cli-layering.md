# Plan 005: Refactor CLI into submodule layering

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/cli.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The `cli.py` file has grown into a 975-line God Object, registering over 20 visualization, training, evaluation, and data commands inline. This causes high friction for editing, merge conflicts, and blurs CLI parsing with application logic. Breaking it down will drastically improve developer experience.

## Current state

- `src/phase_unwrap/cli.py` — God Object containing `PlotApp`, `EvalApp`, `ExportApp` and all underlying command implementations.

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Verify    | `uv run phase-unwrap --help`| Shows all subcommands |
| Tests     | `uv run pytest tests/`   | all pass            |

## Scope

**In scope**:
- `src/phase_unwrap/cli.py`
- `src/phase_unwrap/cli_cmds/` (create this directory)
- `src/phase_unwrap/cli_cmds/__init__.py` (create)
- `src/phase_unwrap/cli_cmds/plot.py` (create)
- `src/phase_unwrap/cli_cmds/train.py` (create)
- `src/phase_unwrap/cli_cmds/eval.py` (create)
- `src/phase_unwrap/cli_cmds/export.py` (create)
- `src/phase_unwrap/cli_cmds/data.py` (create)

## Steps

### Step 1: Create `cli_cmds` package
Create the `src/phase_unwrap/cli_cmds/` directory.

### Step 2: Extract sub-commands
Move the implementations and Typer app instantiations for `data`, `train`, `eval`, `export`, and `plot` into their respective files in `cli_cmds/`.

### Step 3: Stitch in `cli.py`
Rewrite `src/phase_unwrap/cli.py` to simply import these Typer sub-apps and attach them via `app.add_typer(...)`. Make sure to keep `main()` exposed as it's the entrypoint.

**Verify**: `uv run phase-unwrap --help` → exits 0 and lists all commands.

## Test plan

- Run `uv run pytest tests/` to ensure no circular import or CLI structure regressions break tests.

## Done criteria

- [ ] `uv run phase-unwrap --help` exits 0
- [ ] `uv run pytest tests/` exits 0
- [ ] `src/phase_unwrap/cli.py` is under 200 lines
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- Breaking the CLI into submodules causes circular import errors with `core/config.py` that you cannot easily resolve.
