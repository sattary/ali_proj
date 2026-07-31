# Plan 027: Refactor `src/` Over-Engineering & Simplify Utilities (Ponytail Audit)

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 631bac8..HEAD -- src/phase_unwrap/core/config.py src/phase_unwrap/core/utils.py src/phase_unwrap/data/dataset.py src/phase_unwrap/model/blocks.py src/phase_unwrap/model/unet.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: refactor
- **Planned at**: commit `631bac8`, 2026-07-30

## Why this matters

Following the ponytail codebase audit, 5 over-engineered abstractions and redundant utility layers were identified in `src/phase_unwrap/`:
1. **Manual binary search while-loop (`dataset.py`)**: `H5ShardDataset._locate()` implements a custom binary search algorithm. Replacing it with standard library `bisect.bisect_right()` eliminates custom search logic and potential boundary edge cases.
2. **Double fallback config parser ladder (`config.py`)**: `_load_mapping` and `_update_dataclass` contain redundant try-except blocks attempting both JSON and YAML parsing twice. Simplifying to a direct extension-based dispatch reduces code complexity.
3. **Legacy PyTorch compatibility wrapper (`blocks.py`)**: `meshgrid_ij()` handles pre-PyTorch 1.10 `indexing` keyword support. Modern PyTorch (>= 2.0) natively supports `indexing="ij"`, making this wrapper redundant.
4. **Custom `ensure_dir` wrapper (`utils.py`)**: `ensure_dir(path)` is a 1-line wrapper around stdlib `Path(path).mkdir(parents=True, exist_ok=True)`. Inlining or replacing with standard `Path.mkdir` reduces unnecessary utility abstractions.
5. **Legacy facade shim (`unet.py`)**: `src/phase_unwrap/model/unet.py` serves as a re-export shim after the Plan 028 modularization. Updating direct imports to `phase_unwrap.model` across analysis scripts allows clean deprecation of the facade shim.

## Current state

- `src/phase_unwrap/data/dataset.py` (lines 72–81):
```python
    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._shard_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo, idx - self._cumulative[lo]
```
- `src/phase_unwrap/model/blocks.py` (lines 13–18):
```python
def meshgrid_ij(x: torch.Tensor, y: torch.Tensor, **kw):
    """Compatibility wrapper for older PyTorch meshgrid API."""
    try:
        return torch.meshgrid(x, y, indexing="ij", **kw)
    except TypeError:
        return torch.meshgrid(x, y)
```
- `src/phase_unwrap/core/utils.py` (lines 11–13):
```python
def ensure_dir(path: str | os.PathLike) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
```
- `src/phase_unwrap/core/config.py` (lines 158–179):
```python
def _load_mapping(path: Path) -> Dict[str, Any]:
    ...
```

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit Tests | `uv run pytest tests/ -q` | All 57+ tests pass |
| Lint Check | `uv run ruff check src/phase_unwrap/` | Exit 0, no errors |
| Type Check | `uv run mypy src/phase_unwrap/` | Exit 0, no errors |

## Scope

**In scope**:
- `src/phase_unwrap/data/dataset.py` — Replace `_locate` while-loop with `bisect.bisect_right`.
- `src/phase_unwrap/model/blocks.py` — Replace `meshgrid_ij` with native `torch.meshgrid(x, y, indexing="ij")`.
- `src/phase_unwrap/core/config.py` — Clean up `_load_mapping` parser ladder.
- `src/phase_unwrap/core/utils.py` — Simplify `ensure_dir`.

**Out of scope**:
- Loss functions in `src/phase_unwrap/core/losses.py`.
- Model forward pass signatures in `PCLCNModel` or `UNetRes2_AbsPhase`.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit per step with imperative message: e.g. `Use stdlib bisect in H5ShardDataset and native meshgrid in blocks`

---

## Steps

### Step 1: Replace custom binary search with `bisect.bisect_right` in `dataset.py`

In `src/phase_unwrap/data/dataset.py`:
1. Import `bisect` from stdlib.
2. Refactor `_locate`:

```python
import bisect

def _locate(self, idx: int) -> Tuple[int, int]:
    # self._cumulative has leading 0, so bisect_right - 1 gives shard index
    shard_idx = bisect.bisect_right(self._cumulative, idx) - 1
    shard_idx = min(shard_idx, len(self._shard_sizes) - 1)
    local_idx = idx - self._cumulative[shard_idx]
    return shard_idx, local_idx
```

**Verify**: `uv run pytest tests/test_dataset.py -q` → All tests pass.

### Step 2: Use native `torch.meshgrid` in `blocks.py`

In `src/phase_unwrap/model/blocks.py`:
Replace `meshgrid_ij` usages inside `AddCoords` with `torch.meshgrid(..., indexing="ij")` directly and remove the `meshgrid_ij` fallback function.

```python
class AddCoords(nn.Module):
    def __init__(self, h: int = 128, w: int = 128) -> None:
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("coords", torch.stack([xx, yy], dim=0).unsqueeze(0))
```

**Verify**: `uv run pytest tests/test_model.py -q` → All tests pass.

### Step 3: Simplify `_load_mapping` in `config.py`

In `src/phase_unwrap/core/config.py`, simplify `_load_mapping`:

```python
def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("YAML requested but PyYAML is not installed.")
        return yaml.safe_load(text) or {}
    return json.loads(text)
```

**Verify**: `uv run pytest tests/test_config.py -q` → All tests pass.

### Step 4: Run full test suite & linting

```bash
uv run pytest tests/ -q
uv run ruff check src/phase_unwrap/
```

---

## Test plan

- Run existing unit test suite verifying `H5ShardDataset` indexing, `AddCoords` output shape, and config parsing.
- Verification command: `uv run pytest tests/ -q`

## Done criteria

- [ ] `_locate()` uses stdlib `bisect.bisect_right`.
- [ ] `AddCoords` uses native `torch.meshgrid(..., indexing="ij")`.
- [ ] `_load_mapping` simplifies config loading without redundant try-except blocks.
- [ ] All 57+ unit tests pass cleanly.

## STOP conditions

- If `bisect.bisect_right` causes indexing off-by-one errors on shard boundaries, stop and verify `self._cumulative` offset arrays.

## Maintenance notes

- Standard library `bisect` handles $O(\log N)$ shard lookups deterministically without maintenance overhead.
