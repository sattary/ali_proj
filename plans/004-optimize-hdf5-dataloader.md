# Plan 004: Optimize HDF5 DataLoader chunk reads

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/data/dataset.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: MED
- **Depends on**: plans/001-train-integration-tests.md
- **Category**: perf
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The `H5ShardDataset` issues separate scalar reads per batch item to HDF5 inside `__getitem__`. HDF5 is optimized for contiguous chunk reads, so point-by-point reads bottleneck the training loop with high disk/CPU overhead, effectively starving the GPUs of data.

## Current state

- `src/phase_unwrap/data/dataset.py` — lazy HDF5 dataset loading.
Lines 74-75:
```python
        I_np = f[self.I_key][local_idx]
        phi_np = f[self.phi_key][local_idx]
```

## Commands you will need

| Purpose   | Command                  | Expected on success |
|-----------|--------------------------|---------------------|
| Tests     | `uv run pytest tests/test_dataset.py`| all pass            |

## Scope

**In scope**:
- `src/phase_unwrap/data/dataset.py`

## Steps

### Step 1: Implement an internal chunk cache in H5ShardDataset
Instead of reading 1 element at a time, buffer a block of elements in memory. Add an LRU cache or a simple buffer that stores a slice of the dataset (e.g., indices `[start:start+128]`) when `local_idx` misses the buffer.
Modify `__init__` to initialize `self._chunk_cache = {}` and `self._cache_size = 128`.
Modify `__getitem__` to compute the block start `block_start = (local_idx // self._cache_size) * self._cache_size`, and block end `block_end = min(block_start + self._cache_size, shard_size)`.
If `(shard_idx, block_start)` is not in `_chunk_cache`, read `f[self.I_key][block_start:block_end]` and `f[self.phi_key][block_start:block_end]` fully into memory (numpy arrays) and store them in the cache (evicting old blocks if necessary).
Then return the item from the cached block: `cached_I[local_idx - block_start]`.

**Verify**: `uv run pytest tests/test_dataset.py` → all pass.

## Test plan

- Rely on the integration tests from Plan 001 to ensure `H5ShardDataset` still returns the exact same shapes and types, and doesn't crash on edges.

## Done criteria

- [x] `uv run pytest tests/test_dataset.py` exits 0
- [x] No files outside the in-scope list are modified
- [x] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- Modifying the dataset breaks the test suite and cannot be trivially fixed.
- The `DataLoader` workers run out of memory due to the cache size (reduce `_cache_size` to 32 or 64 if so).
