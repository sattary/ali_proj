# Plan 019: Fix gzip storage codec and DataLoader throughput

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat -- src/phase_unwrap/data/generate.py src/phase_unwrap/data/dataset.py`
> If either in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: perf
- **Planned at**: 2026-07-20

## Why this matters

Observed throughput on an RTX 3060: **1.03 it/s** at batch 64, 128x128, for a
`base` UNet that should run north of 10 it/s. The GPU is starving, not
computing. Root cause is the on-disk storage codec, not the model or the
device — so DDP / multi-GPU would do nothing.

Two independent problems:

1. **Codec (primary).** `data/generate.py` writes every sample as its own
   `gzip`-compressed 1-image chunk (`chunks=(1,1,NY,NX)`). Every `__getitem__`
   read therefore costs a CPU `zlib` decompression. At batch 64 that is 64
   decompressions per iteration on the worker pool, 80k per epoch, fully
   CPU-bound. The dataset's 128-block cache (`H5ShardDataset._cache_size`)
   cannot amortize this because each 1-sample chunk is decompressed
   independently regardless of the read span.

2. **Worker lifecycle (secondary).** `persistent_workers` defaults `False` and
   no `prefetch_factor` is set. On Windows `spawn`, workers respawn every epoch,
   and `vis_batch = next(iter(train_loader))` in `train.py` forces an extra
   spawn/kill cycle before epoch 1.

## Current state

- `src/phase_unwrap/data/generate.py` lines 118-119:
```python
        f.create_dataset("I", data=I_buf, chunks=(1, 1, NY, NX), compression="gzip", compression_opts=4)
        f.create_dataset("phi", data=phi_buf, chunks=(1, 1, NY, NX), compression="gzip", compression_opts=4)
```

- `src/phase_unwrap/data/dataset.py` `build_dataloaders`, `dl_kwargs`:
```python
    dl_kwargs = dict(
        batch_size=cfg.optim.batch_size,
        num_workers=safe_workers,
        pin_memory=use_cuda,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    if safe_workers > 0 and getattr(cfg.data, "persistent_workers", False):
        dl_kwargs["persistent_workers"] = True
```

## Scope

**In scope**:
- `src/phase_unwrap/data/generate.py` — codec + chunk shape
- `src/phase_unwrap/data/dataset.py` — `prefetch_factor` in `dl_kwargs`
- `configs/default.yaml` — enable `persistent_workers`, raise workers/prefetch

**Out of scope** (flag only, do not touch):
- `augmentation.py` misleading async docstring
- `train.py` OOM auto-halving handler

## Steps

### Step 1: Switch codec to lzf with block-aligned chunks — `generate.py`
Replace the two `create_dataset` calls so chunks align with the dataset's
128-sample block cache and use `lzf` (≈10x faster decompression than gzip-4):
```python
    chunk_n = min(128, n_in_shard)
    f.create_dataset("I", data=I_buf, chunks=(chunk_n, 1, NY, NX), compression="lzf")
    f.create_dataset("phi", data=phi_buf, chunks=(chunk_n, 1, NY, NX), compression="lzf")
```
Rationale: one chunk read now serves a full 128-block, so the existing
`HDF5BlockSampler` + `_chunk_cache` finally pay off — a block miss triggers
exactly one lzf decompress instead of 128 gzip decompresses. Samples are
byte-identical to the gzip version (codec is lossless); the generator seed path
is untouched, so data stays reproducible.

**Verify**: `uv run pytest tests/ -q` → all pass.

### Step 2: Add prefetch_factor — `dataset.py`
In `dl_kwargs`, when `safe_workers > 0`, set `prefetch_factor=4` so each worker
stages batches ahead of the GPU. Guard it under the worker check (invalid when
`num_workers=0`).

**Verify**: `uv run pytest tests/test_dataset.py -q` → all pass.

### Step 3: Config defaults — `configs/default.yaml`
Under `data:` set `persistent_workers: true`, `workers: 4` (physical core count
on this box), and document `prefetch_factor` is handled in code. Keep
`batch_size` as-is (12 GB card tolerates 64).

**Verify**: `uv run phase-unwrap config ...` loads without error (or a smoke
train for 1 epoch shows it/s > 10 and `nvidia-smi` GPU-util high).

### Step 4: Regenerate the dataset
```bash
uv run phase-unwrap data generate --out-dir data/training_set --seed 1337 --num-samples <N>
```
lzf shards for 180k samples ≈ 12 GB. Reproducible; only encoding changes.

## Done criteria

- [ ] `uv run pytest tests/ -q` exits 0
- [ ] New shards written with lzf (verify: `h5py` reports `compression='lzf'`)
- [ ] Smoke run shows it/s ≫ 1 and GPU-util high in `nvidia-smi`
- [ ] No files outside the in-scope list modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back if:
- Tests break in a way not trivially fixable by the codec change.
- lzf disk footprint is unacceptable (fall back to `compression=None`).
- Regenerated data changes sample values (would indicate a non-codec edit crept in).
