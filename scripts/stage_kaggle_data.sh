#!/usr/bin/env bash
# Stage HDF5 datasets from Kaggle input directory to /tmp for fast I/O
set -e

INPUT_DIR="${1:-/kaggle/input}"
TARGET_DIR="${2:-/tmp/training_set}"

echo "[Staging] Searching for HDF5 shards under '${INPUT_DIR}'..."
mkdir -p "${TARGET_DIR}"

# Find and copy all .h5 shard files to local fast /tmp storage
find "${INPUT_DIR}" -name "*.h5" -exec cp {} "${TARGET_DIR}/" \;

SHARD_COUNT=$(ls -1 "${TARGET_DIR}"/*.h5 2>/dev/null | wc -l || echo 0)
echo "[Staging] Successfully staged ${SHARD_COUNT} HDF5 shards into '${TARGET_DIR}'."
