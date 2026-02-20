# Git Automation for Cloud Training

Auto-push training results to GitHub from Kaggle/Colab environments.

## Features

- **Environment Detection**: Only activates in Kaggle or Colab
- **Configurable Auto-Push**: Push every N epochs
- **Git LFS Support**: Handles large model checkpoints
- **Resume Capability**: Track state to resume interrupted training
- **Dry-Run Mode**: Test setup without actual pushes
- **Smart Zipping**: Compresses artifacts before pushing

## Quick Start

### 1. Setup Git LFS (one-time)

```python
# In your Kaggle/Colab notebook
!python -m phase_unwrap.git_automation.git_lfs_setup
```

### 2. Configure Git

```python
import os
from getpass import getpass

# Set your GitHub PAT
os.environ['GITHUB_PAT'] = getpass("Enter GitHub PAT: ")

# Or use Kaggle Secrets (more secure)
# In Kaggle: Add-ons -> Secrets -> Add 'GITHUB_PAT'
```

### 3. Train with Auto-Push

```bash
# Push every 1000 epochs
phase-unwrap train \
    --epochs 10000 \
    --run-name exp_10k \
    --auto-push-interval 1000

# Test setup first (dry-run)
phase-unwrap train \
    --epochs 100 \
    --run-name test \
    --auto-push-interval 10 \
    --auto-push-dry-run \
    --force-auto-push
```

### 4. Resume After Interruption

```bash
# Resume from checkpoint
phase-unwrap train \
    --resume runs/exp_10k/final.pth \
    --epochs 10000 \
    --run-name exp_10k \
    --auto-push-interval 1000
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--auto-push-interval N` | Enable auto-push every N epochs (train command) |
| `--auto-push` | Push optuna results after HPO completes (tune command) |
| `--auto-push-dry-run` | Test mode (no actual pushes) |
| `--force-auto-push` | Enable outside Kaggle/Colab (testing) |
| `--auto-push-pat TOKEN` | GitHub PAT (or use env var) |

### Tune Auto-Push

After Optuna HPO completes, push results to GitHub:

```bash
phase-unwrap tune \
    --data-dir data/kaggle_full \
    --n-trials 30 \
    --tune-epochs 10 \
    --n-workers 2 \
    --gpu-ids 0,1 \
    --auto-push \
    --auto-push-pat "$GITHUB_PAT"
```

This creates `optuna_{data_name}_n{num_samples}_s{seed}_{timestamp}.zip` containing:
- Optuna study database (`{study_name}.db`)
- `best_config.yaml` with optimal hyperparameters
- `data_config.yaml` with data generation parameters

## How It Works

1. **Epoch End**: Training calls `auto_push_callback.on_epoch_end()`
2. **Check Interval**: Callback checks if it's time to push
3. **Create Zip**: Compresses `run_dir` contents (excludes raw logs/visuals)
4. **Git Operations**: Creates branch, commits, pushes to GitHub
5. **Update State**: Saves `.push_state.json` for resume tracking
6. **Final Push**: Always pushes on last epoch

## Branch Naming

Branches are created as: `results/{run_name}_{timestamp}`

Example: `results/exp_10k_20240219_143022`

## Zip Contents

Each zip contains:
- Checkpoints (best.pth, final.pth)
- metrics.csv
- config.yaml
- metadata.json (epoch info, timestamps)

Excluded:
- Raw logs (*.log)
- Intermediate visuals
- Python cache

## Resume Workflow

1. Training interrupted at epoch 5,420
2. State saved in `.push_state.json`
3. Download latest zip from GitHub branch
4. Extract and use `--resume extracted/checkpoints/final.pth`
5. Training continues from epoch 5,421

## Troubleshooting

### "Not in cloud environment"
Use `--force-auto-push` for testing locally.

### Git LFS quota exceeded
Training continues but auto-push is disabled. Check:
```bash
git lfs status
```

### Push verification fails
The system verifies push with `git status`. If commits remain "ahead", you'll see a warning but training continues.

## Architecture

```
train.py
  └── AutoPushCallback (if enabled)
        ├── ZipPacker          # Create zip artifacts
        ├── GitPusher          # Git operations
        └── StateTracker       # Resume state
```

All components are in `src/phase_unwrap/git_automation/` and isolated from core training logic.
