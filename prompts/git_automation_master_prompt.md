# Master Prompt: Git Automation for Cloud Training

Implement a production-grade git automation system for Deep Learning training on cloud platforms (Kaggle, Colab). This module enables automatic pushing of experiment artifacts to GitHub at configurable intervals during long-running training jobs.

## Architecture Philosophy

This module follows the **Callback Pattern** integrated with training loops. It separates concerns into:

- **Environment Detection**: Only activates in Kaggle/Colab environments (safety)
- **State Tracking**: JSON-based persistence for resume capability after interruptions
- **Artifact Packaging**: Zip compression with configurable exclusions
- **Git Operations**: Branch-per-run strategy with PAT authentication
- **CLI Integration**: Typer decorators for zero-friction CLI addition

## Design Decisions

1. **Branch-per-Run**: Each training run creates a unique branch (`results/{run_name}_{timestamp}`) to avoid conflicts with main code development
2. **Zip Packaging**: Compresses entire run directory (checkpoints, configs, logs) into single artifact for atomic pushes
3. **Dry-Run Mode**: Test configuration without actual git operations
4. **Graceful Degradation**: Push failures don't crash training; state is preserved for retry
5. **Git LFS Integration**: Automatic setup for large files (.pth, .zip artifacts)

## Integration Pattern

```python
# In your training script:
from your_project.git_automation import AutoPushCallback

# Create callback
callback = AutoPushCallback(
    run_dir="runs/experiment_001",
    push_interval=100,  # Push every 100 epochs
    pat=os.environ.get("GITHUB_PAT"),
)

# Hook into training loop
for epoch in range(total_epochs):
    train_epoch(...)
    
    # Auto-push at intervals
    callback.on_epoch_end(epoch, total_epochs, metrics={"MAE": 0.042})

callback.on_train_end()
```

## Directory Structure

```
src/your_project/git_automation/
├── __init__.py           # Exports + MPLBACKEND fix for cloud
├── callback.py           # AutoPushCallback (main integration point)
├── environment.py        # Kaggle/Colab detection
├── git_pusher.py         # Git operations (branch, commit, push)
├── zip_packer.py         # Artifact compression
├── state_tracker.py      # JSON state for resume
├── git_lfs_setup.py      # LFS initialization script
└── cli_integration.py    # Typer decorators
```

## Environment Variables

- `GITHUB_PAT`: GitHub Personal Access Token (required for push)
- `MPLBACKEND`: Set to 'Agg' for non-interactive cloud environments

## File Patterns to LFS Track

- `artifacts/*.zip` (compressed run artifacts)
- `*.pth` (model checkpoints)
- `*.onnx` (exported models)
- `data/**/*.h5` (dataset files)

## Security Considerations

1. PAT is never logged or stored in code
2. Remote URL is sanitized before display (removes PAT)
3. Environment verification prevents accidental pushes from local development

## Cloud Platform Detection

Detects Kaggle via:
- `/kaggle` directory exists
- `KAGGLE_KERNEL_RUN_TYPE` environment variable
- `KAGGLE_CONTAINER_NAME` environment variable

Detects Colab via:
- `google.colab` module importable

## State File Schema

`.push_state.json`:
```json
{
  "epoch": 500,
  "total_epochs": 1000,
  "push_interval": 100,
  "last_push_epoch": 500,
  "best_mae": 0.0423,
  "last_updated": "2024-01-15T10:30:00",
  "zip_path": "runs/exp_001_epoch_500_of_1000_20240115_103000.zip"
}
```

---

```json
{
  "module_name": "git_automation",
  "package_path": "src/{PROJECT_NAME}/git_automation/",
  "components": [
    {
      "name": "AutoPushCallback",
      "file": "callback.py",
      "purpose": "Main callback class for training loop integration",
      "key_methods": [
        "__init__(run_dir, push_interval, pat, dry_run, force, include_checkpoints, repo_dir)",
        "initialize() -> Initialize git pusher and setup branch",
        "on_epoch_end(epoch, total_epochs, metrics) -> bool (pushed or not)",
        "on_train_end(final_metrics) -> Cleanup",
        "get_status() -> dict with current state"
      ],
      "dependencies": ["StateTracker", "ZipPacker", "GitPusher", "environment"]
    },
    {
      "name": "GitPusher",
      "file": "git_pusher.py",
      "purpose": "Execute git operations safely",
      "key_methods": [
        "__init__(repo_dir, branch_name, pat, dry_run)",
        "setup_branch() -> Create/checkout results branch",
        "push_artifact(zip_path, epoch, total_epochs, metrics, is_final) -> bool",
        "cleanup() -> Return to original branch",
        "get_remote_url() -> Sanitized remote URL"
      ],
      "git_operations": ["checkout -b", "add", "commit", "push", "remote set-url"]
    },
    {
      "name": "ZipPacker",
      "file": "zip_packer.py",
      "purpose": "Compress run artifacts into zip with metadata",
      "key_methods": [
        "__init__(run_dir, compression_level, include_checkpoints)",
        "create_zip(epoch, total_epochs, metrics, output_dir) -> zip_path",
        "get_latest_zip() -> Optional[zip_path]"
      ],
      "exclude_patterns": ["*.pyc", "__pycache__", ".pytest_cache", "*.log", "visuals/*"]
    },
    {
      "name": "StateTracker",
      "file": "state_tracker.py",
      "purpose": "JSON persistence for resume capability",
      "key_methods": [
        "__init__(run_dir)",
        "read_state() -> Optional[dict]",
        "write_state(epoch, total_epochs, push_interval, last_push_epoch, best_mae, zip_path)",
        "should_push(current_epoch) -> bool",
        "get_resume_epoch() -> int",
        "is_finished() -> bool",
        "validate_consistency(expected_total, expected_interval) -> bool"
      ],
      "state_file": ".push_state.json"
    },
    {
      "name": "Environment Detection",
      "file": "environment.py",
      "purpose": "Detect and verify cloud environment",
      "functions": [
        "is_kaggle() -> bool",
        "is_colab() -> bool",
        "is_cloud_environment() -> bool",
        "get_environment_name() -> str ('kaggle'|'colab'|'local')",
        "verify_cloud_environment(force=False) -> None (raises RuntimeError if not cloud)"
      ]
    },
    {
      "name": "CLI Integration",
      "file": "cli_integration.py",
      "purpose": "Typer decorators for CLI commands",
      "functions": [
        "add_auto_push_args() -> decorator adding --auto-push-interval, --auto-push-dry-run, --force-auto-push, --auto-push-pat",
        "validate_auto_push_config(interval, dry_run, force) -> bool",
        "create_auto_push_callback(run_dir, interval, dry_run, force, pat, include_checkpoints) -> Optional[AutoPushCallback]"
      ]
    },
    {
      "name": "Git LFS Setup",
      "file": "git_lfs_setup.py",
      "purpose": "One-time LFS initialization script",
      "entry_point": "if __name__ == '__main__': setup_git_lfs()",
      "actions": ["lfs install", "lfs track <patterns>", "add .gitattributes", "commit", "push"]
    }
  ],
  "integration_points": [
    {
      "location": "Training script __init__.py or main entry",
      "code": "import os\nos.environ['MPLBACKEND'] = 'Agg'  # Must be before any matplotlib/seaborn import",
      "reason": "Prevents interactive backend errors in cloud environments"
    },
    {
      "location": "Training loop initialization",
      "code": "callback = AutoPushCallback(...); callback.initialize()",
      "reason": "Sets up git branch before training starts"
    },
    {
      "location": "End of each epoch",
      "code": "callback.on_epoch_end(epoch, total_epochs, metrics_dict)",
      "reason": "Triggers push at configured intervals"
    },
    {
      "location": "End of training",
      "code": "callback.on_train_end(final_metrics)",
      "reason": "Final push and cleanup"
    },
    {
      "location": "CLI command definition",
      "code": "@app.command()\n@add_auto_push_args()\ndef train(..., auto_push_interval: int = None, ...):",
      "reason": "Exposes auto-push options in CLI"
    }
  ],
  "customization_variables": {
    "PROJECT_NAME": "Your Python package name (e.g., 'phase_unwrap')",
    "METRIC_NAME": "Primary metric for commit messages (e.g., 'MAE', 'Accuracy')",
    "CHECKPOINT_EXTENSION": "Model file extension (e.g., '.pth', '.ckpt', '.pt')",
    "DATASET_EXTENSION": "Dataset file extension (e.g., '.h5', '.tfrecord')",
    "LARGE_FILE_PATTERNS": ["List of patterns to track in Git LFS"]
  },
  "dependencies": ["gitpython (optional, fallback to subprocess)", "typer (for CLI integration)"],
  "smoke_tests": [
    {
      "name": "Environment Detection",
      "test": "assert isinstance(is_cloud_environment(), bool)"
    },
    {
      "name": "ZipPacker Creation",
      "test": "Create temp dir with files; ZipPacker(temp_dir).create_zip(1, 10); assert zip exists"
    },
    {
      "name": "StateTracker Write/Read",
      "test": "StateTracker(temp_dir).write_state(100, 1000, 50, 100, 0.5); state = read_state(); assert state['epoch'] == 100"
    },
    {
      "name": "AutoPushCallback Dry Run",
      "test": "Create mock git repo; callback = AutoPushCallback(repo, 100, dry_run=True, force=True); callback.initialize(); assert callback.on_epoch_end(100, 1000) is True"
    },
    {
      "name": "Git LFS Setup Script",
      "test": "Run git_lfs_setup.py in mock repo; assert .gitattributes exists and contains 'artifacts/*.zip'"
    }
  ]
}
```
