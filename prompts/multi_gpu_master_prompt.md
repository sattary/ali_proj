# Master Prompt: Multi-GPU Training Support

Implement seamless multi-GPU training support using PyTorch DataParallel with automatic device detection, checkpoint compatibility, and cloud platform optimization.

## Architecture Philosophy

This module follows **Transparent Multi-GPU Pattern** where:

1. **Single-GPU Code Works Unchanged**: Multi-GPU is opt-in via one function call
2. **Checkpoint Compatibility**: Load single-GPU checkpoints on multi-GPU and vice versa
3. **Automatic Detection**: Detects available GPUs and cloud environments
4. **Minimal Boilerplate**: One-line setup, automatic batch size calculation
5. **Graceful Degradation**: Falls back to single GPU if only one available

## Design Decisions

1. **DataParallel over DistributedDataParallel**: Simpler, works in notebooks (Kaggle/Colab)
2. **Module Unwrapping**: `get_model_state_dict()` handles `model.module` automatically
3. **Device Validation**: Validates requested GPU IDs against available hardware
4. **Kaggle Optimization**: Special detection for Kaggle's 2x T4 setup
5. **Batch Size Clarity**: Distinguishes per-GPU vs total batch size
6. **Information Logging**: Print GPU details on setup for debugging

## Integration Pattern

```python
from your_project.training.multi_gpu import setup_multi_gpu, get_model_state_dict, load_model_state_dict

# Build model (single GPU code unchanged)
model = build_model(cfg.model)

# Wrap for multi-GPU (automatically detects and uses all GPUs)
model = setup_multi_gpu(model)

# Training loop (unchanged)
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(batch)
        # ...

# Save checkpoint (handles DataParallel automatically)
checkpoint = {
    "model": get_model_state_dict(model),  # Unwraps if needed
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pth")

# Load checkpoint (works with or without DataParallel)
checkpoint = torch.load("checkpoint.pth")
load_model_state_dict(model, checkpoint["model"])
```

## Directory Structure

```
src/your_project/training/
├── __init__.py
└── multi_gpu.py          # All multi-GPU utilities
```

## API Reference

### `setup_multi_gpu(model, gpu_ids=None)`

Wraps model in DataParallel if multiple GPUs available.

**Args:**
- `model`: nn.Module to wrap
- `gpu_ids`: List of GPU IDs to use. If None, uses all available.

**Returns:**
- DataParallel-wrapped model (if 2+ GPUs) or original model

**Behavior:**
1. If no CUDA available, returns model unchanged
2. If 1 GPU available, returns model on cuda:0
3. If 2+ GPUs available, wraps in DataParallel

**Example:**
```python
# Use all GPUs
model = setup_multi_gpu(model)

# Use specific GPUs
model = setup_multi_gpu(model, gpu_ids=[0, 2])
```

### `get_model_state_dict(model)`

Extract state dict, unwrapping DataParallel if needed.

**Args:**
- `model`: Model (possibly wrapped in DataParallel)

**Returns:**
- state_dict (unwrapped if necessary)

**Use Case:**
Always use this instead of `model.state_dict()` for checkpoint saving.

### `load_model_state_dict(model, state_dict)`

Load state dict, handling DataParallel wrapper.

**Args:**
- `model`: Model (possibly wrapped in DataParallel)
- `state_dict`: State dict to load

**Behavior:**
If model is DataParallel, loads into `model.module`.

### `detect_kaggle_multi_gpu()`

Detect if running on Kaggle with multiple GPUs.

**Returns:**
- True if Kaggle environment with 2+ GPUs detected

**Use Case:**
```python
if detect_kaggle_multi_gpu():
    print("Running on Kaggle with multiple GPUs!")
```

### `calculate_total_batch_size(batch_size_per_gpu, num_gpus)`

Calculate effective batch size.

**Args:**
- `batch_size_per_gpu`: Batch size per GPU
- `num_gpus`: Number of GPUs

**Returns:**
- Total batch size across all GPUs

### `print_gpu_info()`

Print information about available GPUs.

**Output Example:**
```
[gpu] 2 GPU(s) detected:
  GPU 0: Tesla T4
    Memory: 15.75 GB
    Compute: 7.5
  GPU 1: Tesla T4
    Memory: 15.75 GB
    Compute: 7.5
```

## Kaggle Integration

For Kaggle notebooks with 2x T4 GPUs:

```python
# In notebook settings, enable GPU (not TPU)
# Detect and setup automatically
from your_project.training.multi_gpu import detect_kaggle_multi_gpu, setup_multi_gpu

if detect_kaggle_multi_gpu():
    model = setup_multi_gpu(model)
    # Effective batch size doubles
    effective_batch_size = cfg.batch_size * 2
```

## CLI Integration

Add flags to your training CLI:

```python
@app.command()
def train(
    # ... other args ...
    multi_gpu: bool = typer.Option(False, "--multi-gpu", help="Enable multi-GPU training"),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs (e.g., '0,1')"),
):
    model = build_model(cfg.model)
    
    if multi_gpu:
        gpu_id_list = [int(x) for x in gpu_ids.split(",")] if gpu_ids else None
        model = setup_multi_gpu(model, gpu_id_list)
```

## Checkpoint Compatibility Table

| Save From | Load To | Works? | Notes |
|-----------|---------|--------|-------|
| Single GPU | Single GPU | ✓ | Standard case |
| Single GPU | Multi-GPU | ✓ | DataParallel handles automatically |
| Multi-GPU | Single GPU | ✓ | `get_model_state_dict()` unwraps |
| Multi-GPU | Multi-GPU | ✓ | Any GPU configuration |

## Common Pitfalls

1. **Don't call `model.module` directly**: Always use `get_model_state_dict()` and `load_model_state_dict()`
2. **Batch size confusion**: Specify per-GPU batch size; total = per_gpu × num_gpus
3. **DataLoader workers**: Can use more workers with multi-GPU (CPU not bottlenecked)
4. **Gradient accumulation**: Divide by num_gpus if simulating larger batches

---

```json
{
  "module_name": "multi_gpu",
  "package_path": "src/{PROJECT_NAME}/training/multi_gpu.py",
  "components": [
    {
      "name": "setup_multi_gpu",
      "signature": "setup_multi_gpu(model: nn.Module, gpu_ids: Optional[list[int]] = None) -> nn.Module",
      "description": "Wrap model in DataParallel if multiple GPUs available",
      "logic": [
        "If not torch.cuda.is_available(): return model",
        "Get available GPU count: torch.cuda.device_count()",
        "If count <= 1: return model (on cuda:0 if available)",
        "Validate gpu_ids against available GPUs",
        "Wrap in nn.DataParallel(model, device_ids=gpu_ids)",
        "Print GPU info",
        "Return wrapped model"
      ],
      "returns": "DataParallel-wrapped model or original model"
    },
    {
      "name": "get_model_state_dict",
      "signature": "get_model_state_dict(model: nn.Module) -> dict",
      "description": "Extract state dict, handling DataParallel wrapper",
      "logic": [
        "Check if model is instance of nn.DataParallel",
        "If yes: return model.module.state_dict()",
        "If no: return model.state_dict()"
      ],
      "use_case": "Always use for checkpoint saving"
    },
    {
      "name": "load_model_state_dict",
      "signature": "load_model_state_dict(model: nn.Module, state_dict: dict) -> None",
      "description": "Load state dict, handling DataParallel wrapper",
      "logic": [
        "Check if model is instance of nn.DataParallel",
        "If yes: model.module.load_state_dict(state_dict)",
        "If no: model.load_state_dict(state_dict)"
      ],
      "use_case": "Always use for checkpoint loading"
    },
    {
      "name": "detect_kaggle_multi_gpu",
      "signature": "detect_kaggle_multi_gpu() -> bool",
      "description": "Detect Kaggle environment with multiple GPUs",
      "detection_logic": [
        "Check for /kaggle directory",
        "Check KAGGLE_KERNEL_RUN_TYPE env var",
        "Check torch.cuda.device_count() >= 2"
      ],
      "returns": "True if Kaggle with 2+ GPUs"
    },
    {
      "name": "calculate_total_batch_size",
      "signature": "calculate_total_batch_size(batch_size_per_gpu: int, num_gpus: int) -> int",
      "description": "Calculate effective batch size across GPUs",
      "formula": "batch_size_per_gpu * num_gpus"
    },
    {
      "name": "print_gpu_info",
      "signature": "print_gpu_info() -> None",
      "description": "Print GPU information for debugging",
      "output": "GPU count, name, memory (GB), compute capability per device"
    }
  ],
  "integration_points": [
    {
      "location": "Model initialization",
      "code": "model = build_model(cfg.model); model = setup_multi_gpu(model)",
      "reason": "Enable multi-GPU before optimizer creation"
    },
    {
      "location": "Checkpoint saving",
      "code": "checkpoint = {'model': get_model_state_dict(model), 'optimizer': optimizer.state_dict()}; torch.save(checkpoint, path)",
      "reason": "Save unwrapped state dict for compatibility"
    },
    {
      "location": "Checkpoint loading",
      "code": "checkpoint = torch.load(path); load_model_state_dict(model, checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])",
      "reason": "Load into wrapped or unwrapped model"
    },
    {
      "location": "CLI argument parsing",
      "code": "multi_gpu: bool = typer.Option(False, '--multi-gpu'); gpu_ids: Optional[str] = typer.Option(None, '--gpu-ids')",
      "reason": "User control over GPU usage"
    },
    {
      "location": "Kaggle notebook setup",
      "code": "if detect_kaggle_multi_gpu(): print('Using 2x T4 GPUs'); model = setup_multi_gpu(model)",
      "reason": "Auto-enable on Kaggle"
    }
  ],
  "customization_variables": {
    "PROJECT_NAME": "Your package name",
    "DEFAULT_GPU_IDS": "None (use all) or specific list",
    "CLOUD_DETECTION": "Environment-specific detection logic"
  },
  "dependencies": ["torch", "os (for env detection)"],
  "smoke_tests": [
    {
      "name": "Single GPU Detection",
      "test": "model = nn.Linear(10, 10); result = setup_multi_gpu(model); # Should work regardless of GPU availability"
    },
    {
      "name": "State Dict Unwrapping",
      "test": "model = nn.Linear(10, 10); if torch.cuda.is_available(): model = nn.DataParallel(model.cuda()); sd = get_model_state_dict(model); assert 'module' not in sd.keys()"
    },
    {
      "name": "Checkpoint Compatibility",
      "test": "model1 = nn.Linear(10, 10); model2 = nn.Linear(10, 10); sd = get_model_state_dict(model1); load_model_state_dict(model2, sd); # Should work"
    },
    {
      "name": "Batch Size Calculation",
      "test": "total = calculate_total_batch_size(32, 2); assert total == 64"
    },
    {
      "name": "GPU Info Printing",
      "test": "print_gpu_info()  # Should not raise"
    }
  ]
}
```
