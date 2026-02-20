# Master Prompts for Deep Learning Projects

Production-ready architectural patterns for reusable DL modules.

## Prompts Overview

| # | Prompt | Purpose | Complexity | Reusability |
|---|--------|---------|------------|-------------|
| 1 | [git_automation](git_automation_master_prompt.md) | Auto-push to GitHub during cloud training | High | Any cloud training project |
| 2 | [hierarchical_config](hierarchical_config_master_prompt.md) | Type-safe config with JSON/YAML support | Medium | Universal for any project |
| 3 | [visualization_suite](visualization_suite_master_prompt.md) | Publication-quality plots (Nature journal style) | High | Any DL research project |
| 4 | [multi_gpu](multi_gpu_master_prompt.md) | DataParallel with seamless checkpoint handling | Low | Any PyTorch multi-GPU project |

## Quick Start

1. **Copy** the prompt(s) you need
2. **Customize** variables (PROJECT_NAME, metric names, etc.)
3. **Give to AI agent** for implementation
4. **Run smoke tests** to verify

## Usage Example

```markdown
Implement a git automation system for my project called "medical_imaging".
The main metric is "Dice Score" and checkpoints use ".pth" extension.

---
[Paste full git_automation_master_prompt.md here]
```

## Prompt Structure

Each prompt contains:

1. **Plain Text Description**
   - Architecture philosophy
   - Design decisions
   - Integration patterns
   - Code examples

2. **JSON Specification**
   - Module name and path
   - Components with signatures
   - Integration points
   - Customization variables
   - Smoke tests

## Key Features

- **Callback Pattern**: git_automation, visualization_suite
- **Self-Contained**: Each plot handles its own data loading
- **Transparent**: Multi-GPU doesn't change training code
- **Type-Safe**: Hierarchical config uses dataclasses
- **Publication-Ready**: Nature journal aesthetics built-in

## Dependencies

### git_automation
- git (CLI)
- typer (optional, for CLI)

### hierarchical_config
- pyyaml (optional, for YAML support)

### visualization_suite
- matplotlib
- seaborn
- scipy
- numpy

### multi_gpu
- torch

## Common Customization Variables

```json
{
  "PROJECT_NAME": "your_package_name",
  "PRIMARY_METRIC": "MAE|Accuracy|Dice|mIoU|etc",
  "METRIC_UNITS": "rad|dB|%|mm|etc",
  "CHECKPOINT_EXTENSION": ".pth|.ckpt|.pt",
  "DATASET_EXTENSION": ".h5|.tfrecord|.npy",
  "INPUT_KEY": "I|image|input|x",
  "TARGET_KEY": "phi|label|target|y"
}
```

## Smoke Test Philosophy

Each prompt includes 5 basic tests:
- Component instantiation
- Core method calls
- File I/O operations
- Integration points
- Edge cases

**Not included:** Full test suites (you should write those for your specific domain).

## Next Prompts (Planned)

Potential additions:
- **Early Stopping**: Patience-based with checkpoint saving
- **LR Scheduling**: Cosine annealing with warmup
- **Experiment Tracking**: W&B/MLflow integration layer
- **Data Augmentation**: Compose transforms with visualization
- **Model Export**: ONNX/TorchScript with benchmarking

## License

These prompts are reference implementations. Use them freely in your projects.

---

Created from production codebase: phase_unwrap deep learning project.
