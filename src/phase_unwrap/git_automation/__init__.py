"""
Git Automation for Phase Unwrap - Cloud Training Support.

Enables auto-push to GitHub when training in Kaggle/Colab environments.
Features:
- Environment detection (Kaggle/Colab only)
- Automatic zipping of results
- Git LFS support for large files
- Resume state tracking
- Dry-run mode for testing
"""

import os

# Force non-interactive matplotlib backend before any imports
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")

from .callback import AutoPushCallback
from .cli_integration import add_auto_push_args, validate_auto_push_config
from .environment import is_cloud_environment
from .git_pusher import GitPusher
from .state_tracker import StateTracker
from .zip_packer import ZipPacker

__all__ = [
    "AutoPushCallback",
    "GitPusher",
    "StateTracker",
    "ZipPacker",
    "is_cloud_environment",
    "add_auto_push_args",
    "validate_auto_push_config",
]