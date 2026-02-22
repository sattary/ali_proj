"""
CLI integration for auto-push arguments.
"""

from typing import Optional

import typer

from .callback import AutoPushCallback
from .environment import is_cloud_environment


def add_auto_push_args():
    """
    Decorator to add auto-push arguments to a CLI command.

    Usage:
        @app.command()
        @add_auto_push_args()
        def train(...):
            pass
    """

    def decorator(func):
        # Add --auto-push-interval
        func = typer.Option(
            None,
            "--auto-push-interval",
            help="Enable auto-push every N epochs (Kaggle/Colab only).",
        )(func)

        # Add --auto-push-dry-run
        func = typer.Option(
            False,
            "--auto-push-dry-run",
            help="Test auto-push setup without actually pushing.",
        )(func)

        # Add --force-auto-push
        func = typer.Option(
            False,
            "--force-auto-push",
            help="Force auto-push even outside Kaggle/Colab (for testing).",
        )(func)

        # Add --auto-push-pat
        func = typer.Option(
            None,
            "--auto-push-pat",
            help="GitHub PAT (or set GITHUB_PAT env var).",
        )(func)

        return func

    return decorator


def validate_auto_push_config(
    auto_push_interval: Optional[int],
    auto_push_dry_run: bool,
    force_auto_push: bool,
) -> bool:
    """
    Validate auto-push configuration.

    Args:
        auto_push_interval: The push interval value.
        auto_push_dry_run: Whether dry-run is enabled.
        force_auto_push: Whether to force enable.

    Returns:
        True if auto-push should be enabled.

    Raises:
        RuntimeError: If configuration is invalid.
    """
    if auto_push_interval is None:
        return False

    if auto_push_interval <= 0:
        raise RuntimeError(
            f"--auto-push-interval must be positive, got {auto_push_interval}"
        )

    if not force_auto_push and not auto_push_dry_run:
        if not is_cloud_environment():
            raise RuntimeError(
                "Auto-push is only available in Kaggle or Colab. "
                "Use --auto-push-dry-run to test, or --force-auto-push to override."
            )

    return True


def create_auto_push_callback(
    run_dir: str,
    auto_push_interval: int,
    auto_push_dry_run: bool = False,
    force_auto_push: bool = False,
    auto_push_pat: Optional[str] = None,
    include_checkpoints: bool = True,
) -> Optional[AutoPushCallback]:
    """
    Create an AutoPushCallback if auto-push is enabled.

    Args:
        run_dir: Path to the training run directory.
        auto_push_interval: Push interval (epochs).
        auto_push_dry_run: Test mode.
        force_auto_push: Force enable.
        auto_push_pat: GitHub PAT.
        include_checkpoints: Include .pth files.

    Returns:
        AutoPushCallback instance or None if not enabled.
    """
    if auto_push_interval is None:
        return None

    return AutoPushCallback(
        run_dir=run_dir,
        push_interval=auto_push_interval,
        pat=auto_push_pat,
        dry_run=auto_push_dry_run,
        force=force_auto_push,
        include_checkpoints=include_checkpoints,
    )