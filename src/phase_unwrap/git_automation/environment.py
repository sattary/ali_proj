"""
Environment detection for cloud platforms.
"""

import os


def is_kaggle() -> bool:
    """Detect if running in Kaggle environment."""
    return (
        os.path.exists("/kaggle")
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.environ.get("KAGGLE_CONTAINER_NAME") is not None
    )


def is_colab() -> bool:
    """Detect if running in Google Colab environment."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def is_cloud_environment() -> bool:
    """Detect if running in Kaggle or Colab."""
    return is_kaggle() or is_colab()


def get_environment_name() -> str:
    """Get the name of the cloud environment."""
    if is_kaggle():
        return "kaggle"
    elif is_colab():
        return "colab"
    else:
        return "local"


def verify_cloud_environment(force: bool = False) -> None:
    """
    Verify that we're running in a cloud environment.

    Args:
        force: If True, skip verification (for testing).

    Raises:
        RuntimeError: If not in cloud environment and force=False.
    """
    if force:
        return

    if not is_cloud_environment():
        env = get_environment_name()
        raise RuntimeError(
            f"Auto-push is only available in Kaggle or Colab environments. "
            f"Detected environment: {env}. "
            f"Use --force-auto-push to override (for testing only)."
        )