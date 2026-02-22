"""
State tracking for resume functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class StateTracker:
    """Track push state for resume functionality."""

    STATE_FILE = ".push_state.json"

    def __init__(self, run_dir: str):
        """
        Initialize state tracker.

        Args:
            run_dir: Path to the training run directory.
        """
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / self.STATE_FILE

    def read_state(self) -> Optional[dict]:
        """
        Read the current state.

        Returns:
            State dict or None if no state file exists.
        """
        if not self.state_path.exists():
            return None

        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def write_state(
        self,
        epoch: int,
        total_epochs: int,
        push_interval: int,
        last_push_epoch: int,
        best_mae: float,
        zip_path: Optional[str] = None,
    ) -> None:
        """
        Write the current state.

        Args:
            epoch: Current epoch.
            total_epochs: Total epochs.
            push_interval: Push interval setting.
            last_push_epoch: Last epoch that was pushed.
            best_mae: Best MAE so far.
            zip_path: Path to the last zip file.
        """
        state = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "push_interval": push_interval,
            "last_push_epoch": last_push_epoch,
            "best_mae": best_mae,
            "last_updated": datetime.now().isoformat(),
            "zip_path": zip_path,
        }

        self.state_path.write_text(json.dumps(state, indent=2))

    def should_push(self, current_epoch: int) -> bool:
        """
        Check if we should push at this epoch.

        Args:
            current_epoch: Current epoch number.

        Returns:
            True if push should happen.
        """
        state = self.read_state()
        if state is None:
            # No state yet, check if it's first interval
            return False  # Will be handled by callback initialization

        interval = state.get("push_interval", 0)
        last_push = state.get("last_push_epoch", 0)

        if interval <= 0:
            return False

        return (current_epoch - last_push) >= interval

    def get_resume_epoch(self) -> int:
        """
        Get the epoch to resume from.

        Returns:
            Epoch number to resume from (1 if no state).
        """
        state = self.read_state()
        if state is None:
            return 1

        return state.get("epoch", 1) + 1

    def is_finished(self) -> bool:
        """Check if training was completed."""
        state = self.read_state()
        if state is None:
            return False

        return state.get("epoch", 0) >= state.get("total_epochs", 0)

    def validate_consistency(
        self, expected_total_epochs: int, expected_interval: int
    ) -> bool:
        """
        Validate that state matches expected configuration.

        Args:
            expected_total_epochs: Expected total epochs.
            expected_interval: Expected push interval.

        Returns:
            True if consistent.
        """
        state = self.read_state()
        if state is None:
            return True  # No state to validate

        if state.get("total_epochs") != expected_total_epochs:
            print(
                f"Warning: State total_epochs ({state.get('total_epochs')}) "
                f"doesn't match expected ({expected_total_epochs})"
            )
            return False

        if state.get("push_interval") != expected_interval:
            print(
                f"Warning: State push_interval ({state.get('push_interval')}) "
                f"doesn't match expected ({expected_interval})"
            )
            return False

        return True