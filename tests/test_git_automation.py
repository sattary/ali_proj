"""
Tests for git_automation module.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from phase_unwrap.git_automation import AutoPushCallback
from phase_unwrap.git_automation.environment import is_cloud_environment
from phase_unwrap.git_automation.state_tracker import StateTracker
from phase_unwrap.git_automation.zip_packer import ZipPacker


class TestEnvironmentDetection:
    def test_is_cloud_environment_returns_bool(self):
        """Should return a boolean."""
        result = is_cloud_environment()
        assert isinstance(result, bool)


class TestZipPacker:
    def test_create_zip_structure(self):
        """Should create a properly structured zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock run directory with files
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()

            # Create some test files
            (run_dir / "config.yaml").write_text("test: config")
            (run_dir / "metrics.csv").write_text("epoch,loss\n1,0.5")
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "best.pth").write_text("fake checkpoint")

            packer = ZipPacker(str(run_dir))
            zip_path = packer.create_zip(epoch=100, total_epochs=1000)

            assert Path(zip_path).exists()

            # Verify zip contents
            with zipfile.ZipFile(zip_path, "r") as zf:
                files = zf.namelist()
                assert "config.yaml" in files
                assert "metrics.csv" in files
                assert "checkpoints/best.pth" in files
                assert "metadata.json" in files

                # Check metadata
                meta = json.loads(zf.read("metadata.json"))
                assert meta["epoch"] == 100
                assert meta["total_epochs"] == 1000


class TestStateTracker:
    def test_read_write_state(self):
        """Should correctly save and load state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StateTracker(tmpdir)

            # Write state
            tracker.write_state(
                epoch=500,
                total_epochs=1000,
                push_interval=100,
                last_push_epoch=500,
                best_mae=0.0423,
            )

            # Read state
            state = tracker.read_state()
            assert state is not None
            assert state["epoch"] == 500
            assert state["total_epochs"] == 1000
            assert state["push_interval"] == 100
            assert state["best_mae"] == 0.0423

    def test_should_push(self):
        """Should correctly determine if push is needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StateTracker(tmpdir)

            # Write initial state
            tracker.write_state(
                epoch=500,
                total_epochs=1000,
                push_interval=100,
                last_push_epoch=500,
                best_mae=0.0423,
            )

            # Should not push at epoch 550
            assert not tracker.should_push(550)

            # Should push at epoch 600 (500 + 100)
            assert tracker.should_push(600)

    def test_get_resume_epoch(self):
        """Should return correct resume epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StateTracker(tmpdir)

            # No state - start from 1
            assert tracker.get_resume_epoch() == 1

            # Write state at epoch 500
            tracker.write_state(
                epoch=500,
                total_epochs=1000,
                push_interval=100,
                last_push_epoch=500,
                best_mae=0.0423,
            )

            # Should resume from 501
            assert tracker.get_resume_epoch() == 501


class TestAutoPushCallback:
    def _create_mock_repo(self, base_dir: str) -> str:
        """Create a mock git repository for testing."""
        import subprocess

        repo_dir = Path(base_dir) / "mock_repo"
        repo_dir.mkdir()

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        # Configure git user
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        return str(repo_dir)

    def test_dry_run_mode(self):
        """Should not make actual git calls in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self._create_mock_repo(tmpdir)
            run_dir = Path(repo_dir) / "test_run"
            run_dir.mkdir()

            callback = AutoPushCallback(
                run_dir=str(run_dir),
                push_interval=100,
                dry_run=True,
                force=True,  # Bypass environment check
                repo_dir=repo_dir,
            )

            # Should initialize without error
            callback.initialize()
            assert callback._initialized

            # Should return True for dry-run push
            result = callback.on_epoch_end(epoch=100, total_epochs=1000)
            assert result is True

    def test_callback_not_initialized_without_interval(self):
        """Should not push if interval is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self._create_mock_repo(tmpdir)
            run_dir = Path(repo_dir) / "test_run"
            run_dir.mkdir()

            callback = AutoPushCallback(
                run_dir=str(run_dir),
                push_interval=100,
                dry_run=True,
                force=True,
                repo_dir=repo_dir,
            )

            # Epoch not on interval - should not push
            result = callback.on_epoch_end(epoch=50, total_epochs=1000)
            assert result is False  # Not on interval

            # Epoch on interval - should push (in dry-run)
            result = callback.on_epoch_end(epoch=100, total_epochs=1000)
            assert result is True

    def test_final_push(self):
        """Should push on final epoch regardless of interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self._create_mock_repo(tmpdir)
            run_dir = Path(repo_dir) / "test_run"
            run_dir.mkdir()

            callback = AutoPushCallback(
                run_dir=str(run_dir),
                push_interval=100,
                dry_run=True,
                force=True,
                repo_dir=repo_dir,
            )

            # Final epoch (1000) should trigger push even if not on interval
            result = callback.on_epoch_end(epoch=1000, total_epochs=1000)
            assert result is True
