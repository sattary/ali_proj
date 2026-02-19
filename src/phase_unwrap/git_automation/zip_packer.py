"""
Zip packing for experiment artifacts.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class ZipPacker:
    """Pack experiment artifacts into a zip file."""

    def __init__(
        self,
        run_dir: str,
        compression_level: int = 6,
        include_checkpoints: bool = True,
    ):
        """
        Initialize zip packer.

        Args:
            run_dir: Path to the training run directory.
            compression_level: Zip compression level (0-9).
            include_checkpoints: Whether to include .pth files.
        """
        self.run_dir = Path(run_dir)
        self.compression_level = compression_level
        self.include_checkpoints = include_checkpoints

        # Files to exclude (as L3 engineer decision)
        self.exclude_patterns = [
            "*.pyc",
            "__pycache__",
            ".pytest_cache",
            "*.log",  # Raw logs can be huge
            "visuals/*",  # Skip intermediate visuals, keep final
        ]

    def _should_include(self, file_path: Path) -> bool:
        """Check if file should be included in zip."""
        rel_path = file_path.relative_to(self.run_dir)
        str_path = str(rel_path)

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.endswith("/*"):
                if str_path.startswith(pattern[:-2]):
                    return False
            elif pattern.startswith("*"):
                if str_path.endswith(pattern[1:]):
                    return False
            elif pattern in str_path:
                return False

        # Check checkpoints
        if file_path.suffix == ".pth" and not self.include_checkpoints:
            return False

        return True

    def _collect_files(self) -> List[Path]:
        """Collect all files to include in the zip."""
        files = []

        for item in self.run_dir.rglob("*"):
            if item.is_file() and self._should_include(item):
                files.append(item)

        return sorted(files)

    def create_zip(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Create a zip file of the experiment artifacts.

        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            metrics: Optional metrics dict to include.
            output_dir: Where to save the zip (default: run_dir).

        Returns:
            Path to the created zip file.
        """
        if output_dir is None:
            output_dir = self.run_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate zip filename
        run_name = self.run_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{run_name}_epoch_{epoch}_of_{total_epochs}_{timestamp}.zip"
        zip_path = output_path / zip_name

        # Collect files
        files_to_zip = self._collect_files()

        # Create zip
        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(
            zip_path, "w", compression=compression, compresslevel=self.compression_level
        ) as zf:
            for file_path in files_to_zip:
                arcname = file_path.relative_to(self.run_dir)
                zf.write(file_path, arcname)

            # Add metadata
            metadata = {
                "run_name": run_name,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "timestamp": timestamp,
                "metrics": metrics or {},
                "file_count": len(files_to_zip),
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        return str(zip_path)

    def get_latest_zip(self) -> Optional[str]:
        """Get path to the most recent zip file in the run directory."""
        zip_files = list(self.run_dir.glob("*.zip"))
        if not zip_files:
            return None

        # Sort by modification time
        latest = max(zip_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
