"""
Tests for multi-GPU training support.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from phase_unwrap.training.multi_gpu import (
    calculate_total_batch_size,
    detect_kaggle_multi_gpu,
    get_model_state_dict,
    load_model_state_dict,
    print_gpu_info,
    setup_multi_gpu,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestSetupMultiGPU:
    def test_single_gpu_no_wrap(self):
        """Should not wrap model if only 1 GPU."""
        model = SimpleModel()

        with patch("torch.cuda.device_count", return_value=1):
            result = setup_multi_gpu(model)
            assert result is model
            assert not isinstance(result, nn.DataParallel)

    def test_no_gpu_no_wrap(self):
        """Should not wrap model if no GPU."""
        model = SimpleModel()

        with patch("torch.cuda.is_available", return_value=False):
            result = setup_multi_gpu(model)
            assert result is model
            assert not isinstance(result, nn.DataParallel)

    def test_multi_gpu_wrap(self):
        """Should wrap model in DataParallel if multiple GPUs."""
        model = SimpleModel()

        # Mock CUDA is_initialized to return True (already initialized)
        with patch("torch.cuda.is_initialized", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
                        # Mock _check_balance to avoid device properties lookup
                        with patch("torch.nn.parallel.data_parallel._check_balance"):
                            result = setup_multi_gpu(model)
                            assert isinstance(result, nn.DataParallel)
                            assert result.device_ids == [0, 1]

    def test_specific_gpu_ids(self):
        """Should use specific GPU IDs if provided."""
        model = SimpleModel()

        with patch("torch.cuda.is_initialized", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
                        with patch("torch.nn.parallel.data_parallel._check_balance"):
                            result = setup_multi_gpu(model, gpu_ids=[0, 2])
                            assert isinstance(result, nn.DataParallel)
                            assert result.device_ids == [0, 2]

    def test_invalid_gpu_ids(self):
        """Should raise error for invalid GPU IDs."""
        model = SimpleModel()

        with patch("torch.cuda.device_count", return_value=2):
            with patch("torch.cuda.is_available", return_value=True):
                with pytest.raises(ValueError):
                    setup_multi_gpu(model, gpu_ids=[0, 5])  # 5 is invalid


class TestStateDictHandling:
    def test_get_state_dict_single_gpu(self):
        """Should get state dict from regular model."""
        model = SimpleModel()
        state_dict = get_model_state_dict(model)
        assert "fc.weight" in state_dict
        assert "fc.bias" in state_dict

    def test_get_state_dict_multi_gpu(self):
        """Should get state dict from DataParallel model."""
        model = SimpleModel()
        dp_model = nn.DataParallel(model)

        state_dict = get_model_state_dict(dp_model)
        assert "fc.weight" in state_dict
        assert "fc.bias" in state_dict

    def test_load_state_dict_single_gpu(self):
        """Should load state dict into regular model."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Copy weights from model1 to model2
        load_model_state_dict(model2, model1.state_dict())

        # Check weights match
        assert torch.allclose(model1.fc.weight, model2.fc.weight)
        assert torch.allclose(model1.fc.bias, model2.fc.bias)

    def test_load_state_dict_multi_gpu(self):
        """Should load state dict into DataParallel model."""
        model = SimpleModel()
        dp_model = nn.DataParallel(SimpleModel())

        load_model_state_dict(dp_model, model.state_dict())

        # Check underlying model weights
        assert torch.allclose(model.fc.weight, dp_model.module.fc.weight)


class TestKaggleDetection:
    def test_detect_kaggle_with_multi_gpu(self):
        """Should detect Kaggle with multiple GPUs."""
        with patch("os.path.exists", return_value=True):  # /kaggle exists
            with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "interactive"}):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.device_count", return_value=2):
                        with patch(
                            "torch.cuda.get_device_name",
                            return_value="Tesla T4",
                        ):
                            result = detect_kaggle_multi_gpu()
                            assert result is True

    def test_not_kaggle(self):
        """Should not detect if not Kaggle."""
        with patch("os.path.exists", return_value=False):
            with patch.dict("os.environ", {}, clear=True):
                result = detect_kaggle_multi_gpu()
                assert result is False

    def test_single_gpu_kaggle(self):
        """Should not detect if Kaggle but only 1 GPU."""
        with patch("os.path.exists", return_value=True):
            with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "interactive"}):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.device_count", return_value=1):
                        result = detect_kaggle_multi_gpu()
                        assert result is False


class TestUtilities:
    def test_calculate_total_batch_size(self):
        """Should calculate total batch size correctly."""
        assert calculate_total_batch_size(20, 2) == 40
        assert calculate_total_batch_size(32, 4) == 128
        assert calculate_total_batch_size(16, 1) == 16

    def test_print_gpu_info_no_gpu(self, capsys):
        """Should print no GPUs message."""
        with patch("torch.cuda.is_available", return_value=False):
            print_gpu_info()
            captured = capsys.readouterr()
            assert "No GPUs available" in captured.out

    def test_print_gpu_info_with_gpus(self, capsys):
        """Should print GPU information."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    # Mock GPU properties
                    props = MagicMock()
                    props.name = "Tesla T4"
                    props.total_memory = 16106127360  # 15 GB
                    props.major = 7
                    props.minor = 5
                    mock_props.return_value = props

                    print_gpu_info()
                    captured = capsys.readouterr()
                    assert "2 GPU(s) detected" in captured.out
                    assert "Tesla T4" in captured.out
