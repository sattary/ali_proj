import random
import numpy as np
import pytest
import torch

from phase_unwrap.core.utils import set_seed
from phase_unwrap.data.dataset import _seed_worker


def test_set_seed_reproducibility():
    """Test that set_seed effectively resets RNG state across all libraries."""
    # Draw 1
    set_seed(123)
    t1 = torch.rand(4)
    n1 = np.random.rand(4)
    r1 = random.random()

    # Interfere
    torch.rand(4)
    np.random.rand(4)
    random.random()

    # Draw 2
    set_seed(123)
    t2 = torch.rand(4)
    n2 = np.random.rand(4)
    r2 = random.random()

    assert torch.allclose(t1, t2)
    assert np.allclose(n1, n2)
    assert r1 == r2


def test_set_seed_deterministic():
    """Test that deterministic=True runs without raising and disables benchmark."""
    set_seed(123, deterministic=True)
    assert torch.backends.cudnn.benchmark is False
    # If deterministic algorithms are unavailable or raise, this test would fail.


def test_seed_worker_distinctness():
    """Test that _seed_worker creates distinct states per worker and is reproducible."""
    torch.manual_seed(42)  # Set base seed
    
    # Worker 0
    _seed_worker(0)
    w0_a = random.random()
    
    # Worker 1
    _seed_worker(1)
    w1 = random.random()
    
    assert w0_a != w1, "Workers 0 and 1 should have different RNG states"
    
    # Worker 0 again (should be reproducible if base seed hasn't changed)
    # Note: torch.initial_seed() does not change just by calling rand(), 
    # but let's re-seed just to be safe it's exactly the same environment.
    torch.manual_seed(42)
    _seed_worker(0)
    w0_b = random.random()
    
    assert w0_a == w0_b, "_seed_worker should be reproducible given same base seed"
