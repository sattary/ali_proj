"""
Tests for data generation pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from phase_unwrap.data.generate import NX, NY, _build_grid, generate_sample, generate_to_h5


class TestGenerateSample:
    def test_output_shapes(self):
        x, y, r2 = _build_grid()
        rng = np.random.default_rng(42)
        I, dphi, (g2x, g2y) = generate_sample(x, y, r2, rng)

        assert I.shape == (NY, NX), f"I shape: {I.shape}"
        assert dphi.shape == (NY, NX), f"dphi shape: {dphi.shape}"
        assert g2x.shape == (NY, NX) and g2y.shape == (NY, NX)

    def test_output_dtype(self):
        x, y, r2 = _build_grid()
        rng = np.random.default_rng(42)
        I, dphi, (g2x, g2y) = generate_sample(x, y, r2, rng)

        assert I.dtype == np.float32
        assert dphi.dtype == np.float32
        assert g2x.dtype == np.float32 and g2y.dtype == np.float32

    def test_intensity_positive(self):
        """Interferogram intensity = |E1+E2|^2 must be non-negative."""
        x, y, r2 = _build_grid()
        rng = np.random.default_rng(42)
        for _ in range(10):
            I, _, _ = generate_sample(x, y, r2, rng)
            assert I.min() >= 0.0, "Intensity should be non-negative"

    def test_phase_min_zero(self):
        """Ground truth phase should be shifted so min = 0."""
        x, y, r2 = _build_grid()
        rng = np.random.default_rng(42)
        for _ in range(10):
            _, dphi, _ = generate_sample(x, y, r2, rng)
            assert abs(dphi.min()) < 1e-6, f"Phase min should be ~0, got {dphi.min()}"

    def test_reproducibility(self):
        x, y, r2 = _build_grid()
        I1, phi1, g1 = generate_sample(x, y, r2, np.random.default_rng(42))
        I2, phi2, g2 = generate_sample(x, y, r2, np.random.default_rng(42))
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_array_equal(phi1, phi2)
        np.testing.assert_array_equal(g1[0], g2[0])


class TestGenerateToH5:
    def test_shard_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_to_h5(tmpdir, num_samples=10, shard_size=5, seed=1)
            shards = sorted(Path(tmpdir).glob("*.h5"))
            assert len(shards) == 2

    def test_shard_contents(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_to_h5(tmpdir, num_samples=5, shard_size=5, seed=1)
            shard = sorted(Path(tmpdir).glob("*.h5"))[0]
            with h5py.File(shard, "r") as f:
                assert "I" in f
                assert "phi" in f
                assert "grad_phi2" in f
                assert f["I"].shape == (5, 1, NY, NX)
                assert f["phi"].shape == (5, 1, NY, NX)
                assert f["grad_phi2"].shape == (5, 2, NY, NX)

