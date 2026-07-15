"""
Tests for losses and ops.
"""

from __future__ import annotations

import torch
import pytest

from phase_unwrap.core.losses import MAEGradLoss, compute_metrics
from phase_unwrap.core.ops import FixedSobel, affine_align, curvature_loss, piston_align


class TestComputeMetrics:
    def test_perfect_prediction(self):
        phi = torch.randn(2, 1, 32, 32)
        m = compute_metrics(phi, phi)
        assert m["MAE"].item() == pytest.approx(0.0, abs=1e-6)
        assert m["RMSE"].item() == pytest.approx(0.0, abs=1e-3)

    def test_known_error(self):
        gt = torch.zeros(1, 1, 32, 32)
        pred = torch.ones(1, 1, 32, 32) * 2.0
        m = compute_metrics(pred, gt)
        assert m["MAE"].item() == pytest.approx(2.0, abs=1e-5)
        assert m["RMSE"].item() == pytest.approx(2.0, abs=1e-5)


class TestMAEGradLoss:
    def test_zero_on_perfect(self):
        loss_fn = MAEGradLoss(w_mae=1.0, w_grad=1.0)
        phi = torch.randn(2, 1, 32, 32)
        total, parts = loss_fn(phi, phi)
        assert total.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_exists(self):
        loss_fn = MAEGradLoss(w_mae=1.0, w_grad=0.1)
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        gt = torch.randn(2, 1, 32, 32)
        total, _ = loss_fn(pred, gt)
        total.backward()
        assert pred.grad is not None

    def test_intensity_weighted(self):
        loss_fn = MAEGradLoss(w_mae=1.0, w_grad=0.1, intensity_weighted=True)
        pred = torch.randn(2, 1, 32, 32)
        gt = torch.randn(2, 1, 32, 32)
        I_raw = torch.rand(2, 1, 32, 32) + 0.1
        total, parts = loss_fn(pred, gt, I_raw)
        assert total.item() > 0


class TestAffineAlign:
    def test_identity(self):
        """If pred == gt, alignment should return a=1, c=0."""
        phi = torch.randn(2, 1, 32, 32)
        aligned, a, c = affine_align(phi, phi)
        torch.testing.assert_close(aligned, phi, atol=1e-5, rtol=1e-5)
        assert a.mean().item() == pytest.approx(1.0, abs=0.01)
        assert c.mean().item() == pytest.approx(0.0, abs=0.01)

    def test_known_affine(self):
        """If pred = 2*gt + 5, alignment should recover gt."""
        gt = torch.randn(4, 1, 32, 32)
        pred = 2.0 * gt + 5.0
        aligned, a, c = affine_align(pred, gt)
        torch.testing.assert_close(aligned, gt, atol=1e-4, rtol=1e-4)

    def test_batch_independence(self):
        """Each batch element should be aligned independently."""
        gt = torch.randn(4, 1, 32, 32)
        pred = gt.clone()
        pred[0] = pred[0] * 3.0 + 10.0  # different transform for first
        pred[1] = pred[1] * 0.5 - 2.0
        aligned, _, _ = affine_align(pred, gt)
        torch.testing.assert_close(aligned, gt, atol=1e-4, rtol=1e-4)


class TestPistonAlign:
    def test_does_not_hide_scale_error(self):
        gt = torch.randn(2, 1, 16, 16)
        half = 0.5 * gt
        aff, a, _ = affine_align(half, gt)
        assert (aff - gt).abs().mean() < 1e-3
        assert torch.allclose(a, torch.full_like(a, 2.0), atol=1e-2)
        pist, _ = piston_align(half, gt)
        assert (pist - gt).abs().mean() > 0.1

    def test_removes_offset(self):
        gt = torch.randn(2, 1, 16, 16)
        shifted = gt + 3.7
        aligned, c = piston_align(shifted, gt)
        assert (aligned - gt).abs().mean() < 1e-5
        assert torch.allclose(c.flatten(), torch.full((2,), -3.7), atol=1e-4)


class TestFixedSobel:
    def test_output_shapes(self):
        sobel = FixedSobel()
        x = torch.randn(2, 1, 32, 32)
        gx, gy = sobel(x)
        assert gx.shape == (2, 1, 32, 32)
        assert gy.shape == (2, 1, 32, 32)

    def test_constant_input(self):
        """Gradient of constant field should be ~0."""
        sobel = FixedSobel()
        x = torch.ones(1, 1, 32, 32) * 5.0
        gx, gy = sobel(x)
        assert gx.abs().max().item() < 1e-5
        assert gy.abs().max().item() < 1e-5


class TestCurvatureLoss:
    def test_smooth_field(self):
        """Linear field has zero Laplacian."""
        x = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
        loss = curvature_loss(x)
        assert loss.item() < 0.01  # approximately zero
