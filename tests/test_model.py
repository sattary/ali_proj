"""
Tests for PCLCN model architecture.
"""

from __future__ import annotations

import torch

from phase_unwrap.model import PCLCNModel, EMA, build_model


class TestPCLCNModel:
    def test_pclcn_model_forward(self, default_config):
        model = build_model(default_config.model)
        model.eval()
        I_raw = torch.rand(2, 1, 128, 128)
        grad_phi2 = torch.randn(2, 2, 128, 128)

        phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)

        assert phi_final.shape == (2, 1, 128, 128)
        assert gx_tilde.shape == (2, 1, 128, 128)
        assert gy_tilde.shape == (2, 1, 128, 128)
        assert c_zernike.shape == (2, 15)
        assert phi_zernike.shape == (2, 1, 128, 128)

    def test_pclcn_backward(self):
        model = PCLCNModel(height=128, width=128, base=16)
        I_raw = torch.rand(2, 1, 128, 128)
        grad_phi2 = torch.randn(2, 2, 128, 128)

        phi_final, gx_tilde, gy_tilde, _, _ = model(I_raw, grad_phi2)
        loss = phi_final.mean() + gx_tilde.mean() + gy_tilde.mean()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestEMA:
    def test_ema_diverges(self, default_config):
        """EMA model should diverge from base after training steps."""
        model = build_model(default_config.model)
        ema = EMA(model, decay=0.9)

        # simulate a training step
        I_raw = torch.rand(1, 1, 128, 128)
        grad_phi2 = torch.randn(1, 2, 128, 128)
        phi_final, _, _, _, _ = model(I_raw, grad_phi2)
        phi_final.mean().backward()

        # manually change a parameter
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(model)

        # EMA should be different from model
        for p_ema, p in zip(ema.m.parameters(), model.parameters()):
            assert not torch.allclose(p_ema, p, atol=1e-6)
