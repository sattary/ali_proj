"""
Tests for model architecture.
"""

from __future__ import annotations

import torch

from phase_unwrap.model import build_model, EMA


class TestUNetRes2:
    def test_output_shapes(self, default_config):
        model = build_model(default_config.model)
        model.eval()
        x = torch.randn(2, 2, 128, 128)
        phi_raw, k_off = model(x)

        assert phi_raw.shape == (2, 1, 128, 128), f"phi_raw: {phi_raw.shape}"
        assert k_off.shape == (2, 1, 1, 1), f"k_off: {k_off.shape}"

    def test_different_spatial_sizes(self, default_config):
        """Model should handle any spatial size divisible by 16."""
        model = build_model(default_config.model)
        model.eval()
        for size in [64, 128, 256]:
            x = torch.randn(1, 2, size, size)
            phi_raw, k_off = model(x)
            assert phi_raw.shape == (1, 1, size, size)

    def test_deterministic(self, default_config):
        model = build_model(default_config.model)
        model.eval()
        x = torch.randn(1, 2, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1[0], out2[0])
        torch.testing.assert_close(out1[1], out2[1])

    def test_gradient_flow(self, default_config):
        """Verify gradients propagate to all parameters."""
        model = build_model(default_config.model)
        model.train()
        x = torch.randn(1, 2, 128, 128)
        phi_raw_list, k_off = model(x)
        loss = sum(p.mean() for p in phi_raw_list) + k_off.mean()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestEMA:
    def test_ema_diverges(self, default_config):
        """EMA model should diverge from base after training steps."""
        model = build_model(default_config.model)
        ema = EMA(model, decay=0.9)

        # simulate a training step
        x = torch.randn(1, 2, 128, 128)
        phi_raw_list, k_off = model(x)
        (phi_raw_list[2].mean() + k_off.mean()).backward()

        # manually change a parameter
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(model)

        # EMA should be different from model
        for p_ema, p in zip(ema.m.parameters(), model.parameters()):
            assert not torch.allclose(p_ema, p, atol=1e-6)


class TestPCLCNModel:
    def test_pclcn_model_forward(self):
        from phase_unwrap.model.unet import PCLCNModel
        model = PCLCNModel(height=128, width=128, base=16)
        I_raw = torch.rand(2, 1, 128, 128)
        grad_phi2 = torch.randn(2, 2, 128, 128)

        phi_final, gx_tilde, gy_tilde, c_zernike, phi_zernike = model(I_raw, grad_phi2)

        assert phi_final.shape == (2, 1, 128, 128)
        assert gx_tilde.shape == (2, 1, 128, 128)
        assert gy_tilde.shape == (2, 1, 128, 128)
        assert c_zernike.shape == (2, 15)
        assert phi_zernike.shape == (2, 1, 128, 128)

    def test_pclcn_backward(self):
        from phase_unwrap.model.unet import PCLCNModel
        model = PCLCNModel(height=128, width=128, base=16)
        I_raw = torch.rand(2, 1, 128, 128)
        grad_phi2 = torch.randn(2, 2, 128, 128)

        phi_final, gx_tilde, gy_tilde, _, _ = model(I_raw, grad_phi2)
        loss = phi_final.mean() + gx_tilde.mean() + gy_tilde.mean()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

