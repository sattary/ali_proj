"""Option B hint channel: zero by default, no GT leak unless ablation."""

from __future__ import annotations

import pytest
import torch

from phase_unwrap.data.augmentation import build_phi_hint, prepare_batch


class TestBuildPhiHint:
    def test_zero_is_zeros(self) -> None:
        ref = torch.randn(2, 1, 16, 16)
        h = build_phi_hint(ref, phi_gt=None, hint_mode="zero")
        assert h.shape == ref.shape
        assert torch.count_nonzero(h) == 0

    def test_gt_center_broadcast(self) -> None:
        phi = torch.randn(2, 1, 16, 16)
        h = build_phi_hint(phi, phi_gt=phi, hint_mode="gt_center")
        cy, cx = 8, 8
        for b in range(2):
            assert torch.allclose(h[b], torch.full_like(h[b], float(phi[b, 0, cy, cx])))

    def test_gt_center_requires_phi(self) -> None:
        ref = torch.randn(1, 1, 8, 8)
        with pytest.raises(ValueError, match="gt_center"):
            build_phi_hint(ref, phi_gt=None, hint_mode="gt_center")

    def test_wrapped_rejected(self) -> None:
        ref = torch.randn(1, 1, 8, 8)
        with pytest.raises(NotImplementedError):
            build_phi_hint(ref, phi_gt=None, hint_mode="wrapped")


class TestPrepareBatchOptionB:
    def test_default_hint_is_zero(self) -> None:
        I = torch.rand(2, 1, 32, 32)
        phi = torch.randn(2, 1, 32, 32)
        I_in, phi_out, _, _ = prepare_batch(I, phi, noise_aug=None)
        assert I_in.shape[1] == 2
        assert torch.count_nonzero(I_in[:, 1:2]) == 0
        assert torch.allclose(phi_out, phi)
