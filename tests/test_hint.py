"""Option B hint channel: zero by default, no GT leak."""

from __future__ import annotations

import torch

from phase_unwrap.data.augmentation import build_phi_hint, prepare_batch


class TestBuildPhiHint:
    def test_zero_is_zeros(self) -> None:
        ref = torch.randn(2, 1, 16, 16)
        h = build_phi_hint(ref, phi_gt=None, hint_mode="zero")
        assert h.shape == ref.shape
        assert torch.count_nonzero(h) == 0


class TestPrepareBatchOptionB:
    def test_default_hint_is_zero(self) -> None:
        I = torch.rand(2, 1, 32, 32)
        phi = torch.randn(2, 1, 32, 32)
        I_in, phi_out, _, _ = prepare_batch(I, phi, noise_aug=None)
        assert I_in.shape[1] == 2
        assert torch.count_nonzero(I_in[:, 1:2]) == 0
        assert torch.allclose(phi_out, phi)
