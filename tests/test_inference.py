"""load_inference_state must raise on empty val/test splits, not fall back to train."""

from __future__ import annotations

import inspect

from phase_unwrap.core.inference import load_inference_state


def test_load_inference_state_subset_param():
    """load_inference_state must accept subset and default to 'val'."""
    sig = inspect.signature(load_inference_state)
    assert "subset" in sig.parameters
    assert sig.parameters["subset"].default == "val"


def test_val_branch_does_not_reference_train_loader():
    """
    The subset='val' branch in load_inference_state must raise on empty val
    (not silently fall back to train_loader). Verified statically by reading the
    source — a full runtime test requires a trained checkpoint, which is out of
    scope for this unit test.
    """
    import inspect as _inspect
    import phase_unwrap.core.inference as inf_mod

    src = _inspect.getsource(inf_mod.load_inference_state)
    # The val branch must raise ValueError, not fall back to train_loader.
    assert 'val_frac=0 in config' in src, (
        "subset='val' branch must raise ValueError mentioning val_frac=0"
    )
    # The old silent-fallback pattern must not be present.
    assert 'else train_loader' not in src, (
        "subset='val' branch must not fall back to train_loader"
    )
