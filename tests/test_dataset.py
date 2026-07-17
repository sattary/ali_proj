import h5py
import numpy as np
import pytest
import torch

from phase_unwrap.data.dataset import H5ShardDataset, smart_split


def test_h5shard_dataset(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    with h5py.File(h5_path, "w") as f:
        # Create dummy data: 5 samples of 1x16x16
        I = np.random.randn(5, 1, 16, 16).astype(np.float32)
        phi = np.random.randn(5, 1, 16, 16).astype(np.float32)
        f.create_dataset("I", data=I)
        f.create_dataset("phi", data=phi)

    ds = H5ShardDataset([h5_path], augment=False)
    
    # Assert __len__
    assert len(ds) == 5
    
    # Assert __getitem__
    I_t, phi_t = ds[0]
    assert isinstance(I_t, torch.Tensor)
    assert isinstance(phi_t, torch.Tensor)
    assert I_t.shape == (1, 16, 16)
    assert phi_t.shape == (1, 16, 16)
    
    # Fetch last item
    I_t, phi_t = ds[4]
    assert I_t.shape == (1, 16, 16)

    ds.close()

def test_smart_split_zero():
    paths = ["a.h5", "b.h5", "c.h5", "d.h5", "e.h5"]
    train_paths, val_paths, test_paths = smart_split(paths, val_frac=0.0, test_frac=0.0)
    assert len(train_paths) == 5
    assert len(val_paths) == 0
    assert len(test_paths) == 0
