"""Register numpy safe globals so weights_only=True loads work on PyTorch 2.6+.

Side-effect module: importing it registers the numpy reconstruction helpers
needed to unpickle checkpoints that embed numpy arrays (notably the
``rng_numpy`` entry written by ``CheckpointManager.save``). Import before any
``torch.load(..., weights_only=True)`` so the load stays safe instead of
falling back to the insecure ``weights_only=False``.
"""
import numpy as np
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    # _reconstruct / scalar live under different module paths across numpy 1.x/2.x.
    for _path in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            _mod = __import__(_path, fromlist=("_reconstruct", "scalar"))
            torch.serialization.add_safe_globals([_mod._reconstruct, _mod.scalar])
            break
        except (ImportError, AttributeError):
            continue
    torch.serialization.add_safe_globals([np.ndarray, np.dtype])
    # numpy 2.x represents dtype objects as numpy.dtypes.*DType instances; allowlist
    # them so weights_only=True can unpickle arrays (e.g. np.random.get_state() keys).
    try:
        import numpy.dtypes as _npdt
        torch.serialization.add_safe_globals(
            [v for v in vars(_npdt).values() if isinstance(v, type)]
        )
    except ImportError:
        pass

if __name__ == "__main__":
    import io

    buf = io.BytesIO()
    torch.save({"rng_numpy": np.random.get_state(), "model": {}}, buf)
    buf.seek(0)
    torch.load(buf, weights_only=True)  # UnpicklingError if globals are missing
    print("ok")