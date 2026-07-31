import os
import csv
import threading
from typing import Dict, Any, Optional, List

import torch

if hasattr(torch.serialization, "add_safe_globals"):
    try:
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    except ImportError:
        pass
    try:
        import numpy._core.multiarray
        torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
    except ImportError:
        pass
    import numpy as np
    torch.serialization.add_safe_globals([np.ndarray, np.dtype, np.core.multiarray.scalar if hasattr(np, 'core') else np._core.multiarray.scalar])

from ..plots import save_epoch_visuals

class CSVLogger:
    """Handles appending metrics to a CSV file safely."""
    def __init__(self, metrics_path: str, columns: Optional[List[str]] = None, append: bool = True):
        self.metrics_path = metrics_path
        self.append = append
        self.columns = columns or [
            "epoch",
            "partial",
            "noise_level",
            "train_loss",
            "train_mae",
            "train_grad",
            "train_curv",
            "val_abs_mae",
            "val_topo_mae",
            "val_rmse",
            "val_ssim",
            "val_psnr",
            "val_max_err",
            "val_grad_mae",
            "lr",
            "epoch_time_s",
        ]
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.append and os.path.exists(self.metrics_path):
            os.remove(self.metrics_path)
            
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log(self, metrics: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([metrics.get(c, "") for c in self.columns])

class VisualizationDispatcher:
    """Dispatches visualization saving to a background thread."""
    def __init__(self, vis_dir: str, vis_max: int = 8):
        self.vis_dir = vis_dir
        self.vis_max = vis_max

    def _save_visuals_bg(self, *args: Any, **kwargs: Any) -> None:
        try:
            save_epoch_visuals(*args, **kwargs)
        except Exception as e:
            print(f"Warning: background visual saving failed: {e}")

    def dispatch(
        self,
        epoch: int,
        I_input_v: torch.Tensor,
        phi_aligned_v: torch.Tensor,
        phi_gt_v: torch.Tensor,
        I_raw_c_v: torch.Tensor,
        I_raw_n_v: torch.Tensor,
        noise_level: float,
    ) -> None:
        try:
            threading.Thread(
                target=self._save_visuals_bg,
                args=(
                    I_input_v.cpu(),
                    phi_aligned_v.detach().cpu(),
                    phi_gt_v.cpu(),
                    self.vis_dir,
                    epoch,
                    self.vis_max,
                ),
                kwargs={
                    "I_raw_clean": I_raw_c_v.cpu(),
                    "I_raw_noisy": I_raw_n_v.cpu(),
                    "noise_level": noise_level,
                    "samples_per_file": 4,
                },
            ).start()
        except Exception as e:
            print(f"Warning: dispatching visuals failed: {e}")

class CheckpointManager:
    """Handles saving and loading PyTorch checkpoints."""
    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def save(
        self,
        filename: str,
        epoch: int,
        model: torch.nn.Module,
        ema: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler,
        best_mae: float,
        phase_generator: Optional[torch.Generator] = None,
        noise_generator: Optional[torch.Generator] = None,
        next_sample_id: int = 0,
    ) -> None:
        import random
        import numpy as np
        state = {
            "epoch": epoch,
            "best_mae": best_mae,
            "model": model.state_dict(),
            "model_ema": ema.m.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.random.get_rng_state(),
            "next_sample_id": next_sample_id,
        }
        if phase_generator is not None:
            state["rng_phase"] = phase_generator.get_state()
        if noise_generator is not None:
            state["rng_noise"] = noise_generator.get_state()
        if torch.cuda.is_available():
            state["rng_cuda"] = torch.cuda.get_rng_state_all()
        target_path = os.path.join(self.run_dir, filename)
        tmp_path = target_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)

    @staticmethod
    def load(
        path: str,
        model: torch.nn.Module,
        ema: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        phase_generator: Optional[torch.Generator] = None,
        noise_generator: Optional[torch.Generator] = None,
    ) -> tuple[int, float, int]:
        import random
        import numpy as np
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.m.load_state_dict(ckpt["model_ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        if "rng_python" in ckpt:
            random.setstate(ckpt["rng_python"])
        if "rng_numpy" in ckpt:
            np.random.set_state(ckpt["rng_numpy"])
        if "rng_torch" in ckpt:
            torch.random.set_rng_state(ckpt["rng_torch"])
        if "rng_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        if phase_generator is not None and "rng_phase" in ckpt:
            phase_generator.set_state(ckpt["rng_phase"])
        if noise_generator is not None and "rng_noise" in ckpt:
            noise_generator.set_state(ckpt["rng_noise"])

        return ckpt.get("epoch", 0) + 1, ckpt.get("best_mae", float("inf")), ckpt.get("next_sample_id", 0)

CSV_COLUMNS = [
    "epoch",
    "partial",
    "noise_level",
    "train_loss",
    "train_mae",
    "train_grad",
    "train_curv",
    "val_abs_mae",
    "val_topo_mae",
    "val_rmse",
    "val_ssim",
    "val_psnr",
    "val_max_err",
    "val_grad_mae",
    "lr",
    "epoch_time_s",
]
