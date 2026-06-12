"""
Loss landscape visualization (Li et al., 2018).

Computes a 2D slice of the loss surface around the converged weights
using filter-normalized random directions, then renders as a filled
contour plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from ..core.losses import MAEGradLoss
from .style import SINGLE_COL, nature_style, save_figure


def _get_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.data.clone() for p in model.parameters()]


def _set_parameters(model: torch.nn.Module, params: list[torch.Tensor]) -> None:
    for p, v in zip(model.parameters(), params):
        p.data.copy_(v)


def _random_direction(params: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Generate a filter-normalized random direction.

    Each parameter tensor gets a random Gaussian direction,
    normalized to have the same norm as the original parameter
    (filter normalization from Li et al., 2018).
    """
    direction: list[torch.Tensor] = []
    for p in params:
        d = torch.randn_like(p)
        if p.dim() >= 2:
            # filter normalization: normalize each filter to match param norm
            for idx in range(p.shape[0]):
                p_norm = p[idx].norm()
                d_norm = d[idx].norm()
                if d_norm > 1e-10:
                    d[idx] = d[idx] * (p_norm / d_norm)
        else:
            # scalar / bias: simple norm matching
            p_norm = p.norm()
            d_norm = d.norm()
            if d_norm > 1e-10:
                d = d * (p_norm / d_norm)
        direction.append(d)
    return direction


def _perturb(
    base_params: list[torch.Tensor],
    d1: list[torch.Tensor],
    d2: list[torch.Tensor],
    alpha: float,
    beta: float,
) -> list[torch.Tensor]:
    return [b + alpha * v1 + beta * v2 for b, v1, v2 in zip(base_params, d1, d2)]


@torch.no_grad()
def _evaluate_loss(
    model: torch.nn.Module,
    loss_fn: MAEGradLoss,
    loader,
    device: torch.device,
    num_samples: int,
) -> float:
    """Compute average loss over (at most) num_samples examples."""
    total_loss = 0.0
    n = 0
    for I_input, phi_gt, I_raw_n, _ in loader:
        if n >= num_samples:
            break
        I_input = I_input.to(device)
        phi_gt = phi_gt.to(device)
        I_raw_n = I_raw_n.to(device) if I_raw_n is not None else None

        with autocast(device_type=device.type, enabled=False):
            phi_raw, k_off = model(I_input)
            if isinstance(phi_raw, list):
                phi_abs = [p + k_off for p in phi_raw]
            else:
                phi_abs = phi_raw + k_off
            loss, _ = loss_fn(phi_abs, phi_gt, I_raw_n)
        total_loss += float(loss.item()) * I_input.size(0)
        n += I_input.size(0)
    return total_loss / max(1, n)


def plot_loss_landscape(
    checkpoint_path: str,
    data_dir: str,
    out_path: str = "results/figs/loss_landscape.png",
    grid_size: int = 31,
    alpha_range: float = 1.0,
    num_eval_samples: int = 500,
    config_path: str | None = None,
    subset: str = "val",
) -> None:
    """
    2D loss landscape contour around converged weights.

    Args:
        checkpoint_path: Trained model checkpoint.
        data_dir:        Dataset directory for loss evaluation.
        out_path:        Output figure path.
        grid_size:       Resolution of the 2D grid (grid_size x grid_size).
        alpha_range:     Range for perturbation [-alpha_range, +alpha_range].
        num_eval_samples: Max samples used per loss evaluation (speed tradeoff).
        config_path:     Optional config override.
    """
    from ..core.inference import load_inference_state
    model, cfg, _, device = load_inference_state(checkpoint_path, data_dir='', config_path=config_path)

    loss_fn = MAEGradLoss(w_mae=cfg.loss.w_mae, w_grad=cfg.loss.w_grad)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )
    if subset == "test":
        if test_loader is None:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        loader = val_loader or train_loader
    else:
        loader = train_loader

    base_params = _get_parameters(model)
    d1 = _random_direction(base_params)
    d2 = _random_direction(base_params)

    alphas = np.linspace(-alpha_range, alpha_range, grid_size)
    betas = np.linspace(-alpha_range, alpha_range, grid_size)
    losses = np.zeros((grid_size, grid_size))

    total_evals = grid_size * grid_size
    print(f"Loss landscape: {total_evals} evaluations, {num_eval_samples} samples each")

    with tqdm(total=total_evals, desc="Loss landscape", mininterval=2.0) as pbar:
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = _perturb(base_params, d1, d2, a, b)
                _set_parameters(model, perturbed)
                losses[j, i] = _evaluate_loss(
                    model, loss_fn, loader, device, num_eval_samples
                )
                pbar.update(1)

    # restore original weights
    _set_parameters(model, base_params)

    with nature_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))

        levels = np.linspace(losses.min(), np.percentile(losses, 95), 25)
        cs = ax.contourf(alphas, betas, losses, levels=levels, cmap="viridis")
        ax.contour(
            alphas, betas, losses, levels=levels, colors="k", linewidths=0.2, alpha=0.4
        )

        cb = plt.colorbar(cs, ax=ax, pad=0.02)
        cb.set_label("Loss", fontsize=6)
        cb.ax.tick_params(labelsize=5)

        # mark the optimum (center)
        ax.plot(
            0,
            0,
            marker="*",
            color="white",
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=0.5,
            zorder=5,
        )

        ax.set_xlabel(r"Direction $\mathbf{d}_1$")
        ax.set_ylabel(r"Direction $\mathbf{d}_2$")
        ax.set_title("Loss Landscape")
        ax.set_aspect("equal")

        fig.tight_layout(pad=0.3)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        print(f"Saved loss landscape: {out_path}")
