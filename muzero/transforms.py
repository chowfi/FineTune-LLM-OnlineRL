"""MuZero invertible value scaling and scalar <-> categorical support."""

from __future__ import annotations

import torch

EPS = 0.001


def h_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + EPS * x


def h_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (
        ((torch.sqrt(1.0 + 4.0 * EPS * (torch.abs(x) + 1.0 + EPS)) - 1.0) / (2.0 * EPS))
        ** 2
        - 1.0
    )


def scalar_to_support(
    x: torch.Tensor, vmin: float, vmax: float, bins: int
) -> torch.Tensor:
    """Two-hot encoding of scalars onto a linear support. Output: x.shape + (bins,)."""
    x = x.clamp(vmin, vmax)
    step = (vmax - vmin) / (bins - 1)
    pos = (x - vmin) / step
    low = pos.floor().long().clamp(0, bins - 1)
    high = (low + 1).clamp(0, bins - 1)
    frac = pos - low.float()
    out = torch.zeros(*x.shape, bins, dtype=torch.float32, device=x.device)
    out.scatter_(-1, low.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    out.scatter_add_(-1, high.unsqueeze(-1), frac.unsqueeze(-1))
    return out


def support_to_scalar(
    logits: torch.Tensor, vmin: float, vmax: float, bins: int
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    atoms = torch.linspace(vmin, vmax, bins, device=logits.device)
    return (probs * atoms).sum(dim=-1)
