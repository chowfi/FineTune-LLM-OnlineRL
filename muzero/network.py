"""Representation / dynamics / prediction networks + SimSiam projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.transforms import h_inverse, support_to_scalar


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


def action_planes(actions: torch.Tensor, device) -> torch.Tensor:
    """Action indices (B,) -> (B, 2, 10, 9): from-square and to-square one-hots."""
    b = actions.shape[0]
    planes = torch.zeros(b, 2, 90, device=device)
    frm = (actions // 90).long()
    to = (actions % 90).long()
    idx = torch.arange(b, device=device)
    planes[idx, 0, frm] = 1.0
    planes[idx, 1, to] = 1.0
    return planes.view(b, 2, 10, 9)


def normalize_hidden(h: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max scaling to [0, 1] (MuZero appendix G)."""
    flat = h.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-8)
    return flat.view_as(h)


def _tower(in_ch: int, ch: int, blocks: int) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(),
    ]
    layers += [ResBlock(ch) for _ in range(blocks)]
    return nn.Sequential(*layers)


def _head(ch: int, reduced: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ch, reduced, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(reduced * 90, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out),
    )


class MuZeroNet(nn.Module):
    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        self.representation = _tower(cfg.input_planes, ch, cfg.repr_blocks)
        self.dynamics = _tower(ch + 2, ch, cfg.dyn_blocks)
        self.reward_head = _head(ch, 2, 128, cfg.reward_bins)
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 90, cfg.action_space),
        )
        self.value_head = _head(ch, 2, 256, cfg.value_bins)
        self.moves_left_head = _head(ch, 2, 128, cfg.moves_left_max + 1)
        self.material_head = _head(ch, 2, 128, 1)
        self.projector = nn.Sequential(
            nn.Conv2d(ch, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 90, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.predictor = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1024)
        )

    def _predict(self, hidden: torch.Tensor) -> dict:
        value_logits = self.value_head(hidden)
        value = h_inverse(
            support_to_scalar(
                value_logits,
                -self.cfg.value_max,
                self.cfg.value_max,
                self.cfg.value_bins,
            )
        )
        return {
            "hidden": hidden,
            "policy_logits": self.policy_head(hidden),
            "value_logits": value_logits,
            "value": value,
            "moves_left_logits": self.moves_left_head(hidden),
            "material": self.material_head(hidden).squeeze(-1),
        }

    def initial_inference(self, obs: torch.Tensor) -> dict:
        return self._predict(normalize_hidden(self.representation(obs)))

    def recurrent_inference(self, hidden: torch.Tensor, actions: torch.Tensor) -> dict:
        x = torch.cat([hidden, action_planes(actions, hidden.device)], dim=1)
        next_hidden = normalize_hidden(self.dynamics(x))
        reward_logits = self.reward_head(next_hidden)
        out = self._predict(next_hidden)
        out["reward_logits"] = reward_logits
        out["reward"] = support_to_scalar(
            reward_logits,
            -self.cfg.reward_max,
            self.cfg.reward_max,
            self.cfg.reward_bins,
        )
        return out

    def project(self, hidden: torch.Tensor, with_predictor: bool) -> torch.Tensor:
        p = self.projector(hidden)
        return self.predictor(p) if with_predictor else p
