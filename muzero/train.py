"""Training: K-step unrolled combined loss + the main orchestration loop."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet, normalize_hidden
from muzero.transforms import h_transform, scalar_to_support


def scale_gradient(t: torch.Tensor, scale: float) -> torch.Tensor:
    return t * scale + t.detach() * (1.0 - scale)


def _soft_ce(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(-1)


class MuZeroTrainer:
    def __init__(self, cfg: MuZeroConfig, net: MuZeroNet):
        self.cfg = cfg
        self.net = net
        self.optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.train_steps = 0

    def _to_tensors(self, batch: dict) -> dict:
        dev = next(self.net.parameters()).device
        out = {}
        for k, v in batch.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("actions", "target_moves_left", "consistency_k"):
                t = t.long()
            else:
                t = t.float()
            out[k] = t.to(dev)
        return out

    def train_batch(self, batch: dict) -> dict:
        cfg = self.cfg
        self.net.train()
        b = self._to_tensors(batch)
        B, K = b["actions"].shape

        value_support = scalar_to_support(
            h_transform(b["target_value"]),
            -cfg.value_max,
            cfg.value_max,
            cfg.value_bins,
        )
        reward_support = scalar_to_support(
            b["target_reward"], -cfg.reward_max, cfg.reward_max, cfg.reward_bins
        )

        out = self.net.initial_inference(b["obs"])
        losses = {
            "policy": (
                _soft_ce(out["policy_logits"], b["target_policy"][:, 0])
                * b["policy_mask"][:, 0]
            ).mean(),
            "value": _soft_ce(out["value_logits"], value_support[:, 0]).mean(),
            "reward": torch.zeros((), device=b["obs"].device),
            "moves_left": F.cross_entropy(
                out["moves_left_logits"], b["target_moves_left"][:, 0]
            ),
            "material": F.mse_loss(out["material"], b["target_material"][:, 0]),
        }
        hidden = out["hidden"]
        latents = []
        for k in range(1, K + 1):
            out_k = self.net.recurrent_inference(hidden, b["actions"][:, k - 1])
            hidden = scale_gradient(out_k["hidden"], 0.5)
            latents.append(hidden)
            losses["policy"] = (
                losses["policy"]
                + (
                    _soft_ce(out_k["policy_logits"], b["target_policy"][:, k])
                    * b["policy_mask"][:, k]
                ).mean()
                / K
            )
            losses["value"] = (
                losses["value"]
                + _soft_ce(out_k["value_logits"], value_support[:, k]).mean() / K
            )
            losses["reward"] = (
                losses["reward"]
                + _soft_ce(out_k["reward_logits"], reward_support[:, k - 1]).mean() / K
            )
            losses["moves_left"] = (
                losses["moves_left"]
                + F.cross_entropy(
                    out_k["moves_left_logits"], b["target_moves_left"][:, k]
                )
                / K
            )
            losses["material"] = (
                losses["material"]
                + F.mse_loss(out_k["material"], b["target_material"][:, k]) / K
            )

        # SimSiam consistency at one sampled unroll offset per sample.
        k_c = b["consistency_k"]
        mask = (k_c > 0).float()
        if mask.sum() > 0:
            stacked = torch.stack(latents, dim=1)  # (B, K, ch, 10, 9)
            gather = (
                (k_c.clamp(min=1) - 1)
                .view(B, 1, 1, 1, 1)
                .expand(-1, 1, *stacked.shape[2:])
            )
            dyn_latent = stacked.gather(1, gather).squeeze(1)
            with torch.no_grad():
                target_latent = self.net.representation(b["consistency_obs"])
                target_proj = self.net.project(
                    normalize_hidden(target_latent), with_predictor=False
                )
            pred = self.net.project(dyn_latent, with_predictor=True)
            cos = F.cosine_similarity(pred, target_proj.detach(), dim=-1)
            losses["consistency"] = -(cos * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            losses["consistency"] = torch.zeros((), device=b["obs"].device)

        w = cfg.loss_weights
        total = (
            w[0] * losses["policy"]
            + w[1] * losses["value"]
            + w[2] * losses["reward"]
            + w[3] * losses["moves_left"]
            + w[4] * losses["material"]
            + w[5] * losses["consistency"]
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.train_steps += 1
        result = {k: float(v.detach()) for k, v in losses.items()}
        result["total"] = float(total.detach())
        return result
