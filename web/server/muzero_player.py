"""MuZero checkpoint as a web-play opponent (exact gate-strength search)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from muzero.config import MuZeroConfig
from muzero.encoding import absolute_visits, index_to_move
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet
from muzero.selfplay import canonical_root


class MuZeroPlayer:
    """Loads a canonical (114-plane) checkpoint and picks argmax-MCTS moves.

    `num_simulations`/`config` are test hooks only; production uses the
    MuZeroConfig defaults (800 sims — "always full strength" per spec)."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        num_simulations: Optional[int] = None,
        config: Optional[MuZeroConfig] = None,
    ):
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"MuZero checkpoint not found: {path} — copy one from the "
                "training box or set XIANGQI_MUZERO_CKPT"
            )
        cfg = config or MuZeroConfig()
        cfg.device = device
        if num_simulations is not None:
            cfg.num_simulations = num_simulations
        self.cfg = cfg
        net = MuZeroNet(cfg).to(device)
        ckpt = torch.load(str(path), map_location=device)
        try:
            net.load_state_dict(ckpt["ally"])
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"Incompatible MuZero checkpoint {path} (pre-canonicalization "
                f"115-plane checkpoints cannot be loaded): {exc}"
            ) from exc
        net.eval()
        self.runner = NetRunner(net, device)
        self.mcts = MCTS(cfg)

    def choose_move(self, env) -> str:
        """Gate-strength move: no noise, argmax visits, ABSOLUTE algebraic."""
        obs, legal = canonical_root(env)
        ((visits, _, _),) = self.mcts.run(self.runner, [(obs, legal)], add_noise=False)
        visits = absolute_visits(visits, env.side_to_move)
        return index_to_move(max(visits, key=visits.get))
