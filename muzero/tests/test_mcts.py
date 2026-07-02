from dataclasses import replace

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet


def test_mcts_respects_mask_and_visit_budget():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=16,
        interior_topk=8,
        device="cpu",
    )
    torch.manual_seed(0)
    runner = NetRunner(MuZeroNet(cfg), "cpu")
    rng = np.random.default_rng(0)
    legal_a = np.array([5, 100, 8099], dtype=np.int64)
    legal_b = np.array([0, 1, 2, 3], dtype=np.int64)
    roots = [
        (rng.standard_normal((115, 10, 9)).astype(np.float32), legal_a),
        (rng.standard_normal((115, 10, 9)).astype(np.float32), legal_b),
    ]
    results = MCTS(cfg).run(runner, roots, add_noise=True)
    assert len(results) == 2
    for (visits, root_value), legal in zip(results, [legal_a, legal_b]):
        assert sum(visits.values()) == 16
        assert set(visits.keys()) <= set(legal.tolist())
        assert np.isfinite(root_value)
