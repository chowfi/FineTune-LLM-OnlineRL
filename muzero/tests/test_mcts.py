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
    for (visits, root_value, search_kl), legal in zip(results, [legal_a, legal_b]):
        assert sum(visits.values()) == 16
        assert set(visits.keys()) <= set(legal.tolist())
        assert np.isfinite(root_value)
        # KL(visit distribution || raw pre-noise prior): finite, non-negative.
        assert np.isfinite(search_kl) and search_kl >= -1e-9


def test_backup_stats_track_child_q():
    from muzero.mcts import MinMaxStats, Node

    cfg = replace(MuZeroConfig(), device="cpu")
    mcts = MCTS(cfg)
    root, child = Node(0.5), Node(1.0)
    child.prior_action = 7
    child.reward = 0.3
    root.cand_actions = np.array([7], dtype=np.int64)
    root.cand_priors = np.array([1.0], dtype=np.float32)
    root.children = {0: child}
    stats = MinMaxStats()
    mcts._backup([root, child], -1.0, stats)
    q1 = child.reward + cfg.discount * -child.value()  # 0.3 + 1.0 = 1.3
    assert stats.minimum == stats.maximum == q1
    mcts._backup([root, child], -0.5, stats)
    q2 = child.reward + cfg.discount * -child.value()  # 0.3 + 0.75 = 1.05
    assert stats.minimum == min(q1, q2) and stats.maximum == max(q1, q2)


def test_mcts_deterministic_with_seeded_rng():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=8,
        interior_topk=8,
        device="cpu",
    )
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    obs = np.random.default_rng(1).standard_normal((115, 10, 9)).astype(np.float32)
    legal = np.array([3, 44, 500], dtype=np.int64)
    r1 = MCTS(cfg, rng=np.random.default_rng(5)).run(
        NetRunner(net, "cpu"), [(obs, legal)], add_noise=True
    )
    r2 = MCTS(cfg, rng=np.random.default_rng(5)).run(
        NetRunner(net, "cpu"), [(obs, legal)], add_noise=True
    )
    assert r1[0][0] == r2[0][0]
