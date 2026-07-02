from dataclasses import replace

import numpy as np

from muzero.config import MuZeroConfig
from muzero.replay_buffer import GameHistory, ReplayBuffer


def make_game(length=20, result="red_win", truncated=False):
    g = GameHistory()
    rng = np.random.default_rng(0)
    for t in range(length + 1):
        g.boards.append(rng.integers(-16, 17, size=(10, 9)).astype(np.int8))
        g.to_play_history.append("w" if t % 2 == 0 else "b")
        g.rep_history.append(1)
        g.no_progress_history.append(t)
    for t in range(length):
        g.actions.append(int(rng.integers(0, 8100)))
        g.rewards.append(0.0)
        g.policy_indices.append(np.array([1, 2, 3], dtype=np.int64))
        g.policy_probs.append(np.array([0.5, 0.3, 0.2], dtype=np.float32))
        g.root_values.append(0.1)
    g.rewards[-1] = 1.0
    g.result = result
    g.truncated = truncated
    return g


def test_add_and_fifo_eviction():
    cfg = replace(MuZeroConfig(), buffer_games=3)
    buf = ReplayBuffer(cfg)
    for _ in range(5):
        buf.add(make_game())
    assert len(buf.games) == 3


def test_n_step_value_perspective_signs():
    cfg = replace(MuZeroConfig(), td_steps=10)
    g = make_game(length=4)  # rewards [0,0,0,1.0], no bootstrap past end
    buf = ReplayBuffer(cfg)
    # From state 3 the mover receives +1 immediately.
    assert buf.n_step_value(g, 3) == 1.0
    # From state 2 the opponent gets +1 next ply -> -1 for the state-2 mover.
    assert buf.n_step_value(g, 2) == -1.0


def test_sample_batch_shapes():
    cfg = replace(MuZeroConfig(), batch_size=8, unroll_steps=8)
    buf = ReplayBuffer(cfg)
    for _ in range(4):
        buf.add(make_game())
    batch = buf.sample_batch(8)
    assert batch["obs"].shape == (8, 115, 10, 9)
    assert batch["actions"].shape == (8, 8)
    assert batch["target_policy"].shape == (8, 9, 8100)
    assert batch["policy_mask"].shape == (8, 9)
    assert batch["target_value"].shape == (8, 9)
    assert batch["target_reward"].shape == (8, 8)
    assert batch["target_moves_left"].shape == (8, 9)
    assert batch["target_material"].shape == (8, 9)
    assert batch["consistency_obs"].shape == (8, 115, 10, 9)
    assert batch["consistency_k"].shape == (8,)
    np.testing.assert_allclose(
        np.asarray(batch["target_policy"]).sum(-1), 1.0, rtol=1e-4
    )


def test_sample_batch_returns_mean_buffer_age():
    cfg = replace(MuZeroConfig(), batch_size=8, unroll_steps=8)
    buf = ReplayBuffer(cfg)
    for _ in range(4):
        buf.add(make_game())
    batch = buf.sample_batch(8)
    assert "mean_buffer_age" in batch
    assert np.asarray(batch["mean_buffer_age"]).ndim == 0
    assert np.isfinite(batch["mean_buffer_age"])
    assert batch["mean_buffer_age"] >= 0.0


def test_buffer_index_assigned_in_add_order():
    cfg = replace(MuZeroConfig(), buffer_games=10)
    buf = ReplayBuffer(cfg)
    for _ in range(3):
        buf.add(make_game())
    assert [g.buffer_index for g in buf.games] == [0, 1, 2]
    assert buf.total_games_added == 3
