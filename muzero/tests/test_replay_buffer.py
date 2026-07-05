from dataclasses import replace

import numpy as np

from muzero.config import MuZeroConfig
from muzero.encoding import flip_action, mirror_action, move_to_index
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
    assert batch["obs"].shape == (8, 114, 10, 9)
    assert batch["actions"].shape == (8, 8)
    assert batch["target_policy"].shape == (8, 9, 8100)
    assert batch["policy_mask"].shape == (8, 9)
    assert batch["target_value"].shape == (8, 9)
    assert batch["target_reward"].shape == (8, 8)
    assert batch["target_moves_left"].shape == (8, 9)
    assert batch["target_material"].shape == (8, 9)
    assert batch["consistency_obs"].shape == (8, 114, 10, 9)
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


def make_alternating_game(length=6):
    """Deterministic game: red plays a6a5, black plays a3a4, one-hot policies.

    Boards hold kings plus one pawn each, with an extra red rook so the
    material balance is nonzero ((9+1-1)/10 = +0.9 red-perspective)."""
    g = GameHistory()
    board = np.zeros((10, 9), dtype=np.int8)
    board[9, 4], board[0, 4] = 1, -1  # kings
    board[6, 0], board[3, 0] = 12, -12  # one pawn each
    board[9, 0] = 8  # extra red rook -> red is +9.0 material
    for t in range(length + 1):
        g.boards.append(board.copy())
        g.to_play_history.append("w" if t % 2 == 0 else "b")
        g.rep_history.append(1)
        g.no_progress_history.append(t)
    for t in range(length):
        a = move_to_index("a6a5") if t % 2 == 0 else move_to_index("a3a4")
        g.actions.append(a)
        g.rewards.append(0.0)
        g.policy_indices.append(np.array([a], dtype=np.int64))
        g.policy_probs.append(np.array([1.0], dtype=np.float32))
        g.root_values.append(0.0)
    g.result = "draw_max_plies"
    return g


def test_make_target_flips_black_plies_keystone():
    """THE doorway-consistency test: for every black ply the action and
    policy target, decoded in the flipped frame, must name the same physical
    move that was played. A boards-flipped-but-targets-not bug cannot pass."""
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    tgt = buf.make_target(g, 0, mirror=False)
    for k in range(cfg.unroll_steps):
        stored = g.actions[k]
        expected = stored if g.to_play_history[k] == "w" else flip_action(stored)
        assert tgt["actions"][k] == expected
        assert int(np.argmax(tgt["target_policy"][k])) == expected
    # black's a3a4 must literally become red's a6a5 in the canonical frame
    assert tgt["actions"][1] == move_to_index("a6a5")


def test_make_target_material_is_mover_perspective():
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    tgt = buf.make_target(g, 0, mirror=False)
    for k in range(cfg.unroll_steps + 1):
        expected = 0.9 if g.to_play_history[k] == "w" else -0.9
        assert tgt["target_material"][k] == np.float32(expected)


def test_make_target_mirror_consistency():
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    buf.rng = np.random.default_rng(7)
    plain = buf.make_target(g, 0, mirror=False)
    # same k_c draw for both calls: mirror is passed explicitly, which skips
    # the coin-flip draw, so k_c is rng draw #1 in both branches. If make_target
    # ever adds an rng draw conditioned on `mirror`, this desyncs and the
    # consistency_obs assertion below will fail spuriously — fix the draw
    # order, not the assertion.
    buf.rng = np.random.default_rng(7)
    mirrored = buf.make_target(g, 0, mirror=True)
    # observations mirror along the column axis, plane-for-plane
    np.testing.assert_array_equal(mirrored["obs"], plain["obs"][:, :, ::-1])
    np.testing.assert_array_equal(
        mirrored["consistency_obs"], plain["consistency_obs"][:, :, ::-1]
    )
    # every action / policy target mirrors with the boards, never without
    for k in range(cfg.unroll_steps):
        assert mirrored["actions"][k] == mirror_action(int(plain["actions"][k]))
        assert int(np.argmax(mirrored["target_policy"][k])) == mirror_action(
            int(np.argmax(plain["target_policy"][k]))
        )
    # frame-independent targets are identical
    np.testing.assert_array_equal(mirrored["target_value"], plain["target_value"])
    np.testing.assert_array_equal(mirrored["target_material"], plain["target_material"])


def test_make_target_is_deterministic_given_seed():
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    g = make_alternating_game(length=6)
    buf_a, buf_b = ReplayBuffer(cfg), ReplayBuffer(cfg)
    a = buf_a.make_target(g, 0)
    b = buf_b.make_target(g, 0)
    np.testing.assert_array_equal(a["obs"], b["obs"])  # same seed -> same draw
