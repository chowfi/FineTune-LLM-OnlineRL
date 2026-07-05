from dataclasses import replace

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.encoding import move_to_index
from muzero.env import XiangqiEnv
from muzero.mcts import NetRunner
from muzero.network import MuZeroNet
from muzero.replay_buffer import GameHistory, ReplayBuffer
from muzero.selfplay import (
    SelfPlayCoordinator,
    SelfPlayWorker,
    _Game,
    select_action,
)
from muzero.tests.helpers import FakeEvaluator, make_evaluator, requires_engine


def test_coordinator_promotes_after_streak():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        device="cpu",
        self_play_mode="frozen_enemy",
    )
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    for _ in range(2):
        coord.report_result(ally_won=True, draw=False)
    assert coord.era == 0
    coord.report_result(ally_won=False, draw=False)  # streak resets
    for _ in range(3):
        coord.report_result(ally_won=True, draw=False)
    assert coord.era == 1
    for pa, pe in zip(ally.parameters(), enemy.parameters()):
        assert torch.equal(pa, pe)


def test_select_action_temperature():
    visits = {7: 10, 9: 2}
    rng = np.random.default_rng(0)
    assert select_action(visits, ply=100, temperature_moves=30, rng=rng) == 7  # argmax
    picks = {
        select_action(visits, ply=0, temperature_moves=30, rng=rng) for _ in range(50)
    }
    assert picks == {7, 9}  # sampling explores both


def test_concurrent_adds_keep_deques_aligned():
    import threading as th

    from muzero.tests.test_replay_buffer import make_game

    cfg = replace(MuZeroConfig(), buffer_games=200)
    buf = ReplayBuffer(cfg)
    threads = [
        th.Thread(target=lambda: [buf.add(make_game(length=6)) for _ in range(20)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf.games) == len(buf.priorities) == 80
    for g, p in zip(buf.games, buf.priorities):
        assert len(p) == len(g)


def _make_worker(cfg, evaluator):
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    return SelfPlayWorker(
        cfg,
        NetRunner(ally, "cpu"),
        NetRunner(enemy, "cpu"),
        buf,
        coord,
        evaluator,
        worker_id=0,
    )


def _new_game(cfg, evaluator, ally_side="w"):
    env = XiangqiEnv(cfg, evaluator)
    env.reset(ally_side=ally_side)
    history = GameHistory()
    history.ally_side = ally_side
    return _Game(env, history, "e6e5")


def test_record_and_step_tracks_ally_entropy_and_cp_pairs():
    # Pinned to frozen_enemy: this test documents ally-only diagnostics,
    # which only hold in frozen mode (latest mode records every move).
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        device="cpu",
        self_play_mode="frozen_enemy",
    )
    # FakeEvaluator.evaluate_cp always answers "+50 for whoever is to move in
    # the fen" (standard UCI side-to-move-relative convention). After the
    # ally ("w") moves, side_to_move flips to "b", so env.red_cp() (which
    # reports from red's/w's absolute perspective) negates it to -50.
    evaluator = FakeEvaluator(
        cp_fn=lambda fen: 50.0, legal_fn=lambda fen: ["a0a1", "a0b0"]
    )
    worker = _make_worker(cfg, evaluator)
    game = _new_game(cfg, evaluator, ally_side="w")

    # 1) ally ("w") to move: entropy recorded pre-step; cp paired post-step
    # (side flips to "b" once the ally has moved).
    action = move_to_index("e6e5")
    visits = {action: 3, move_to_index("e7e6"): 1}
    done, info = worker._record_and_step(
        game, action=action, visits=visits, root_value=0.25
    )
    assert not done
    assert len(game.ally_entropies) == 1
    assert game.ally_entropies[0] > 0.0  # non-degenerate visit distribution
    assert len(game.ally_value_cp_pairs) == 1
    root_value, ally_cp = game.ally_value_cp_pairs[0]
    assert root_value == 0.25
    assert ally_cp == -50.0  # ally_side == "w" -> red_cp used as-is

    # 2) enemy ("b") to move: no entropy recorded, and no new cp pair (the
    # ally didn't just move).
    action2 = move_to_index("e3e4")
    done2, info2 = worker._record_and_step(
        game, action=action2, visits={action2: 1}, root_value=-0.1
    )
    assert not done2
    assert len(game.ally_entropies) == 1  # unchanged
    assert len(game.ally_value_cp_pairs) == 1  # unchanged


def test_record_and_step_latest_mode_records_every_move():
    # Default latest mode: diagnostics cover EVERY move, not just the ally's.
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    assert cfg.self_play_mode == "latest"
    evaluator = FakeEvaluator(
        cp_fn=lambda fen: 50.0, legal_fn=lambda fen: ["a0a1", "a0b0"]
    )
    worker = _make_worker(cfg, evaluator)
    game = _new_game(cfg, evaluator, ally_side="w")

    # 1) ally-color ("w") move: entropy + value-cp pair + ally_cps entry.
    action = move_to_index("e6e5")
    visits = {action: 3, move_to_index("e7e6"): 1}
    done, info = worker._record_and_step(
        game, action=action, visits=visits, root_value=0.25
    )
    assert not done
    assert len(game.ally_entropies) == 1
    assert game.ally_value_cp_pairs == [(0.25, info["red_cp"])]  # mover "w"
    assert game.ally_cps == [info["red_cp"]]  # tracked color is "w"

    # 2) enemy-color ("b") move: entropy + value-cp pair recorded too, with
    # the cp in MOVER (black) perspective; ally_cps records every ply in the
    # tracked color's perspective.
    action2 = move_to_index("e3e4")
    done2, info2 = worker._record_and_step(
        game, action=action2, visits={action2: 1}, root_value=-0.1
    )
    assert not done2
    assert len(game.ally_entropies) == 2
    assert len(game.ally_value_cp_pairs) == 2
    root_value2, cp2 = game.ally_value_cp_pairs[1]
    assert root_value2 == -0.1
    assert info2["red_cp"] == 50.0  # black just moved; red is to move (+50)
    assert cp2 == -info2["red_cp"]  # mover-perspective: negated for black
    # both plies recorded, tracked-color ("w" = red) perspective
    assert game.ally_cps == [info["red_cp"], info2["red_cp"]]


def test_record_and_step_one_hot_ally_entropy_is_zero():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=lambda fen: ["a0a1"])
    worker = _make_worker(cfg, evaluator)
    game = _new_game(cfg, evaluator, ally_side="w")
    action = move_to_index("e6e5")
    worker._record_and_step(game, action=action, visits={action: 1}, root_value=0.0)
    assert len(game.ally_entropies) == 1
    assert abs(game.ally_entropies[0]) < 1e-9  # one-hot visits -> ~0 entropy


def test_finish_summary_includes_new_diagnostic_fields():
    # Pinned to frozen_enemy: value_cp_pairs is asserted as ally-only here,
    # which only holds in frozen mode (latest mode records every move).
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        device="cpu",
        max_game_plies=2,
        self_play_mode="frozen_enemy",
    )
    evaluator = FakeEvaluator(
        cp_fn=lambda fen: 50.0, legal_fn=lambda fen: ["a0a1", "a0b0"]
    )
    worker = _make_worker(cfg, evaluator)
    game = _new_game(cfg, evaluator, ally_side="w")
    action = move_to_index("e6e5")
    worker._record_and_step(
        game,
        action=action,
        visits={action: 3, move_to_index("e7e6"): 1},
        root_value=0.25,
    )
    action2 = move_to_index("e3e4")
    done, _ = worker._record_and_step(
        game, action=action2, visits={action2: 1}, root_value=-0.1
    )
    assert done  # max_game_plies=2 reached
    summary = worker._finish(game)
    assert summary["mean_root_entropy"] == game.ally_entropies[0]
    assert summary["value_cp_pairs"] == [(0.25, -50.0)]
    # ally ("w") cp after each ply: -50 (post own move), +50 (post reply)
    assert summary["mean_ally_cp"] == 0.0
    assert summary["games_this_era"] == 1


@requires_engine
def test_selfplay_smoke_generates_games():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=8,
        interior_topk=8,
        games_per_worker=2,
        max_game_plies=6,
        device="cpu",
    )
    torch.manual_seed(0)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    worker = SelfPlayWorker(
        cfg,
        NetRunner(ally, "cpu"),
        NetRunner(enemy, "cpu"),
        buf,
        coord,
        make_evaluator(),
        worker_id=0,
    )
    summaries = worker.generate(num_games=2)
    assert len(summaries) == 2
    assert len(buf.games) == 2
    game = buf.games[0]
    assert 1 <= len(game) <= 6
    assert len(game.boards) == len(game) + 1


def test_coordinator_promotion_disabled_in_latest_mode():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    assert cfg.self_play_mode == "latest"
    torch.manual_seed(0)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    before = [p.detach().clone() for p in enemy.parameters()]
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    for _ in range(5):
        promoted = coord.report_result(ally_won=True, draw=False)
        assert promoted is False
    assert coord.era == 0
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, enemy.parameters()))


def test_round_groups_by_mode():
    """Latest mode: one spec covering all games with noise; frozen mode: the
    original ally/enemy two-spec split (noise on ally roots only)."""

    def make_worker(mode):
        cfg = replace(
            MuZeroConfig(),
            channels=16,
            repr_blocks=1,
            dyn_blocks=1,
            device="cpu",
            self_play_mode=mode,
        )
        torch.manual_seed(0)
        net = MuZeroNet(cfg)
        ally_runner, enemy_runner = NetRunner(net, "cpu"), NetRunner(net, "cpu")
        coord = SelfPlayCoordinator(cfg, net, net)
        buf = ReplayBuffer(cfg)
        return SelfPlayWorker(
            cfg, ally_runner, enemy_runner, buf, coord, object(), worker_id=0
        )

    latest = make_worker("latest")
    assert latest._round_groups([]) == [(latest.ally_runner, None, True)]

    frozen = make_worker("frozen_enemy")
    assert frozen._round_groups([]) == [
        (frozen.ally_runner, True, True),
        (frozen.enemy_runner, False, False),
    ]


def test_blunder_and_search_kl_tracking():
    # Constant stm-perspective cp of 150 -> every mover's delta is -300,
    # beyond the 200cp blunder threshold.
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    evaluator = FakeEvaluator(cp_fn=lambda fen: 150.0, legal_fn=lambda fen: ["a0a1"])
    worker = _make_worker(cfg, evaluator)
    game = _new_game(cfg, evaluator, ally_side="w")

    action = move_to_index("e6e5")
    worker._record_and_step(
        game, action=action, visits={action: 4}, root_value=0.0, search_kl=0.37
    )
    assert game.cp_moves == 1 and game.blunders == 1
    assert game.search_kls == [0.37]

    # Forced-opening style call (no search): no KL recorded, cp still counted.
    action2 = move_to_index("e3e4")
    worker._record_and_step(game, action=action2, visits={action2: 1}, root_value=0.0)
    assert game.cp_moves == 2 and game.blunders == 2
    assert game.search_kls == [0.37]

    summary = worker._finish(game)
    assert summary["blunders"] == 2 and summary["cp_moves"] == 2
    assert summary["mean_search_kl"] == 0.37


def test_generate_stores_absolute_actions_for_black():
    """Catches PARTIAL root-adapter wiring (flip-in without unflip-out, or
    vice versa): either one-sided bug stores flip_action(a6a5)=a3a4 for
    black's ply, which this test rejects.

    Note the limits: FakeEvaluator legal strings are ENGINE UCI (rank 0 =
    bottom), so "a3a4" resolves through env.legal_moves() to internal
    absolute "a6a5". And because flip_action is an involution, a root with
    ONE legal move cannot distinguish a correctly-wired adapter from a
    completely absent one — total absence is instead covered by the
    engine-gated legal-set bijection test and the training watch criteria."""
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        num_workers=1,
        games_per_worker=1,
        max_game_plies=2,
        device="cpu",
    )
    torch.manual_seed(0)
    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=lambda fen: ["a3a4"])
    ally = MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    coord = SelfPlayCoordinator(cfg, ally, ally)
    runner = NetRunner(ally, "cpu")
    worker = SelfPlayWorker(cfg, runner, runner, buf, coord, evaluator, worker_id=0)
    summaries = worker.generate(1)
    assert len(summaries) == 1
    game = buf.games[0]
    # ply 0 is the forced red opening (absolute by construction);
    # ply 1 is black's MCTS move and must be stored ABSOLUTE ("a6a5"):
    assert game.actions[1] == move_to_index("a6a5")
    assert list(game.policy_indices[1]) == [move_to_index("a6a5")]
