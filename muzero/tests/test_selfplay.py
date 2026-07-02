from dataclasses import replace

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.mcts import NetRunner
from muzero.network import MuZeroNet
from muzero.replay_buffer import ReplayBuffer
from muzero.selfplay import SelfPlayCoordinator, SelfPlayWorker, select_action
from muzero.tests.helpers import make_evaluator, requires_engine


def test_coordinator_promotes_after_streak():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
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
