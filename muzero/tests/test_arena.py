from dataclasses import replace

import torch

from muzero.arena import (
    discover_checkpoints,
    fit_arena_elo,
    games_needed,
    play_pair,
)
from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.tests.helpers import FakeEvaluator


def tiny_cfg(**over):
    defaults = dict(
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=2,
        interior_topk=2,
        max_game_plies=2,
        device="cpu",
    )
    defaults.update(over)
    return replace(MuZeroConfig(), **defaults)


def save_tiny(dirpath, name, iteration, seed):
    torch.manual_seed(seed)
    net = MuZeroNet(tiny_cfg())
    path = dirpath / name
    torch.save({"ally": net.state_dict(), "iteration": iteration}, path)
    return path


def fake_evaluator():
    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    return FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)


def test_discover_checkpoints_sorts_by_iteration(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    save_tiny(archive, "iter_0040.pt", 40, seed=1)
    save_tiny(archive, "iter_0020.pt", 20, seed=2)
    extra = save_tiny(tmp_path, "iter80-prebufferfix.pt", 81, seed=3)
    found = discover_checkpoints(str(archive), extras=[str(extra)])
    assert [c.label for c in found] == ["iter_0020", "iter_0040", "iter80-prebufferfix"]
    assert [c.iteration for c in found] == [20, 40, 81]


def test_games_needed_respects_existing_rows():
    rows = [
        {"white": "a", "black": "b", "result": "draw", "sims": 2},
        {"white": "b", "black": "a", "result": "win", "sims": 2},
        {"white": "a", "black": "b", "result": "win", "sims": 800},  # other sims
    ]
    assert games_needed(rows, "a", "b", sims=2, games_per_pair=4) == 2
    assert games_needed(rows, "a", "b", sims=800, games_per_pair=4) == 3


def test_play_pair_writes_valid_rows(tmp_path):
    a = save_tiny(tmp_path, "iter_0020.pt", 20, seed=1)
    b = save_tiny(tmp_path, "iter_0040.pt", 40, seed=2)
    cfg = tiny_cfg()
    rows = play_pair(
        cfg,
        fake_evaluator(),
        ("iter_0020", str(a)),
        ("iter_0040", str(b)),
        n_games=2,
    )
    assert len(rows) == 2
    whites = {r["white"] for r in rows}
    assert whites == {"iter_0020", "iter_0040"}  # colors alternate
    for r in rows:
        assert r["result"] in ("win", "loss", "draw")
        assert r["sims"] == cfg.num_simulations
    # max_game_plies=2 forces draws in this stub world
    assert all(r["result"] == "draw" for r in rows)


def test_fit_arena_elo_anchors_oldest_and_rates_winner_higher():
    # synthetic: "new" beats "old" 15-5 as an even color split
    rows = []
    for i in range(20):
        white, black = ("new", "old") if i % 2 == 0 else ("old", "new")
        new_is_white = white == "new"
        new_wins = i < 15
        result = "win" if (new_wins == new_is_white) else "loss"
        rows.append({"white": white, "black": black, "result": result, "sims": 2})
    ratings = fit_arena_elo(rows, order=["old", "new"])
    assert ratings["old"] == 0.0
    assert 100.0 < ratings["new"] < 400.0  # ~+190 for 75%
