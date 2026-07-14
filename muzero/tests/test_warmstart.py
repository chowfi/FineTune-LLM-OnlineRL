from dataclasses import replace

import numpy as np

from muzero.config import MuZeroConfig
from muzero.replay_buffer import ReplayBuffer
from muzero.tests.helpers import (
    FakeEvaluator,
    PIKAFISH_BIN,
    make_evaluator,
    requires_engine,
)
from muzero.warmstart import SimpleUciEngine, generate_warmstart_games

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


@requires_engine
def test_multipv_returns_scored_moves():
    eng = SimpleUciEngine(PIKAFISH_BIN, movetime_ms=50, multipv=4)
    try:
        lines = eng.search(START_FEN)
    finally:
        eng.close()
    assert 1 <= len(lines) <= 4
    for uci, cp in lines:
        assert len(uci) == 4 and isinstance(cp, float)


@requires_engine
def test_warmstart_fills_buffer():
    cfg = replace(
        MuZeroConfig(), warmstart_plies=8, max_game_plies=6, warmstart_movetime_ms=20
    )
    buf = ReplayBuffer(cfg)
    stats = generate_warmstart_games(cfg, buf, make_evaluator())
    assert stats["plies"] >= 8
    assert len(buf.games) >= 1
    g = buf.games[0]
    assert len(g.policy_indices[0]) >= 1
    assert abs(g.root_values[0]) <= 1.0


def test_go_command_modes():
    """nodes=None keeps movetime search; nodes=N switches to node-limited.
    Constructed via __new__ so no engine process is spawned."""
    eng = SimpleUciEngine.__new__(SimpleUciEngine)
    eng.movetime_ms = 10
    eng.nodes = None
    assert eng._go_command() == "go movetime 10"
    eng.nodes = 128
    assert eng._go_command() == "go nodes 128"


@requires_engine
def test_node_limited_search_returns_move():
    eng = SimpleUciEngine(PIKAFISH_BIN, movetime_ms=10, multipv=1, nodes=8)
    try:
        lines = eng.search(START_FEN)
    finally:
        eng.close()
    assert lines and len(lines[0][0]) == 4


def test_init_kills_process_on_handshake_failure(monkeypatch):
    """A dead/silent engine binary must not leak the spawned subprocess."""
    import io

    import pytest

    import muzero.warmstart as warmstart

    class FakeProc:
        def __init__(self):
            self.killed = False
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")  # immediate EOF -> "engine died"

        def kill(self):
            self.killed = True

    holder = {}

    def fake_popen(*args, **kwargs):
        holder["proc"] = FakeProc()
        return holder["proc"]

    monkeypatch.setattr(warmstart.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="engine died"):
        SimpleUciEngine("nonexistent-binary", movetime_ms=10, multipv=1)
    assert holder["proc"].killed


def test_pick_move_index_samples_early_and_plays_best_late():
    from muzero.warmstart import _pick_move_index

    lines = [("a6a5", 0.0), ("b6b5", -30.0)]
    rng = np.random.default_rng(0)
    early_picks = {
        _pick_move_index(lines, ply=0, temperature_moves=30, rng=rng)
        for _ in range(200)
    }
    assert early_picks == {0, 1}  # both engine choices actually get played
    counts = [0, 0]
    rng = np.random.default_rng(1)
    for _ in range(400):
        counts[_pick_move_index(lines, 0, 30, rng)] += 1
    assert counts[0] > counts[1]  # better-scored move favored
    # at/after temperature_moves: always the best line
    assert all(
        _pick_move_index(lines, ply, 30, np.random.default_rng(i)) == 0
        for i, ply in enumerate((30, 31, 100))
    )
    # a single candidate is always index 0, any ply
    assert _pick_move_index([("a6a5", 0.0)], 0, 30, np.random.default_rng(2)) == 0


def test_play_engine_game_produces_buffer_ready_history():
    from muzero.warmstart import play_engine_game

    class ScriptedEngine:
        def search(self, fen):
            stm = fen.split()[1]
            # ENGINE-UCI; converts to the legal algebraic moves below
            return [("a6a5", 0.0)] if stm == "w" else [("i3i4", 0.0)]

    cfg = replace(
        MuZeroConfig(),
        max_game_plies=2,
        temperature_moves=0,  # deterministic: always best line
        opening_book=("a6a5",),  # -> algebraic "a3a4", legal for white below
    )

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    history = play_engine_game(
        cfg, ScriptedEngine(), evaluator, np.random.default_rng(0)
    )
    assert len(history) == 2  # opening ply + one engine ply, then ply cap
    assert len(history.rewards) == 2
    assert len(history.policy_indices) == 2
    assert history.result is not None
    buf = ReplayBuffer(cfg)
    buf.add(history)  # buffer-compatible record
    assert len(buf.games) == 1


def _seed_test_fixture(monkeypatch, n_games):
    """Run generate_seed_games with a scripted engine; returns (stats, buffer,
    constructed) where constructed counts fake-engine instantiations."""
    import muzero.warmstart as warmstart
    from muzero.warmstart import generate_seed_games

    constructed = []

    class FakeUci:
        def __init__(self, binary_path, movetime_ms, multipv):
            constructed.append(self)
            self.closed = False

        def search(self, fen):
            stm = fen.split()[1]
            return [("a6a5", 0.0)] if stm == "w" else [("i3i4", 0.0)]

        def close(self):
            self.closed = True

    monkeypatch.setattr(warmstart, "SimpleUciEngine", FakeUci)
    cfg = replace(
        MuZeroConfig(),
        max_game_plies=2,
        temperature_moves=0,
        opening_book=("a6a5",),
        seed_games_per_loop=n_games,
    )

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    buf = ReplayBuffer(cfg)
    stats = generate_seed_games(cfg, buf, evaluator, n_games, np.random.default_rng(0))
    return stats, buf, constructed


def test_generate_seed_games_plays_n_and_fills_buffer(monkeypatch):
    stats, buf, constructed = _seed_test_fixture(monkeypatch, n_games=3)
    assert stats == {"games": 3, "plies": 6}  # 3 games x 2-ply cap
    assert len(buf.games) == 3
    assert len(constructed) == 1 and constructed[0].closed


def test_generate_seed_games_zero_is_a_noop(monkeypatch):
    stats, buf, constructed = _seed_test_fixture(monkeypatch, n_games=0)
    assert stats == {"games": 0, "plies": 0}
    assert len(buf.games) == 0
    assert constructed == []  # engine never constructed


@requires_engine
def test_generate_seed_games_real_engine_smoke():
    from muzero.tests.helpers import make_evaluator
    from muzero.warmstart import generate_seed_games

    cfg = replace(MuZeroConfig(), max_game_plies=6, warmstart_movetime_ms=20)
    buf = ReplayBuffer(cfg)
    stats = generate_seed_games(cfg, buf, make_evaluator(), 1, np.random.default_rng(0))
    assert stats["games"] == 1 and stats["plies"] >= 1
    assert len(buf.games) == 1
