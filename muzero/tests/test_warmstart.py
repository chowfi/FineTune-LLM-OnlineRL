from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.replay_buffer import ReplayBuffer
from muzero.tests.helpers import PIKAFISH_BIN, make_evaluator, requires_engine
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
