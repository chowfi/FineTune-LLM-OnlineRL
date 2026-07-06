from web.server.muzero_session import MuZeroGameSession
from web.tests.test_muzero_player import build_player, fake_evaluator, tiny_cfg

SNAPSHOT_KEYS = {
    "board",
    "graphic",
    "fen",
    "allyMode",
    "engineKind",
    "humanSide",
    "sideToMove",
    "turn",
    "gameOver",
    "winner",
    "lastAllyMove",
    "lastEngineMove",
    "engineThinking",
}


def make_session(tmp_path, **cfg_over):
    cfg = tiny_cfg(**cfg_over)
    player = build_player(tmp_path, cfg)
    return MuZeroGameSession(fake_evaluator(), player, config=cfg)


def make_session_from(tmp_path, cfg):
    player = build_player(tmp_path, cfg)
    return MuZeroGameSession(fake_evaluator(), player, config=cfg)


def test_snapshot_contract_and_red_start(tmp_path):
    sess = make_session(tmp_path)
    snap = sess.reset(human_side="red")
    assert SNAPSHOT_KEYS <= set(snap)
    assert snap["engineKind"] == "muzero"
    assert snap["humanSide"] == "red"
    assert snap["turn"] == "human"  # red moves first
    assert snap["allyMode"] == "human"
    assert snap["gameOver"] is False


def test_red_full_exchange(tmp_path):
    sess = make_session(tmp_path)
    sess.reset(human_side="red")
    snap, err = sess.apply_human_move("a6a5")
    assert err is None
    assert snap["lastAllyMove"] == "a6a5"
    assert snap["turn"] == "engine"
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["lastEngineMove"] == "i3i4"  # absolute, model's only legal
    assert snap["turn"] == "human"


def test_black_engine_moves_first(tmp_path):
    sess = make_session(tmp_path)
    snap = sess.reset(human_side="black")
    assert snap["humanSide"] == "black"
    assert snap["turn"] == "engine"  # model is Red, moves first
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["lastEngineMove"] == "a6a5"
    assert snap["turn"] == "human"


def test_illegal_move_and_wrong_turn_rejected(tmp_path):
    sess = make_session(tmp_path)
    sess.reset(human_side="red")
    _, err = sess.apply_human_move("e6e5")
    assert err == "Move is not legal (Pikafish)"
    _, err = sess.apply_engine_move()
    assert err == "Not engine turn"
    _, err = sess.apply_greedy_ally()
    assert err == "Greedy ally is not supported with the MuZero engine"


def test_max_plies_draw_maps_to_draw_winner(tmp_path):
    sess = make_session(tmp_path, max_game_plies=2)
    sess.reset(human_side="red")
    sess.apply_human_move("a6a5")
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["gameOver"] is True
    assert snap["winner"] == "draw"  # draw_max_plies -> "draw"
    assert snap["turn"] == "none"
    _, err = sess.apply_human_move("a6a5")
    assert err == "Game is already over"


def test_mate_no_legal_reply_declares_mover_winner(tmp_path):
    from muzero.tests.helpers import FakeEvaluator
    from web.tests.test_muzero_player import build_player, tiny_cfg

    # red's move is always available; black has NO reply after red moves.
    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else []

    cfg = tiny_cfg()
    sess = MuZeroGameSession(
        FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal),
        build_player(tmp_path, cfg),
        config=cfg,
    )
    sess.reset(human_side="red")
    snap, err = sess.apply_human_move("a6a5")
    assert err is None
    assert snap["gameOver"] is True
    assert snap["winner"] == "red"  # opponent has no legal move -> mover wins
    _, err = sess.apply_engine_move()
    assert err == "Game is already over"


def test_session_does_not_mutate_caller_config(tmp_path):
    cfg = tiny_cfg()
    before = cfg.truncation_consecutive
    sess = make_session_from(tmp_path, cfg)
    assert sess.cfg.truncation_consecutive == 10**9
    assert cfg.truncation_consecutive == before


def test_engine_error_keeps_session_usable(tmp_path):
    sess = make_session(tmp_path)
    sess.reset(human_side="black")  # engine (red) to move

    class Boom:
        def choose_move(self, env):
            raise RuntimeError("synthetic failure")

    sess.player = Boom()
    snap, err = sess.apply_engine_move()
    assert err == "Engine error: synthetic failure"
    assert snap["engineThinking"] is False
    assert snap["gameOver"] is False  # env unstepped, session intact
