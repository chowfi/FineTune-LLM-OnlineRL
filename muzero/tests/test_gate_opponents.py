import numpy as np

from muzero.gate_opponents import greedy_capture_move


class StubEnv:
    """greedy_capture_move only touches .legal_moves() and .board."""

    def __init__(self, board, moves):
        self.board = board
        self._moves = list(moves)

    def legal_moves(self):
        return list(self._moves)


def empty_board():
    return np.zeros((10, 9), dtype=np.int8)


def test_prefers_highest_value_capture():
    board = empty_board()
    board[4, 4] = 10  # red cannon (the mover's piece; value irrelevant)
    board[4, 0] = -12  # black pawn (value 1.0) at (4,0) = "a4"
    board[4, 8] = -8  # black rook (value 9.0) at (4,8) = "i4"
    moves = ["e4a4", "e4i4"]  # capture pawn vs capture rook
    rng = np.random.default_rng(0)
    for _ in range(5):  # deterministic regardless of rng draws
        assert greedy_capture_move(StubEnv(board, moves), rng) == "e4i4"


def test_tie_between_equal_captures_stays_in_tied_set():
    board = empty_board()
    board[4, 4] = 10
    board[4, 0] = -12  # pawn
    board[4, 8] = -13  # pawn (same value)
    moves = ["e4a4", "e4i4", "e4e5"]  # two pawn captures + one quiet move
    rng = np.random.default_rng(0)
    picks = {greedy_capture_move(StubEnv(board, moves), rng) for _ in range(20)}
    assert picks <= {"e4a4", "e4i4"}  # never the quiet move
    assert len(picks) == 2  # both ties reachable


def test_no_captures_falls_back_to_random_legal():
    board = empty_board()
    board[4, 4] = 10
    moves = ["e4e5", "e4e3", "e4d4"]
    rng = np.random.default_rng(0)
    picks = {greedy_capture_move(StubEnv(board, moves), rng) for _ in range(30)}
    assert picks <= set(moves)
    assert len(picks) > 1  # actually random, not pinned


def test_empty_legal_list_returns_none():
    assert (
        greedy_capture_move(StubEnv(empty_board(), []), np.random.default_rng(0))
        is None
    )


def test_black_greedy_captures_red_pieces():
    board = empty_board()
    board[5, 4] = -10  # black cannon (mover)
    board[5, 0] = 12  # red pawn (value 1.0)
    board[5, 8] = 8  # red rook (value 9.0)
    moves = ["e5a5", "e5i5"]
    rng = np.random.default_rng(0)
    assert greedy_capture_move(StubEnv(board, moves), rng) == "e5i5"


def test_returns_plain_python_str():
    board = empty_board()
    board[4, 4] = 10
    pick = greedy_capture_move(StubEnv(board, ["e4e5"]), np.random.default_rng(0))
    assert type(pick) is str  # not np.str_ — callers compare/serialize these


def test_seeded_rng_is_reproducible():
    board = empty_board()
    board[4, 4] = 10
    board[4, 0] = -12
    board[4, 8] = -13  # tied pawn captures -> rng actually consulted
    moves = ["e4a4", "e4i4", "e4e5"]
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    seq_a = [greedy_capture_move(StubEnv(board, moves), rng_a) for _ in range(10)]
    seq_b = [greedy_capture_move(StubEnv(board, moves), rng_b) for _ in range(10)]
    assert seq_a == seq_b


def test_king_capture_outranks_everything():
    board = empty_board()
    board[4, 4] = 10
    board[4, 0] = -8  # rook (9.0)
    board[4, 8] = -1  # king
    moves = ["e4a4", "e4i4"]
    rng = np.random.default_rng(0)
    assert greedy_capture_move(StubEnv(board, moves), rng) == "e4i4"


def test_gate_rung_plays_greedy_without_engine(tmp_path):
    """_run_gate_rung + greedy opponent runs on FakeEvaluator (no Pikafish)."""
    import torch
    from dataclasses import replace

    from muzero.config import MuZeroConfig
    from muzero.mcts import NetRunner
    from muzero.network import MuZeroNet
    from muzero.tests.helpers import FakeEvaluator
    from muzero.train import _run_gate_rung

    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        gate_games=2,
        max_game_plies=2,
        device="cpu",
    )
    torch.manual_seed(0)
    runner = NetRunner(MuZeroNet(cfg), "cpu")

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    rng = np.random.default_rng(0)

    def greedy(env):
        return greedy_capture_move(env, rng)

    wins, draws = _run_gate_rung(cfg, runner, evaluator, greedy)
    assert 0 <= wins + draws <= cfg.gate_games  # both 2-ply games complete
    assert draws == 2  # max_game_plies=2 -> both games drawn at the cap
