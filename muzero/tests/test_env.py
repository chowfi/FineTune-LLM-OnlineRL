from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from muzero.tests.helpers import FakeEvaluator, make_evaluator, requires_engine


@requires_engine
def test_reset_and_legal_moves():
    env = XiangqiEnv(MuZeroConfig(), make_evaluator())
    board = env.reset()
    assert board.shape == (10, 9)
    assert env.side_to_move == "w"
    legal = env.legal_moves()
    assert len(legal) == 44  # known perft(1) of the start position
    assert all(len(m) == 4 for m in legal)


@requires_engine
def test_step_pawn_push():
    env = XiangqiEnv(MuZeroConfig(), make_evaluator())
    env.reset()
    board, reward, done, info = env.step("e6e5")  # red central pawn (engine e3e4)
    assert not done
    assert env.side_to_move == "b"
    assert env.plies == 1
    assert board[6, 4] == 0 and board[5, 4] > 0
    assert isinstance(reward, float)


def test_observation_shape_with_fake_engine():
    env = XiangqiEnv(MuZeroConfig(), FakeEvaluator())
    env.reset()
    obs = env.observation()
    assert obs.shape == (114, 10, 9)


@requires_engine
def test_flip_maps_legal_move_sets():
    """Flipping the board + flipping each legal move must yield exactly the
    legal-move set of the flipped position (colors swapped, red to move)."""
    from muzero.encoding import flip_action, flip_board, index_to_move, move_to_index
    from src.xiangqi_board import board_to_uci_fen, engine_uci_to_algebraic

    evaluator = make_evaluator()
    cfg = MuZeroConfig()
    env = XiangqiEnv(cfg, evaluator)
    env.reset()
    env.step(engine_uci_to_algebraic("h2e2"))  # red central cannon; black to move
    black_legal = {move_to_index(m) for m in env.legal_moves()}
    assert black_legal  # guard: bijection of two empty sets would pass vacuously
    flipped_fen = board_to_uci_fen(flip_board(env.board), "w")
    flipped_legal = {
        move_to_index(engine_uci_to_algebraic(u))
        for u in evaluator.list_legal_moves(flipped_fen)
    }
    flipped_black = {flip_action(a) for a in black_legal}
    diff = flipped_black.symmetric_difference(flipped_legal)
    assert flipped_black == flipped_legal, (
        f"legal-set mismatch; offending moves: {sorted(index_to_move(a) for a in diff)}"
    )
