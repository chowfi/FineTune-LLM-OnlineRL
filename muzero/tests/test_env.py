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
    assert obs.shape == (115, 10, 9)
