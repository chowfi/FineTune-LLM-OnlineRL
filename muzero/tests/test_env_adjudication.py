from dataclasses import replace

import numpy as np
import pytest

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from muzero.tests.helpers import FakeEvaluator

# Horse shuffle in internal coords (red horse h9<->g7, black horse h0<->g2).
SHUFFLE = ["h9g7", "h0g2", "g7h9", "g2h0"]


def test_threefold_repetition_is_draw_with_penalty():
    cfg = MuZeroConfig()
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 0.0))
    env.reset()
    done = False
    outputs = []
    for i in range(8):  # start position recurs at plies 0, 4, 8
        assert not done
        _, reward, done, info = env.step(SHUFFLE[i % 4])
        outputs.append((reward, info))
    assert done
    assert env.result == "draw_repetition"
    reward, info = outputs[-1]
    assert info["repetition_penalized"] == "b"  # black completed the repetition
    assert reward <= cfg.repetition_penalty


def test_hopeless_ally_game_truncates_as_loss():
    cfg = replace(
        MuZeroConfig(), truncation_consecutive=3, self_play_mode="frozen_enemy"
    )
    # evaluate_cp is side-to-move perspective; +900 after every move means the
    # mover always left the opponent at +900, i.e. red is at -900 after red moves.
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 900.0))
    env.reset(ally_side="w")
    done = False
    plies = 0
    while not done:
        _, reward, done, info = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "black_win"
    assert env.truncated and info["truncated"]
    assert plies == 5  # red's 3rd saturated move
    # Terminal -1 plus shaping on the final move: red-perspective cp swings
    # +900 -> -900 (delta -1800) because the fake engine always reports +900
    # for the side to move.
    shaping = cfg.shaping_weight * float(np.tanh(-1800.0 / cfg.shaping_cp_scale))
    assert reward == pytest.approx(-1.0 + shaping, abs=1e-6)
