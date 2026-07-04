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


def test_symmetric_truncation_fires_on_non_ally_side():
    # Default mode is "latest" -> symmetric truncation. Script the engine so
    # ONLY black is hopeless: evaluate_cp is side-to-move perspective, so
    # +900 with white to move means red is +900 (black just left itself lost),
    # and -900 with black to move means red is +900 again (red is fine).
    cfg = replace(MuZeroConfig(), truncation_consecutive=3)
    assert cfg.truncation_symmetric
    env = XiangqiEnv(
        cfg, FakeEvaluator(cp_fn=lambda fen: 900.0 if " w " in fen else -900.0)
    )
    env.reset(ally_side="w")  # the LOSER (black) is NOT the ally
    done = False
    plies = 0
    while not done:
        _, reward, done, info = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "red_win"  # black, the saturated side, loses
    assert env.truncated and info["truncated"]
    assert plies == 6  # black's 3rd saturated move (plies 2, 4, 6)
    assert reward < 0  # final mover is black, the loser: -1 + shaping


def test_asymmetric_mode_ignores_non_ally_saturation():
    # frozen_enemy mode: only the ally's streak counts. Make RED hopeless
    # while the ally is BLACK -> no truncation; the shuffle ends in the
    # 3-fold repetition draw at ply 8 instead.
    #
    # A *constant* cp_fn is unsuitable here: evaluate_cp is side-to-move
    # perspective, so a constant cp makes every mover's own-move cp compute
    # to the same saturated value regardless of color (that symmetry is
    # exactly what test_hopeless_ally_game_truncates_as_loss above relies on
    # for its ally="w" scenario). To make ONLY red hopeless, the fake
    # evaluator must differentiate by side to move: report black (about to
    # move, after red's move) as heavily winning, so red's mover-cp is very
    # negative; report white (about to move, after black's move) as neutral,
    # so black's mover-cp never saturates.
    cfg = replace(
        MuZeroConfig(), truncation_consecutive=3, self_play_mode="frozen_enemy"
    )
    assert not cfg.truncation_symmetric
    env = XiangqiEnv(
        cfg, FakeEvaluator(cp_fn=lambda fen: 900.0 if " b " in fen else 0.0)
    )
    env.reset(ally_side="b")
    done = False
    plies = 0
    while not done:
        _, _, done, _ = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "draw_repetition"
    assert not env.truncated
    assert plies == 8


def test_step_reports_mover_cp_delta():
    # Constant stm-perspective cp c makes every mover's delta -2c: before the
    # move the mover sees +c; after, the opponent sees +c, i.e. the mover -c.
    cfg = MuZeroConfig()
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 150.0))
    env.reset()
    _, _, _, info = env.step(SHUFFLE[0])
    assert info["mover_cp_delta"] == -300.0

    env2 = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: None))
    env2.reset()
    _, _, _, info2 = env2.step(SHUFFLE[0])
    assert info2["mover_cp_delta"] is None
