from dataclasses import replace

import pytest
import torch

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.tests.helpers import FakeEvaluator
from web.server.muzero_player import MuZeroPlayer


def tiny_cfg(**over):
    return replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        max_game_plies=4,
        device="cpu",
        **over,
    )


def build_player(tmp_path, cfg):
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    ckpt = tmp_path / "latest.pt"
    torch.save({"ally": net.state_dict()}, ckpt)
    return MuZeroPlayer(str(ckpt), device="cpu", config=cfg)


def fake_evaluator():
    # ENGINE-UCI legals; env converts rank r -> 9-r to internal algebraic:
    # red to move gets internal "a6a5", black gets internal "i3i4".
    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    return FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)


def test_choose_move_returns_absolute_legal_move_for_both_colors(tmp_path):
    from muzero.env import XiangqiEnv

    cfg = tiny_cfg()
    player = build_player(tmp_path, cfg)
    env = XiangqiEnv(cfg, fake_evaluator())
    env.reset()
    assert player.choose_move(env) == "a6a5"  # red: single legal, absolute
    env.step("a6a5")
    assert player.choose_move(env) == "i3i4"  # black: single legal, absolute


def test_missing_checkpoint_raises_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
        MuZeroPlayer(str(tmp_path / "nonexistent.pt"))


def test_incompatible_checkpoint_raises_actionable_error(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"ally": {}}, ckpt)  # empty state dict -> load_state_dict fails
    with pytest.raises(RuntimeError, match="Incompatible"):
        MuZeroPlayer(str(ckpt), config=tiny_cfg())
