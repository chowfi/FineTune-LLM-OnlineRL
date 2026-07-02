import torch

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet, action_planes


def tiny_config():
    return MuZeroConfig(channels=16, repr_blocks=1, dyn_blocks=1, device="cpu")


def test_action_planes():
    planes = action_planes(torch.tensor([9]), "cpu")  # a0a1: from sq 0, to sq 9
    assert planes.shape == (1, 2, 10, 9)
    assert planes[0, 0, 0, 0] == 1.0  # from-square a0 -> row 0, col 0
    assert planes[0, 1, 1, 0] == 1.0  # to-square a1 -> row 1, col 0


def test_inference_shapes():
    cfg = tiny_config()
    net = MuZeroNet(cfg)
    obs = torch.randn(4, 115, 10, 9)
    out = net.initial_inference(obs)
    assert out["hidden"].shape == (4, 16, 10, 9)
    assert out["policy_logits"].shape == (4, 8100)
    assert out["value_logits"].shape == (4, cfg.value_bins)
    assert out["value"].shape == (4,)
    assert out["moves_left_logits"].shape == (4, cfg.moves_left_max + 1)
    assert out["material"].shape == (4,)

    out2 = net.recurrent_inference(out["hidden"], torch.tensor([9, 9, 9, 9]))
    assert out2["hidden"].shape == (4, 16, 10, 9)
    assert out2["reward_logits"].shape == (4, cfg.reward_bins)
    assert out2["reward"].shape == (4,)


def test_projection_shapes():
    cfg = tiny_config()
    net = MuZeroNet(cfg)
    hidden = torch.randn(4, 16, 10, 9)
    assert net.project(hidden, with_predictor=False).shape == (4, 1024)
    assert net.project(hidden, with_predictor=True).shape == (4, 1024)


def test_default_param_count_within_spec():
    net = MuZeroNet(MuZeroConfig(device="cpu"))
    params = sum(p.numel() for p in net.parameters()) / 1e6
    assert 20.0 <= params <= 35.0, params
