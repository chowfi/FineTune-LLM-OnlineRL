import torch

from muzero.transforms import (
    h_inverse,
    h_transform,
    scalar_to_support,
    support_to_scalar,
)


def test_h_transform_round_trip():
    x = torch.tensor([-250.0, -1.0, 0.0, 0.5, 1.0, 250.0])
    assert torch.allclose(h_inverse(h_transform(x)), x, atol=1e-3)


def test_support_round_trip():
    x = torch.tensor([[-1.7, 0.0, 0.31, 1.9]])
    support = scalar_to_support(x, -2.0, 2.0, 21)
    assert support.shape == (1, 4, 21)
    assert torch.allclose(support.sum(-1), torch.ones(1, 4), atol=1e-6)
    # support_to_scalar expects logits; log of the two-hot distribution works
    back = support_to_scalar(torch.log(support + 1e-12), -2.0, 2.0, 21)
    assert torch.allclose(back, x, atol=1e-3)


def test_support_edges_and_clamping():
    x = torch.tensor([[-2.0, 2.0, -5.0, 5.0]])  # exact edges + out-of-range
    support = scalar_to_support(x, -2.0, 2.0, 21)
    assert torch.allclose(support.sum(-1), torch.ones(1, 4), atol=1e-6)
    back = support_to_scalar(torch.log(support + 1e-12), -2.0, 2.0, 21)
    assert torch.allclose(back, torch.tensor([[-2.0, 2.0, -2.0, 2.0]]), atol=1e-3)


def test_value_head_config_round_trip():
    # value head: 601 bins over [-3, 3] in h-transformed units
    from muzero.transforms import h_transform

    raw = torch.tensor([[-2.0, -1.0, 0.0, 0.5, 2.0]])
    hx = h_transform(raw)
    assert hx.abs().max() < 3.0
    support = scalar_to_support(hx, -3.0, 3.0, 601)
    back = support_to_scalar(torch.log(support + 1e-12), -3.0, 3.0, 601)
    assert torch.allclose(back, hx, atol=1e-3)


def test_float64_input_does_not_crash():
    x = torch.tensor([[0.5]], dtype=torch.float64)
    support = scalar_to_support(x, -2.0, 2.0, 21)
    assert support.dtype == torch.float32
