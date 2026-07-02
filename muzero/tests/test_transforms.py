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
