from dataclasses import replace

import torch

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.replay_buffer import ReplayBuffer
from muzero.tests.test_replay_buffer import make_game
from muzero.train import MuZeroTrainer


def test_train_batch_runs_and_updates_params():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        batch_size=4,
        unroll_steps=8,
        device="cpu",
    )
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    for _ in range(2):
        buf.add(make_game())
    trainer = MuZeroTrainer(cfg, net)
    before = [p.detach().clone() for p in net.parameters()]
    losses = trainer.train_batch(buf.sample_batch(4))
    for key in (
        "policy",
        "value",
        "reward",
        "moves_left",
        "material",
        "consistency",
        "total",
    ):
        assert key in losses and torch.isfinite(torch.tensor(losses[key])), key
    changed = any(
        not torch.equal(b, p.detach()) for b, p in zip(before, net.parameters())
    )
    assert changed


def test_train_batch_reports_finite_buffer_age():
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        batch_size=4,
        unroll_steps=8,
        device="cpu",
    )
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    for _ in range(2):
        buf.add(make_game())
    trainer = MuZeroTrainer(cfg, net)
    batch = buf.sample_batch(4)
    result = trainer.train_batch(batch)
    assert "buffer_age" in result
    assert result["buffer_age"] == result["buffer_age"]  # not NaN
    assert result["buffer_age"] >= 0.0
    assert "mean_buffer_age" not in batch  # popped before tensorizing


def test_load_checkpoint_without_enemy_key(tmp_path):
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    torch.manual_seed(0)
    source = MuZeroNet(cfg)
    trainer_src = MuZeroTrainer(cfg, source)
    path = tmp_path / "latest.pt"
    torch.save(
        {
            "ally": source.state_dict(),
            "optimizer": trainer_src.optimizer.state_dict(),
            "iteration": 7,
            "era": 0,
            "streak": 1,
        },
        path,
    )  # note: no "enemy" key, as latest-mode checkpoints are written

    torch.manual_seed(1)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    trainer = MuZeroTrainer(cfg, ally)
    from muzero.train import load_checkpoint

    ckpt = load_checkpoint(str(path), ally, enemy, trainer.optimizer, "cpu")
    assert ckpt["iteration"] == 7
    for pa, pe in zip(ally.parameters(), enemy.parameters()):
        assert torch.equal(pa, pe)  # enemy fell back to ally weights
