"""Diagnose SimSiam consistency-branch collapse (loss/consistency pinned at -1).

The consistency loss is -cosine(predictor(projector(dynamics latent)),
stopgrad(projector(representation(actual future obs)))). A reading of exactly
-1.0 on every batch can mean either (a) the world model truly predicts future
latents perfectly (implausible at iteration 3) or (b) the projector maps every
input to nearly the same direction, making the match trivial — representation
collapse. This tool separates the two by measuring how similar the projections
of DIFFERENT samples are to each other:

- healthy: loss-cosine can be high, but pairwise cosine between different
  samples' projections stays well below 1 (the projection still depends on
  the input);
- collapsed: pairwise cosine between different samples ~= 1 (everything maps
  to one direction), so the loss-cosine of ~1 carries no information.

Run on the training machine against the live checkpoint (no engine needed —
observations are synthesized, which is fine because collapse is a property of
the network, visible on any inputs):

    uv run python -m muzero.diagnose_consistency \
        --ckpt checkpoints/muzero_xiangqi/latest.pt

Local sanity modes (used to validate this tool and produce reference numbers):

    uv run python -m muzero.diagnose_consistency            # fresh net
    uv run python -m muzero.diagnose_consistency --tiny --train-steps 60
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet, normalize_hidden
from muzero.replay_buffer import GameHistory, ReplayBuffer


def build_synthetic_buffer(
    cfg: MuZeroConfig, games: int = 6, plies: int = 24, seed: int = 0
):
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(cfg)
    for _ in range(games):
        game = GameHistory()
        for t in range(plies + 1):
            game.boards.append(rng.integers(-16, 17, size=(10, 9)).astype(np.int8))
            game.to_play_history.append("w" if t % 2 == 0 else "b")
            game.rep_history.append(1)
            game.no_progress_history.append(t)
        for _t in range(plies):
            game.actions.append(int(rng.integers(0, cfg.action_space)))
            game.rewards.append(float(rng.normal(0.0, 0.2)))
            game.policy_indices.append(np.array([1, 2, 3], dtype=np.int64))
            game.policy_probs.append(np.array([0.5, 0.3, 0.2], dtype=np.float32))
            game.root_values.append(float(rng.normal(0.0, 0.3)))
        game.result = "draw_max_plies"
        buf.add(game)
    return buf


def pairwise_cosine(x: torch.Tensor) -> float:
    """Mean cosine between all pairs of DIFFERENT rows of x (B, D)."""
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.T
    off_diagonal = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    return float(off_diagonal.mean())


@torch.no_grad()
def diagnose(net: MuZeroNet, batch: dict) -> dict:
    net.eval()
    keep = torch.from_numpy(batch["consistency_k"]).long() > 0
    obs = torch.from_numpy(batch["obs"]).float()[keep]
    c_obs = torch.from_numpy(batch["consistency_obs"]).float()[keep]
    k_c = torch.from_numpy(batch["consistency_k"]).long()[keep]
    actions = torch.from_numpy(batch["actions"]).long()[keep]

    hidden = net.initial_inference(obs)["hidden"]
    unrolled = []
    for k in range(actions.shape[1]):
        hidden = net.recurrent_inference(hidden, actions[:, k])["hidden"]
        unrolled.append(hidden)
    stacked = torch.stack(unrolled, dim=1)
    gather = (k_c - 1).view(-1, 1, 1, 1, 1).expand(-1, 1, *stacked.shape[2:])
    dyn_latent = stacked.gather(1, gather).squeeze(1)

    target_latent = net.representation(c_obs)
    target_proj = net.project(normalize_hidden(target_latent), with_predictor=False)
    pred = net.project(dyn_latent, with_predictor=True)

    return {
        "loss_cosine (what training reports, negated)": float(
            F.cosine_similarity(pred, target_proj, dim=-1).mean()
        ),
        "pairwise cos: target projections": pairwise_cosine(target_proj),
        "pairwise cos: predictor outputs": pairwise_cosine(pred),
        "pairwise cos: trunk latents (pre-projector)": pairwise_cosine(
            target_latent.flatten(1)
        ),
        "per-dim std of normalized target projections": float(
            F.normalize(target_proj, dim=-1).std(dim=0).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to load (ally net)"
    )
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument(
        "--tiny", action="store_true", help="tiny net (local sanity mode)"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="train N steps on synthetic games first (local collapse repro)",
    )
    args = parser.parse_args()

    cfg = MuZeroConfig(device="cpu")
    if args.tiny:
        cfg = replace(cfg, channels=16, repr_blocks=1, dyn_blocks=1, batch_size=16)
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        net.load_state_dict(state["ally"])
        print(
            f"loaded ally weights from {args.ckpt} (iteration {state.get('iteration', '?')})"
        )

    buf = build_synthetic_buffer(cfg)
    if args.train_steps > 0:
        from muzero.train import MuZeroTrainer

        trainer = MuZeroTrainer(cfg, net)
        for step in range(args.train_steps):
            losses = trainer.train_batch(buf.sample_batch(cfg.batch_size))
            if step % 10 == 0 or step == args.train_steps - 1:
                print(f"  step {step:3d} loss/consistency={losses['consistency']:+.4f}")

    stats = diagnose(net, buf.sample_batch(args.batch))
    print("\n--- consistency-branch diagnostic ---")
    for name, value in stats.items():
        print(f"{name}: {value:+.4f}")
    print(
        "\ninterpretation: 'pairwise cos: target projections' is the verdict.\n"
        "  < ~0.5  healthy — projections still depend on the input; a high\n"
        "          loss-cosine would then reflect genuine predictive matching.\n"
        "  > ~0.95 collapsed — every input maps to one direction; the -1.0\n"
        "          training loss is trivial and carries no learning signal.\n"
        "compare with the trunk row: diverse trunk + collapsed projections\n"
        "means the collapse lives in the projector/predictor (fix there);\n"
        "a collapsed trunk too would be a deeper representation problem."
    )


if __name__ == "__main__":
    main()
