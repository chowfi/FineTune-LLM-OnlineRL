"""Training: K-step unrolled combined loss + the main orchestration loop."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet, normalize_hidden
from muzero.transforms import h_transform, scalar_to_support


def scale_gradient(t: torch.Tensor, scale: float) -> torch.Tensor:
    return t * scale + t.detach() * (1.0 - scale)


def _soft_ce(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(-1)


class MuZeroTrainer:
    def __init__(self, cfg: MuZeroConfig, net: MuZeroNet):
        self.cfg = cfg
        self.net = net
        self.optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.train_steps = 0

    def _to_tensors(self, batch: dict) -> dict:
        dev = next(self.net.parameters()).device
        out = {}
        for k, v in batch.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("actions", "target_moves_left", "consistency_k"):
                t = t.long()
            else:
                t = t.float()
            out[k] = t.to(dev)
        return out

    def train_batch(self, batch: dict) -> dict:
        cfg = self.cfg
        mean_buffer_age = float(batch.pop("mean_buffer_age", 0.0))
        self.net.train()
        b = self._to_tensors(batch)
        B, K = b["actions"].shape

        value_support = scalar_to_support(
            h_transform(b["target_value"]),
            -cfg.value_max,
            cfg.value_max,
            cfg.value_bins,
        )
        reward_support = scalar_to_support(
            b["target_reward"], -cfg.reward_max, cfg.reward_max, cfg.reward_bins
        )

        out = self.net.initial_inference(b["obs"])
        losses = {
            "policy": (
                _soft_ce(out["policy_logits"], b["target_policy"][:, 0])
                * b["policy_mask"][:, 0]
            ).sum()
            / b["policy_mask"][:, 0].sum().clamp(min=1.0),
            "value": _soft_ce(out["value_logits"], value_support[:, 0]).mean(),
            "reward": torch.zeros((), device=b["obs"].device),
            "moves_left": F.cross_entropy(
                out["moves_left_logits"], b["target_moves_left"][:, 0]
            ),
            "material": F.mse_loss(out["material"], b["target_material"][:, 0]),
        }
        hidden = out["hidden"]
        latents = []
        for k in range(1, K + 1):
            out_k = self.net.recurrent_inference(hidden, b["actions"][:, k - 1])
            hidden = scale_gradient(out_k["hidden"], 0.5)
            latents.append(hidden)
            pm = b["policy_mask"][:, k]
            losses["policy"] = (
                losses["policy"]
                + (
                    _soft_ce(out_k["policy_logits"], b["target_policy"][:, k]) * pm
                ).sum()
                / pm.sum().clamp(min=1.0)
                / K
            )
            losses["value"] = (
                losses["value"]
                + _soft_ce(out_k["value_logits"], value_support[:, k]).mean() / K
            )
            losses["reward"] = (
                losses["reward"]
                + _soft_ce(out_k["reward_logits"], reward_support[:, k - 1]).mean() / K
            )
            losses["moves_left"] = (
                losses["moves_left"]
                + F.cross_entropy(
                    out_k["moves_left_logits"], b["target_moves_left"][:, k]
                )
                / K
            )
            losses["material"] = (
                losses["material"]
                + F.mse_loss(out_k["material"], b["target_material"][:, k]) / K
            )

        # SimSiam consistency at one sampled unroll offset per sample.
        k_c = b["consistency_k"]
        mask = (k_c > 0).float()
        if mask.sum() > 0:
            stacked = torch.stack(latents, dim=1)  # (B, K, ch, 10, 9)
            gather = (
                (k_c.clamp(min=1) - 1)
                .view(B, 1, 1, 1, 1)
                .expand(-1, 1, *stacked.shape[2:])
            )
            dyn_latent = stacked.gather(1, gather).squeeze(1)
            with torch.no_grad():
                self.net.eval()  # keep BN running stats clean for the target branch
                target_latent = self.net.representation(b["consistency_obs"])
                target_proj = self.net.project(
                    normalize_hidden(target_latent), with_predictor=False
                )
                self.net.train()
            pred = self.net.project(dyn_latent, with_predictor=True)
            cos = F.cosine_similarity(pred, target_proj.detach(), dim=-1)
            losses["consistency"] = -(cos * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            losses["consistency"] = torch.zeros((), device=b["obs"].device)

        w = cfg.loss_weights
        total = (
            w[0] * losses["policy"]
            + w[1] * losses["value"]
            + w[2] * losses["reward"]
            + w[3] * losses["moves_left"]
            + w[4] * losses["material"]
            + w[5] * losses["consistency"]
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.train_steps += 1
        result = {k: float(v.detach()) for k, v in losses.items()}
        result["total"] = float(total.detach())
        result["buffer_age"] = mean_buffer_age
        return result


def run_gate(cfg: MuZeroConfig, runner, evaluator) -> dict:
    """Ally (MCTS, no noise, argmax) vs raw Pikafish at gate movetime."""
    from muzero.encoding import index_to_move, move_to_index
    from muzero.env import XiangqiEnv
    from muzero.mcts import MCTS
    from muzero.warmstart import SimpleUciEngine
    from src.xiangqi_board import engine_uci_to_algebraic

    engine = SimpleUciEngine(cfg.pikafish_bin, cfg.gate_movetime_ms, multipv=1)
    mcts = MCTS(cfg)
    wins = draws = 0
    try:
        for i in range(cfg.gate_games):
            ally_side = "w" if i % 2 == 0 else "b"
            env = XiangqiEnv(cfg, evaluator)
            env.reset(ally_side=ally_side)
            done = False
            while not done:
                if env.side_to_move == ally_side:
                    legal = np.array(
                        [move_to_index(m) for m in env.legal_moves()], dtype=np.int64
                    )
                    ((visits, _),) = mcts.run(
                        runner,
                        [(env.observation().astype(np.float32), legal)],
                        add_noise=False,
                    )
                    move = index_to_move(max(visits, key=visits.get))
                else:
                    lines = engine.search(env.fen())
                    if not lines:
                        break
                    move = engine_uci_to_algebraic(lines[0][0])
                _, _, done, _ = env.step(move)
            if env.result in ("red_win", "black_win"):
                if (env.result == "red_win") == (ally_side == "w"):
                    wins += 1
            else:
                draws += 1
    finally:
        engine.close()
    n = cfg.gate_games
    return {
        "gate/win_rate": wins / n,
        "gate/draw_rate": draws / n,
        "gate/loss_rate": (n - wins - draws) / n,
    }


def main():
    import argparse
    import copy
    import os

    from src.pikafish_eval import PikafishEvaluator

    from muzero.mcts import NetRunner
    from muzero.metrics import MetricsLogger, aggregate_game_summaries
    from muzero.replay_buffer import ReplayBuffer
    from muzero.selfplay import SelfPlayCoordinator, SelfPlayWorker
    from muzero.warmstart import generate_warmstart_games

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    args = parser.parse_args()

    cfg = MuZeroConfig()
    if args.device:
        cfg.device = args.device
    if args.smoke:
        cfg.channels, cfg.repr_blocks, cfg.dyn_blocks = 16, 1, 1
        cfg.num_simulations, cfg.interior_topk = 8, 8
        cfg.num_workers, cfg.games_per_worker = 1, 2
        cfg.max_game_plies, cfg.batch_size = 6, 4
        cfg.warmstart_plies, cfg.warmstart_train_batches = 8, 2
        cfg.games_per_train_loop, cfg.gate_every_loops = 2, 10**9

    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    ally = MuZeroNet(cfg).to(device)
    enemy = copy.deepcopy(ally).to(device)
    enemy.eval()
    trainer = MuZeroTrainer(cfg, ally)
    start_iteration = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ally.load_state_dict(ckpt["ally"])
        enemy.load_state_dict(ckpt["enemy"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"]

    buffer = ReplayBuffer(cfg)
    ally_runner, enemy_runner = NetRunner(ally, device), NetRunner(enemy, device)
    # Runners are created before the coordinator so the coordinator can be
    # handed the enemy runner's lock: promotion must not swap weights
    # mid-forward-pass (see Task 10).
    coordinator = SelfPlayCoordinator(cfg, ally, enemy, enemy_lock=enemy_runner.lock)
    if ckpt is not None:
        coordinator.era = ckpt.get("era", 0)
        coordinator.streak = ckpt.get("streak", 0)

    def make_evaluator():
        return PikafishEvaluator(
            binary_path=cfg.pikafish_bin,
            depth=cfg.pikafish_depth,
            timeout_sec=cfg.pikafish_timeout_sec,
            movetime_ms=cfg.pikafish_movetime_ms,
            verbose=False,
        )

    workers = [
        SelfPlayWorker(
            cfg,
            ally_runner,
            enemy_runner,
            buffer,
            coordinator,
            make_evaluator(),
            worker_id=w,
        )
        for w in range(cfg.num_workers)
    ]
    gate_evaluator = make_evaluator()
    logger = MetricsLogger(cfg, enabled=not args.no_wandb)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # buffer is not persisted; resumed runs skip warmstart and refill from
    # live self-play
    if start_iteration == 0 and not buffer.games:
        print("[warmstart] generating Pikafish games ...")
        stats = generate_warmstart_games(cfg, buffer, workers[0].evaluator)
        print(f"[warmstart] {stats['games']} games / {stats['plies']} plies")
        for _ in range(cfg.warmstart_train_batches):
            trainer.train_batch(buffer.sample_batch(cfg.batch_size))

    import threading

    for it in range(start_iteration, args.iterations):
        # -- generate --
        games_per_worker_now = cfg.games_per_worker
        results, threads = [], []
        for w in workers:
            t = threading.Thread(
                target=lambda w=w: results.extend(w.generate(games_per_worker_now))
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        metrics = aggregate_game_summaries(results)

        # -- train: ~games_per_train_loop games' worth of positions --
        num_batches = max(
            1,
            int(cfg.games_per_train_loop * buffer.mean_game_length() // cfg.batch_size),
        )
        loss_sums: dict = {}
        for _ in range(num_batches):
            for k, v in trainer.train_batch(
                buffer.sample_batch(cfg.batch_size)
            ).items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
        metrics.update({f"loss/{k}": v / num_batches for k, v in loss_sums.items()})
        metrics.update(
            {
                "buffer/games": len(buffer.games),
                "buffer/mean_game_length": buffer.mean_game_length(),
                "buffer/total_games_added": buffer.total_games_added,
                "train/batches": num_batches,
                "train/steps": trainer.train_steps,
            }
        )

        # -- gate --
        if (it + 1) % cfg.gate_every_loops == 0:
            metrics.update(run_gate(cfg, ally_runner, gate_evaluator))

        logger.log(metrics, step=it)
        print(
            f"[iter {it}] "
            + " ".join(
                f"{k}={v:.3f}"
                for k, v in sorted(metrics.items())
                if isinstance(v, float)
            )
        )
        ckpt_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
        tmp_path = ckpt_path + ".tmp"
        torch.save(
            {
                "ally": ally.state_dict(),
                "enemy": enemy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": it + 1,
                "era": coordinator.era,
                "streak": coordinator.streak,
            },
            tmp_path,
        )
        os.replace(tmp_path, ckpt_path)


if __name__ == "__main__":
    main()
