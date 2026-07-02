"""Self-play: lockstep multi-game workers, opening book, frozen-enemy promotion.

Workers are threads (engine I/O and GPU calls release the GIL); each worker
owns one PikafishEvaluator and `games_per_worker` concurrent games. The ally
net moves for `ally_side`; the frozen enemy net moves for the other side."""

from __future__ import annotations

import threading

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.encoding import index_to_move, move_to_index
from muzero.env import XiangqiEnv
from muzero.mcts import MCTS, NetRunner
from muzero.replay_buffer import GameHistory, ReplayBuffer
from src.xiangqi_board import engine_uci_to_algebraic


def select_action(visits: dict, ply: int, temperature_moves: int, rng) -> int:
    actions = np.array(list(visits.keys()), dtype=np.int64)
    counts = np.array(list(visits.values()), dtype=np.float64)
    if ply >= temperature_moves:
        return int(actions[np.argmax(counts)])
    probs = counts / counts.sum()
    return int(rng.choice(actions, p=probs))


class SelfPlayCoordinator:
    """Tracks consecutive ally wins across all workers; promotes the enemy."""

    def __init__(self, config: MuZeroConfig, ally_net, enemy_net, enemy_lock=None):
        self.config = config
        self.ally_net = ally_net
        self.enemy_net = enemy_net
        self.lock = threading.Lock()
        # Serializes the weight swap against enemy-net inference; pass the
        # enemy NetRunner's lock so promotion can't tear concurrent reads.
        self.enemy_lock = enemy_lock if enemy_lock is not None else threading.Lock()
        self.streak = 0
        self.era = 0
        self.games_this_era = 0

    def report_result(self, ally_won: bool, draw: bool) -> bool:
        with self.lock:
            self.games_this_era += 1
            self.streak = self.streak + 1 if (ally_won and not draw) else 0
            if self.streak >= self.config.promote_after_consecutive_wins:
                with self.enemy_lock, torch.no_grad():
                    self.enemy_net.load_state_dict(self.ally_net.state_dict())
                self.enemy_net.eval()
                self.streak = 0
                self.era += 1
                self.games_this_era = 0
                return True
        return False


class _Game:
    def __init__(self, env: XiangqiEnv, history: GameHistory, opening_uci: str):
        self.env = env
        self.history = history
        self.opening_uci = opening_uci
        self.ally_entropies = []
        self.ally_value_cp_pairs = []


class SelfPlayWorker:
    def __init__(
        self,
        config,
        ally_runner: NetRunner,
        enemy_runner: NetRunner,
        buffer: ReplayBuffer,
        coordinator: SelfPlayCoordinator,
        evaluator,
        worker_id: int,
    ):
        self.cfg = config
        self.ally_runner = ally_runner
        self.enemy_runner = enemy_runner
        self.buffer = buffer
        self.coordinator = coordinator
        self.evaluator = evaluator
        self.worker_id = worker_id
        self.rng = np.random.default_rng(config.seed + worker_id + 1)
        self.mcts = MCTS(config, rng=self.rng)  # seeded noise, thread-local RNG
        self.games_started = 0

    def _new_game(self) -> _Game:
        ally_side = "w" if self.games_started % 2 == 0 else "b"
        opening = self.cfg.opening_book[self.games_started % len(self.cfg.opening_book)]
        self.games_started += 1
        env = XiangqiEnv(self.cfg, self.evaluator)
        env.reset(ally_side=ally_side)
        history = GameHistory()
        history.ally_side = ally_side
        game = _Game(env, history, opening)
        self._play_forced_opening(game)
        return game

    def _play_forced_opening(self, game: _Game):
        move = engine_uci_to_algebraic(game.opening_uci)
        idx = move_to_index(move)
        self._record_and_step(game, idx, {idx: 1}, root_value=0.0)

    def _record_and_step(
        self, game: _Game, action: int, visits: dict, root_value: float
    ):
        h = game.history
        total = sum(visits.values())
        h.actions.append(action)
        h.policy_indices.append(np.array(list(visits.keys()), dtype=np.int64))
        h.policy_probs.append(
            np.array([v / total for v in visits.values()], dtype=np.float32)
        )
        h.root_values.append(float(root_value))
        if game.env.side_to_move == game.env.ally_side:  # ally is about to move
            p = np.array([v / total for v in visits.values()], dtype=np.float64)
            game.ally_entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
        _, reward, done, info = game.env.step(index_to_move(action))
        h.rewards.append(reward)
        if (
            info.get("red_cp") is not None
            and game.env.side_to_move != game.env.ally_side
        ):
            # the ALLY just moved (side flipped); pair its root value with the engine eval
            ally_cp = info["red_cp"] if game.env.ally_side == "w" else -info["red_cp"]
            game.ally_value_cp_pairs.append((float(root_value), float(ally_cp)))
        return done, info

    def _finish(self, game: _Game) -> dict:
        env, h = game.env, game.history
        h.boards = [b.copy() for b in env.boards]
        h.to_play_history = list(env.to_play_history)
        h.rep_history = list(env.rep_history)
        h.no_progress_history = list(env.no_progress_history)
        h.result = env.result
        h.truncated = env.truncated
        ally_won = (env.result == "red_win") == (
            env.ally_side == "w"
        ) and env.result in ("red_win", "black_win")
        draw = env.result not in ("red_win", "black_win")
        promoted = self.coordinator.report_result(ally_won=ally_won, draw=draw)
        self.buffer.add(h)
        final_red_cp = env.red_cp()
        return {
            "result": env.result,
            "ally_side": env.ally_side,
            "ally_won": ally_won,
            "draw": draw,
            "plies": len(h),
            "truncated": env.truncated,
            "promoted": promoted,
            "final_red_cp": final_red_cp,
            "era": self.coordinator.era,
            "mean_root_entropy": (
                float(np.mean(game.ally_entropies)) if game.ally_entropies else 0.0
            ),
            "value_cp_pairs": list(game.ally_value_cp_pairs),
            "mean_ally_cp": (
                float(np.mean([cp for _, cp in game.ally_value_cp_pairs]))
                if game.ally_value_cp_pairs
                else None
            ),
            "games_this_era": self.coordinator.games_this_era,
        }

    def generate(self, num_games: int) -> list:
        """Play `num_games` to completion in lockstep; returns game summaries."""
        summaries = []
        active = [
            self._new_game() for _ in range(min(self.cfg.games_per_worker, num_games))
        ]
        while active:
            for runner, want_ally in (
                (self.ally_runner, True),
                (self.enemy_runner, False),
            ):
                group = [
                    g
                    for g in active
                    if (g.env.side_to_move == g.env.ally_side) == want_ally
                ]
                if not group:
                    continue
                roots = []
                for g in group:
                    legal = np.array(
                        [move_to_index(m) for m in g.env.legal_moves()], dtype=np.int64
                    )
                    roots.append((g.env.observation().astype(np.float32), legal))
                results = self.mcts.run(runner, roots, add_noise=want_ally)
                for g, (visits, root_value) in zip(group, results):
                    action = select_action(
                        visits, g.env.plies, self.cfg.temperature_moves, self.rng
                    )
                    done, _ = self._record_and_step(g, action, visits, root_value)
                    if done:
                        summaries.append(self._finish(g))
                        active.remove(g)
                        if self.games_started < num_games:
                            active.append(self._new_game())
        return summaries
