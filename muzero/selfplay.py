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
        self.promotion_enabled = config.self_play_mode == "frozen_enemy"
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
            if (
                self.promotion_enabled
                and self.streak >= self.config.promote_after_consecutive_wins
            ):
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
        # Tracked-color cp after EVERY ply (both movers). Sampling only after
        # the tracked side's own moves biased the mean toward the sawtooth
        # trough (your blunder is priced in, the reply blunder is not).
        self.ally_cps = []
        self.blunders = 0  # moves whose mover-perspective eval dropped hard
        self.cp_moves = 0  # moves with engine eval data (blunder divisor)
        self.search_kls = []  # per-MCTS-move KL(visits || raw prior)


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
        self.latest_mode = config.self_play_mode == "latest"
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
        self,
        game: _Game,
        action: int,
        visits: dict,
        root_value: float,
        search_kl: float | None = None,
    ):
        h = game.history
        total = sum(visits.values())
        h.actions.append(action)
        h.policy_indices.append(np.array(list(visits.keys()), dtype=np.int64))
        h.policy_probs.append(
            np.array([v / total for v in visits.values()], dtype=np.float32)
        )
        h.root_values.append(float(root_value))
        if search_kl is not None:  # None on forced opening moves (no search)
            game.search_kls.append(float(search_kl))
        mover = game.env.side_to_move  # about to move
        if self.latest_mode or mover == game.env.ally_side:
            p = np.array([v / total for v in visits.values()], dtype=np.float64)
            game.ally_entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
        _, reward, done, info = game.env.step(index_to_move(action))
        h.rewards.append(reward)
        if info.get("mover_cp_delta") is not None:
            game.cp_moves += 1
            if info["mover_cp_delta"] <= -self.cfg.blunder_cp_threshold:
                game.blunders += 1
        if info.get("red_cp") is not None:
            mover_cp = info["red_cp"] if mover == "w" else -info["red_cp"]
            if self.latest_mode or mover == game.env.ally_side:
                # root_value is mover-perspective; pair it with mover-persp cp
                game.ally_value_cp_pairs.append((float(root_value), float(mover_cp)))
            ally_cp = info["red_cp"] if game.env.ally_side == "w" else -info["red_cp"]
            game.ally_cps.append(float(ally_cp))
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
        return {
            "result": env.result,
            "ally_side": env.ally_side,
            "ally_won": ally_won,
            "draw": draw,
            "plies": len(h),
            "truncated": env.truncated,
            "promoted": promoted,
            "era": self.coordinator.era,
            "mean_root_entropy": (
                float(np.mean(game.ally_entropies)) if game.ally_entropies else 0.0
            ),
            "value_cp_pairs": list(game.ally_value_cp_pairs),
            "mean_ally_cp": (float(np.mean(game.ally_cps)) if game.ally_cps else None),
            "games_this_era": self.coordinator.games_this_era,
            "blunders": game.blunders,
            "cp_moves": game.cp_moves,
            "mean_search_kl": (
                float(np.mean(game.search_kls)) if game.search_kls else None
            ),
        }

    def _round_groups(self, active: list) -> list:
        """Specs for this round's MCTS calls: (runner, side_filter, noise).

        side_filter None means "all active games" (latest mode)."""
        if self.latest_mode:
            return [(self.ally_runner, None, True)]
        return [(self.ally_runner, True, True), (self.enemy_runner, False, False)]

    def generate(self, num_games: int) -> list:
        """Play `num_games` to completion in lockstep; returns game summaries."""
        summaries = []
        active = [
            self._new_game() for _ in range(min(self.cfg.games_per_worker, num_games))
        ]
        while active:
            for runner, want_ally, add_noise in self._round_groups(active):
                group = [
                    g
                    for g in active
                    if want_ally is None
                    or (g.env.side_to_move == g.env.ally_side) == want_ally
                ]
                if not group:
                    continue
                roots = []
                for g in group:
                    legal = np.array(
                        [move_to_index(m) for m in g.env.legal_moves()], dtype=np.int64
                    )
                    roots.append((g.env.observation().astype(np.float32), legal))
                results = self.mcts.run(runner, roots, add_noise=add_noise)
                for g, (visits, root_value, search_kl) in zip(group, results):
                    action = select_action(
                        visits, g.env.plies, self.cfg.temperature_moves, self.rng
                    )
                    done, _ = self._record_and_step(
                        g, action, visits, root_value, search_kl=search_kl
                    )
                    if done:
                        summaries.append(self._finish(g))
                        active.remove(g)
                        if self.games_started < num_games:
                            active.append(self._new_game())
        return summaries
