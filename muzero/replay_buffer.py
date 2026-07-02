"""Whole-game FIFO replay buffer with proportional prioritized sampling.

Observations are reconstructed from stored int8 boards at sample time
(storing encoded 115-plane stacks for 5000 games would need ~30 GB)."""

from __future__ import annotations

from collections import deque

import numpy as np

from muzero.config import MuZeroConfig
from muzero.encoding import encode_observation, material_balance


class GameHistory:
    """One finished game. Index t: boards/to_play/rep/no_progress have L+1
    entries (state before ply t, plus terminal state); the rest have L."""

    def __init__(self):
        self.boards = []
        self.to_play_history = []
        self.rep_history = []
        self.no_progress_history = []
        self.actions = []
        self.rewards = []  # mover-perspective, shaping + terminal + penalties
        self.policy_indices = []  # sparse root visit distributions
        self.policy_probs = []
        self.root_values = []  # mover-perspective MCTS root values
        self.result = None
        self.truncated = False
        self.ally_side = "w"

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.games: deque = deque(maxlen=config.buffer_games)
        self.priorities: deque = deque(maxlen=config.buffer_games)
        self.total_games_added = 0
        self.rng = np.random.default_rng(config.seed)

    # -- adding ---------------------------------------------------------------

    def add(self, game: GameHistory) -> None:
        pri = (
            np.array(
                [
                    abs(game.root_values[t] - self.n_step_value(game, t)) + 1e-3
                    for t in range(len(game))
                ],
                dtype=np.float32,
            )
            ** self.config.per_alpha
        )
        if game.truncated:
            tail = 2 * self.config.truncation_consecutive
            pri[-tail:] *= self.config.truncated_tail_weight
        self.games.append(game)
        self.priorities.append(pri)
        self.total_games_added += 1

    # -- targets ----------------------------------------------------------------

    def n_step_value(self, game: GameHistory, t: int) -> float:
        """Mover-perspective n-step return for state t. Rewards and bootstrap
        values alternate sign because each ply flips whose move it is."""
        cfg = self.config
        g = 0.0
        for j in range(cfg.td_steps):
            k = t + j
            if k >= len(game):
                return g
            g += (cfg.discount**j) * ((-1) ** j) * game.rewards[k]
        k = t + cfg.td_steps
        if k < len(game.root_values):
            g += (
                (cfg.discount**cfg.td_steps)
                * ((-1) ** cfg.td_steps)
                * game.root_values[k]
            )
        return g

    def _dense_policy(self, game: GameHistory, t: int) -> np.ndarray:
        dense = np.zeros(self.config.action_space, dtype=np.float32)
        dense[game.policy_indices[t]] = game.policy_probs[t]
        s = dense.sum()
        return dense / s if s > 0 else np.full_like(dense, 1.0 / dense.shape[0])

    def make_target(self, game: GameHistory, t: int) -> dict:
        cfg = self.config
        K, L = cfg.unroll_steps, len(game)
        uniform = np.full(cfg.action_space, 1.0 / cfg.action_space, dtype=np.float32)

        obs = encode_observation(
            game.boards[: t + 1],
            game.to_play_history[t],
            game.rep_history[t],
            game.no_progress_history[t],
            cfg.history_length,
        )
        actions = np.array(
            [
                (
                    game.actions[t + k]
                    if t + k < L
                    else int(self.rng.integers(0, cfg.action_space))
                )
                for k in range(K)
            ],
            dtype=np.int64,
        )
        policies, pmask, values, moves_left, material = [], [], [], [], []
        for k in range(K + 1):
            s = t + k
            if s < L:
                policies.append(self._dense_policy(game, s))
                pmask.append(1.0)
                values.append(self.n_step_value(game, s))
            else:
                policies.append(uniform)
                pmask.append(0.0)
                values.append(0.0)  # absorbing states train value to 0
            moves_left.append(min(max(L - s, 0), cfg.moves_left_max))
            material.append(material_balance(game.boards[min(s, L)]) / 10.0)
        rewards = np.array(
            [game.rewards[t + k] if t + k < L else 0.0 for k in range(K)],
            dtype=np.float32,
        )
        # SimSiam target: one random real future observation within the unroll.
        max_kc = min(K, L - t)
        k_c = int(self.rng.integers(1, max_kc + 1)) if max_kc >= 1 else 0
        c_obs = (
            encode_observation(
                game.boards[: t + k_c + 1],
                game.to_play_history[t + k_c],
                game.rep_history[t + k_c],
                game.no_progress_history[t + k_c],
                cfg.history_length,
            )
            if k_c > 0
            else np.zeros((cfg.input_planes, 10, 9), dtype=np.float32)
        )
        return {
            "obs": obs,
            "actions": actions,
            "target_policy": np.stack(policies),
            "policy_mask": np.array(pmask, dtype=np.float32),
            "target_value": np.array(values, dtype=np.float32),
            "target_reward": rewards,
            "target_moves_left": np.array(moves_left, dtype=np.int64),
            "target_material": np.array(material, dtype=np.float32),
            "consistency_obs": c_obs,
            "consistency_k": k_c,
        }

    # -- sampling ---------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> dict:
        flat, owners = [], []
        for gi, pri in enumerate(self.priorities):
            flat.append(pri)
            owners.append(np.full(len(pri), gi, dtype=np.int64))
        flat = np.concatenate(flat)
        owners = np.concatenate(owners)
        offsets = np.concatenate([[0], np.cumsum([len(p) for p in self.priorities])])
        probs = flat / flat.sum()
        picks = self.rng.choice(flat.shape[0], size=batch_size, p=probs)
        samples = [
            self.make_target(self.games[owners[i]], int(i - offsets[owners[i]]))
            for i in picks
        ]
        return {k: np.stack([s[k] for s in samples]) for k in samples[0]}

    def mean_game_length(self) -> float:
        return float(np.mean([len(g) for g in self.games])) if self.games else 0.0
