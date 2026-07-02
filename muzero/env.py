"""Xiangqi env wrapper: gym_xiangqi applies moves; Pikafish is the sole
legality and evaluation source. Enforces repetition-draw and hopeless-game
truncation per the design spec."""

from __future__ import annotations

import gym
import numpy as np
from gym_xiangqi.utils import move_to_action_space

from muzero.config import MuZeroConfig
from muzero.encoding import encode_observation
from src.xiangqi_board import (
    algebraic_to_board_coords,
    board_to_uci_fen,
    engine_uci_to_algebraic,
)


class XiangqiEnv:
    def __init__(self, config: MuZeroConfig, evaluator):
        self.config = config
        self.evaluator = evaluator
        self._gym = None

    # -- lifecycle -----------------------------------------------------------

    def reset(self, ally_side: str = "w") -> np.ndarray:
        if self._gym is not None:
            self._gym.close()
        self._gym = gym.make("gym_xiangqi:xiangqi-v0")
        res = self._gym.reset()
        obs = res[0] if isinstance(res, tuple) else res
        self.board = self._extract_board(obs)
        self.side_to_move = "w"
        self.ally_side = ally_side
        self.plies = 0
        self.no_progress = 0
        self.result = None
        self.truncated = False
        self._rep_counts: dict = {}
        self._rep_cps: dict = {}
        self._sat_streak = 0
        # Per-state histories (index t = state before ply t), used to rebuild
        # observations in the replay buffer.
        self.boards = [self.board.copy()]
        self.to_play_history = ["w"]
        self.rep_history = [self._bump_repetition()]
        self.no_progress_history = [0]
        return self.board

    def _extract_board(self, obs) -> np.ndarray:
        state = getattr(self._gym.unwrapped, "state", None)
        if state is None:
            state = obs
        return np.array(state, dtype=np.int8).reshape(10, 9)

    # -- queries -------------------------------------------------------------

    def fen(self) -> str:
        return board_to_uci_fen(self.board, self.side_to_move)

    def legal_moves(self) -> list:
        engine_moves = self.evaluator.list_legal_moves(self.fen()) or []
        moves = []
        for u in engine_moves:
            m = engine_uci_to_algebraic(u)
            if m is not None:
                moves.append(m)
        return moves

    def red_cp(self):
        cp = self.evaluator.evaluate_cp(self.fen())
        if cp is None:
            return None
        return float(cp) if self.side_to_move == "w" else -float(cp)

    def observation(self) -> np.ndarray:
        return encode_observation(
            self.boards,
            self.side_to_move,
            self.rep_history[-1],
            self.no_progress,
            self.config.history_length,
        )

    # -- stepping ------------------------------------------------------------

    def step(self, move: str):
        """Apply an internal-algebraic move for the side to move.

        Returns (board, reward, done, info); reward is mover-perspective and
        includes shaping, terminal +/-1, and any repetition penalty."""
        assert self.result is None, "game already over"
        mover = self.side_to_move
        (fr, fc), (tr, tc) = algebraic_to_board_coords(move)
        # gym_xiangqi's action encoding wants the unsigned piece index
        # (1-16); the board stores it signed (negative = black).
        piece_id = abs(int(self.board[fr, fc]))
        captured = int(self.board[tr, tc]) != 0
        cp_before_red = self.red_cp()

        action = int(move_to_action_space(piece_id, (fr, fc), (tr, tc)))
        res = self._gym.step(action)
        if len(res) == 5:
            obs, _, term, trunc, _ = res
            gym_done = bool(term or trunc)
        else:
            obs, _, gym_done, _ = res
        self.board = self._extract_board(obs)
        self.side_to_move = "b" if mover == "w" else "w"
        self.plies += 1
        self.no_progress = 0 if captured else self.no_progress + 1
        rep = self._bump_repetition()
        self.boards.append(self.board.copy())
        self.to_play_history.append(self.side_to_move)
        self.rep_history.append(rep)
        self.no_progress_history.append(self.no_progress)

        cp_after_red = self.red_cp()
        reward = self._shaping_reward(mover, cp_before_red, cp_after_red)
        info = {
            "red_cp": cp_after_red,
            "truncated": False,
            "repetition_penalized": None,
        }

        if gym_done or not self.legal_moves():
            # King captured, or opponent has no legal move (mate/stalemate):
            # mover wins in Xiangqi.
            self.result = "red_win" if mover == "w" else "black_win"
            reward += 1.0
        elif rep >= 3 and self._no_threat():
            self.result = "draw_repetition"
            mover_cp = self._mover_cp(mover, cp_after_red)
            if mover_cp is not None and mover_cp >= self.config.repetition_cp_ok:
                reward += self.config.repetition_penalty
                info["repetition_penalized"] = mover
        elif self.plies >= self.config.max_game_plies:
            self.result = "draw_max_plies"
        elif self._check_truncation(mover, cp_after_red):
            self.result = "black_win" if self.ally_side == "w" else "red_win"
            self.truncated = True
            info["truncated"] = True
            reward += -1.0  # mover here is always the saturated ally

        info["result"] = self.result
        return self.board, float(reward), self.result is not None, info

    # -- internals -----------------------------------------------------------

    def _mover_cp(self, mover: str, red_cp):
        if red_cp is None:
            return None
        return red_cp if mover == "w" else -red_cp

    def _shaping_reward(self, mover: str, cp_before_red, cp_after_red) -> float:
        if cp_before_red is None or cp_after_red is None:
            return 0.0
        delta_red = cp_after_red - cp_before_red
        delta = delta_red if mover == "w" else -delta_red
        return self.config.shaping_weight * float(
            np.tanh(delta / self.config.shaping_cp_scale)
        )

    def _position_key(self) -> str:
        return self.fen().rsplit(" ", 2)[0]  # board + stm (drop move counters)

    def _bump_repetition(self) -> int:
        key = self._position_key()
        self._rep_counts[key] = self._rep_counts.get(key, 0) + 1
        cp = None
        if self._rep_counts[key] >= 2:  # only pay for cp once repeats start
            cp = self.red_cp()
        if cp is not None:
            self._rep_cps.setdefault(key, []).append(cp)
        return self._rep_counts[key]

    def _no_threat(self) -> bool:
        cps = self._rep_cps.get(self._position_key(), [])
        if len(cps) < 2:
            return True  # no eval signal -> treat as threat-free shuffle
        return (max(cps) - min(cps)) < self.config.repetition_swing_cp

    def _check_truncation(self, mover: str, cp_after_red) -> bool:
        if mover != self.ally_side:
            return False
        mover_cp = self._mover_cp(mover, cp_after_red)
        if mover_cp is None:
            return False
        if mover_cp <= self.config.truncation_cp:
            self._sat_streak += 1
        else:
            self._sat_streak = 0
        return self._sat_streak >= self.config.truncation_consecutive
