"""In-memory Xiangqi game: Red ally (human or greedy) vs LoRA engine Black."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import gym
import gym_xiangqi  # noqa: F401
import numpy as np

from src.pikafish_eval import PikafishEvaluator
from web.server.board_view import board_grid
from web.server.engine_player import EnginePlayer
from web.server.flip_utils import action_to_algebraic
from web.server.greedy_agent import pick_greedy_ally_move
from src.xiangqi_board import (
    algebraic_to_board_coords,
    board_to_fen,
    board_to_graphic,
    board_to_uci_fen,
    engine_uci_to_algebraic,
)

AllyMode = Literal["human", "greedy"]


def _mask_actions_from_pikafish(
    board_state: np.ndarray,
    env: gym.Env,
    pikafish: PikafishEvaluator,
    *,
    ally: bool,
) -> Tuple[np.ndarray, bool]:
    attr = "ally_actions" if ally else "enemy_actions"
    env_legal = np.where(getattr(env, attr) == 1)[0]
    if not pikafish.enabled or len(env_legal) == 0:
        return env_legal, False

    stm = "w" if ally else "b"
    fen_before = board_to_uci_fen(board_state, side_to_move=stm)
    engine_legal = pikafish.list_legal_moves(fen_before)
    if not engine_legal:
        return env_legal, False

    attr_actions = np.where(getattr(env, attr) == 1)[0]
    alg_to_action = {action_to_algebraic(int(a)): int(a) for a in attr_actions}
    engine_actions: List[int] = []
    for move_str in engine_legal:
        alg = engine_uci_to_algebraic(move_str)
        if alg and alg in alg_to_action:
            engine_actions.append(alg_to_action[alg])

    if not engine_actions:
        return env_legal, False

    unique_actions = np.array(sorted(set(engine_actions)), dtype=int)
    getattr(env, attr).fill(0)
    getattr(env, attr)[unique_actions] = 1
    return unique_actions, True


def _algebraic_legals_for_side(
    board_state: np.ndarray,
    env: gym.Env,
    pikafish: PikafishEvaluator,
    *,
    ally: bool,
) -> List[str]:
    _mask_actions_from_pikafish(board_state, env, pikafish, ally=ally)
    attr = "ally_actions" if ally else "enemy_actions"
    actions = np.where(getattr(env, attr) == 1)[0]
    return [action_to_algebraic(int(a)) for a in actions]


def _winner_from_rewards(ally_reward: float, enemy_reward: float) -> Optional[str]:
    if ally_reward >= 100:
        return "red"
    if enemy_reward >= 100:
        return "black"
    return None


class GameSession:
    def __init__(
        self,
        pikafish: PikafishEvaluator,
        engine: Optional[EnginePlayer] = None,
    ):
        self.pikafish = pikafish
        self.engine = engine
        self.env = gym.make("xiangqi-v0")
        self.ally_mode: AllyMode = "human"
        self.game_over = False
        self.winner: Optional[str] = None
        self.last_ally_move: Optional[str] = None
        self.last_engine_move: Optional[str] = None
        self.engine_thinking = False
        # gym-xiangqi keeps both ally_actions and enemy_actions populated after
        # a step; track the real side to move explicitly.
        self.side_to_move: Literal["ally", "enemy"] = "ally"
        self.reset(ally_mode="human")

    def _is_ally_turn(self) -> bool:
        return not self.game_over and self.side_to_move == "ally"

    def _is_engine_turn(self) -> bool:
        return not self.game_over and self.side_to_move == "enemy"

    def reset(self, ally_mode: AllyMode = "human") -> Dict[str, Any]:
        self.env.reset()
        self.ally_mode = ally_mode if ally_mode in ("human", "greedy") else "human"
        self.game_over = False
        self.winner = None
        self.last_ally_move = None
        self.last_engine_move = None
        self.engine_thinking = False
        self.side_to_move = "ally"
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        if self.game_over:
            turn = "none"
        elif self.side_to_move == "enemy":
            turn = "engine"
        elif self.side_to_move == "ally":
            turn = "greedy" if self.ally_mode == "greedy" else "human"
        else:
            turn = "none"

        return {
            "board": board_grid(self.env.state),
            "graphic": board_to_graphic(self.env.state),
            "fen": board_to_fen(self.env.state),
            "allyMode": self.ally_mode,
            "sideToMove": self.side_to_move,
            "turn": turn,
            "gameOver": self.game_over,
            "winner": self.winner,
            "lastAllyMove": self.last_ally_move,
            "lastEngineMove": self.last_engine_move,
            "engineThinking": self.engine_thinking,
            # Fixed session metadata for the UI (not derived from ally_mode):
            # the LLM session always seats the ally as Red.
            "engineKind": "llm",
            "humanSide": "red",
        }

    def legal_targets_from(self, from_sq: str) -> List[str]:
        if self.game_over or self.ally_mode != "human" or not self._is_ally_turn():
            return []
        from_sq = (from_sq or "").strip().lower()
        if len(from_sq) != 2:
            return []
        legals = _algebraic_legals_for_side(
            self.env.state, self.env, self.pikafish, ally=True
        )
        targets: List[str] = []
        for mv in legals:
            if mv.startswith(from_sq):
                targets.append(mv[2:])
        return sorted(set(targets))

    def _action_for_ally_move(self, move: str) -> Tuple[bool, str, Optional[int]]:
        move = (move or "").strip().lower()
        parsed = algebraic_to_board_coords(move)
        if parsed is None:
            return False, "Invalid move format (expected e.g. b7b4)", None

        legals = _algebraic_legals_for_side(
            self.env.state, self.env, self.pikafish, ally=True
        )
        if move not in legals:
            return False, "Move is not legal (Pikafish)", None

        for a in np.where(self.env.ally_actions == 1)[0]:
            if action_to_algebraic(int(a)) == move:
                return True, "", int(a)
        return False, "Move not in ally action space", None

    def _apply_ally_action(
        self, action: int, move_str: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        _, ally_reward, done, _ = self.env.step(int(action))
        self.last_ally_move = move_str
        win = _winner_from_rewards(float(ally_reward), 0.0)
        if done or win:
            self.game_over = True
            self.winner = win or ("red" if float(ally_reward) >= 100 else "black")
        else:
            self.side_to_move = "enemy"
        return self.snapshot(), None

    def apply_human_move(self, move: str) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if self.ally_mode != "human":
            return self.snapshot(), "Ally mode is not human"
        if not self._is_ally_turn():
            return self.snapshot(), "Not ally turn"

        ok, err, action = self._action_for_ally_move(move)
        if not ok or action is None:
            return self.snapshot(), err
        return self._apply_ally_action(action, move)

    def apply_greedy_ally(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if self.ally_mode != "greedy":
            return self.snapshot(), "Ally mode is not greedy"
        if not self._is_ally_turn():
            return self.snapshot(), "Not ally turn"

        _mask_actions_from_pikafish(self.env.state, self.env, self.pikafish, ally=True)
        try:
            action, _tag = pick_greedy_ally_move(self.env)
        except RuntimeError as exc:
            return self.snapshot(), str(exc)

        move_str = action_to_algebraic(action)
        legals = _algebraic_legals_for_side(
            self.env.state, self.env, self.pikafish, ally=True
        )
        if move_str not in legals:
            return self.snapshot(), f"Greedy ally produced illegal move {move_str!r}"

        return self._apply_ally_action(action, move_str)

    def apply_engine_move(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if not self._is_engine_turn():
            return self.snapshot(), "Not engine turn"
        if self.engine is None:
            return self.snapshot(), "Engine not loaded"

        self.engine_thinking = True
        try:
            # Restrict scoring to Pikafish-legal enemy moves (gym alone is wider).
            _mask_actions_from_pikafish(
                self.env.state, self.env, self.pikafish, ally=False
            )
            engine_move, engine_action = self.engine.choose_black_move(
                self.env, self.last_ally_move
            )
            legals = [
                action_to_algebraic(int(a))
                for a in np.where(self.env.enemy_actions == 1)[0]
            ]
            if engine_move not in legals:
                return (
                    self.snapshot(),
                    f"Engine produced illegal move {engine_move!r}",
                )

            _, enemy_reward, done, _ = self.env.step(engine_action)
            self.last_engine_move = engine_move
            win = _winner_from_rewards(0.0, float(enemy_reward))
            if done or win:
                self.game_over = True
                self.winner = win or ("black" if float(enemy_reward) >= 100 else "red")
            else:
                self.side_to_move = "ally"
        finally:
            self.engine_thinking = False

        return self.snapshot(), None


def build_pikafish() -> PikafishEvaluator:
    binary = os.environ.get("PIKAFISH_BIN", "").strip()
    if not binary:
        raise RuntimeError(
            "Set PIKAFISH_BIN to your Pikafish executable, e.g. "
            "export PIKAFISH_BIN=/home/fchow/bin/pikafish"
        )
    eng = PikafishEvaluator(binary, depth=8, movetime_ms=200, verbose=False)
    if not eng.enabled:
        raise RuntimeError(f"Pikafish failed to start: {binary}")
    return eng
