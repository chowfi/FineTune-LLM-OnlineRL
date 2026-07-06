"""Human vs MuZero session on muzero's own env (spec 2026-07-06).

Snapshot contract mirrors web.server.game_session.GameSession so the same
board.js works: `lastAllyMove` = the HUMAN's move and `lastEngineMove` =
the model's move, whichever color each is playing."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from src.xiangqi_board import board_to_fen, board_to_graphic
from web.server.board_view import board_grid
from web.server.muzero_player import MuZeroPlayer

HumanSide = Literal["red", "black"]

_RESULT_WINNER = {
    "red_win": "red",
    "black_win": "black",
    "draw_repetition": "draw",
    "draw_max_plies": "draw",
}


class MuZeroGameSession:
    engine_kind = "muzero"

    def __init__(
        self,
        pikafish,
        player: Optional[MuZeroPlayer],
        config: Optional[MuZeroConfig] = None,
    ):
        # Humans get to finish (or save) lost games: disable the training-time
        # hopeless-cp auto-adjudication. Repetition draws + ply cap remain.
        # replace() copies, so a caller-shared config is never mutated.
        cfg = (
            replace(config, truncation_consecutive=10**9)
            if config
            else MuZeroConfig(truncation_consecutive=10**9)
        )
        self.cfg = cfg
        self.pikafish = pikafish
        self.player = player
        self.env = XiangqiEnv(cfg, pikafish)
        self.human_side: HumanSide = "red"
        self.game_over = False
        self.winner: Optional[str] = None
        self.last_ally_move: Optional[str] = None  # human's last move
        self.last_engine_move: Optional[str] = None  # model's last move
        # Kept for snapshot parity with GameSession; under the current
        # sync-in-async runtime it is never observably True from /api/state.
        self.engine_thinking = False
        self.reset(human_side="red")

    def _human_to_move(self) -> bool:
        return self.env.side_to_move == ("w" if self.human_side == "red" else "b")

    def reset(self, human_side: HumanSide = "red") -> Dict[str, Any]:
        self.human_side = human_side if human_side in ("red", "black") else "red"
        self.env.reset(ally_side="w" if self.human_side == "red" else "b")
        self.game_over = False
        self.winner = None
        self.last_ally_move = None
        self.last_engine_move = None
        self.engine_thinking = False
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        if self.game_over:
            turn = "none"
        elif self._human_to_move():
            turn = "human"
        else:
            turn = "engine"
        return {
            "board": board_grid(self.env.board),
            "graphic": board_to_graphic(self.env.board),
            "fen": board_to_fen(self.env.board),
            "allyMode": "human",
            "engineKind": self.engine_kind,
            "humanSide": self.human_side,
            "sideToMove": "ally" if self._human_to_move() else "enemy",
            "turn": turn,
            "gameOver": self.game_over,
            "winner": self.winner,
            "lastAllyMove": self.last_ally_move,
            "lastEngineMove": self.last_engine_move,
            "engineThinking": self.engine_thinking,
        }

    def legal_targets_from(self, from_sq: str) -> List[str]:
        if self.game_over or not self._human_to_move():
            return []
        from_sq = (from_sq or "").strip().lower()
        if len(from_sq) != 2:
            return []
        return sorted({m[2:] for m in self.env.legal_moves() if m.startswith(from_sq)})

    def _finish_if_over(self) -> None:
        if self.env.result is not None:
            self.game_over = True
            self.winner = _RESULT_WINNER.get(self.env.result, "draw")

    def apply_human_move(self, move: str) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if not self._human_to_move():
            return self.snapshot(), "Not your turn"
        move = (move or "").strip().lower()
        if move not in self.env.legal_moves():
            return self.snapshot(), "Move is not legal (Pikafish)"
        self.env.step(move)
        self.last_ally_move = move
        self._finish_if_over()
        return self.snapshot(), None

    def apply_engine_move(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if self._human_to_move():
            return self.snapshot(), "Not engine turn"
        if self.player is None:
            return self.snapshot(), "MuZero engine not loaded"
        self.engine_thinking = True
        try:
            try:
                move = self.player.choose_move(self.env)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI as a 400
                # Clear before snapshotting: a return expression is evaluated
                # before `finally` runs, so the flag must be reset here.
                self.engine_thinking = False
                return self.snapshot(), f"Engine error: {exc}"
            if move not in self.env.legal_moves():
                return self.snapshot(), f"Engine produced illegal move {move!r}"
            self.env.step(move)
            self.last_engine_move = move
            self._finish_if_over()
        finally:
            self.engine_thinking = False
        return self.snapshot(), None

    def apply_greedy_ally(self) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.snapshot(), "Greedy ally is not supported with the MuZero engine"
