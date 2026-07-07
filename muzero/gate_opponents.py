"""Engine-free gate opponents (spec 2026-07-07-greedy-gate-rung-design)."""

from __future__ import annotations

import numpy as np

from muzero.encoding import PIECE_TYPE, PIECE_VALUE
from src.xiangqi_board import algebraic_to_board_coords

# PIECE_VALUE rates kings 0.0 (material-balance convention); for a greedy
# opponent, taking the king ends the game and must outrank any material.
_KING_CAPTURE_VALUE = 100.0


def greedy_capture_move(env, rng: np.random.Generator) -> str | None:
    """Highest-value capture if any (ties broken randomly), else a uniform
    random legal move. Returns None when there are no legal moves.

    Only reads `env.legal_moves()` and `env.board` (signed int8, +red)."""
    moves = env.legal_moves()
    if not moves:
        return None
    best_value = 0.0
    best_moves: list[str] = []
    for move in moves:
        (_, _), (tr, tc) = algebraic_to_board_coords(move)
        target = int(env.board[tr, tc])
        if target == 0:
            continue
        if abs(target) == 1:
            value = _KING_CAPTURE_VALUE
        else:
            value = PIECE_VALUE[PIECE_TYPE[abs(target)]]
        if value > best_value:
            best_value, best_moves = value, [move]
        elif value == best_value:
            best_moves.append(move)
    pool = best_moves if best_moves else moves
    return str(rng.choice(pool))
