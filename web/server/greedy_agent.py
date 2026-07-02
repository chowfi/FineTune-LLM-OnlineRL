"""Capture-greedy ally mover (epsilon=0), matching GreedyEnemyAgent without curriculum."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from gym import Env
from gym_xiangqi.constants import PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move


def pick_greedy_ally_move(env: Env) -> Tuple[int, str]:
    """Return ``(action_id, policy_tag)`` for the side to move on ``ally_actions``."""
    actions = np.where(env.ally_actions == 1)[0]
    if len(actions) == 0:
        raise RuntimeError("No legal ally moves for greedy agent")

    board = env.state
    best_score = -1.0
    best_actions: list[int] = []
    for action in actions:
        _, _, end = action_space_to_move(int(action))
        target = int(board[end[0]][end[1]])
        # Ally pieces are positive; captures are negative (enemy) piece ids.
        score = float(PIECE_POINTS[abs(target)]) if target < 0 else 0.0
        if score > best_score:
            best_score = score
            best_actions = [int(action)]
        elif score == best_score:
            best_actions.append(int(action))
    return int(random.choice(best_actions)), "greedy"
