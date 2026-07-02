"""Board flip helpers for engine (Black) moves — mirrors LLM_RL_agent_FSDP_v2."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from gym import Env
from gym_xiangqi.utils import action_space_to_move, move_to_action_space

from xiangqi_board import algebraic_to_board_coords, board_coords_to_algebraic


def action_to_algebraic(action: int) -> str:
    _, start, end = action_space_to_move(int(action))
    return board_coords_to_algebraic(start[0], start[1], end[0], end[1])


def flip_move(move_str: str) -> str:
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return move_str
    (from_row, from_col), (to_row, to_col) = parsed
    return board_coords_to_algebraic(9 - from_row, from_col, 9 - to_row, to_col)


def get_flipped_enemy_legal_actions(env: Env) -> Tuple[List[int], Dict[int, int]]:
    """Map each legal enemy action to its flipped-board action id."""
    original_actions = np.where(env.enemy_actions == 1)[0]
    flipped_actions: List[int] = []
    flipped_to_original: Dict[int, int] = {}
    flipped_board = -env.state[::-1, :]
    for orig_act in original_actions:
        orig_move_str = action_to_algebraic(int(orig_act))
        flipped_move_str = flip_move(orig_move_str)
        parsed = algebraic_to_board_coords(flipped_move_str)
        if parsed is None:
            continue
        (from_row, from_col), (to_row, to_col) = parsed
        piece_id = int(flipped_board[from_row][from_col])
        if piece_id > 0:
            flipped_act = int(
                move_to_action_space(piece_id, (from_row, from_col), (to_row, to_col))
            )
            flipped_actions.append(flipped_act)
            flipped_to_original[flipped_act] = int(orig_act)
    return flipped_actions, flipped_to_original
