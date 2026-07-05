"""Board -> tensor planes and internal-algebraic move <-> flat action index."""

from __future__ import annotations

import numpy as np

from src.xiangqi_board import algebraic_to_board_coords, board_coords_to_algebraic

# gym_xiangqi piece ids (abs value) -> type index
# 0 king, 1 advisor, 2 elephant, 3 horse, 4 rook, 5 cannon, 6 pawn
PIECE_TYPE = {
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 5,
    11: 5,
    12: 6,
    13: 6,
    14: 6,
    15: 6,
    16: 6,
}
PIECE_VALUE = {0: 0.0, 1: 2.0, 2: 2.0, 3: 4.0, 4: 9.0, 5: 4.5, 6: 1.0}


def move_to_index(move: str) -> int:
    coords = algebraic_to_board_coords(move)
    if coords is None:
        raise ValueError(f"bad move: {move!r}")
    (fr, fc), (tr, tc) = coords
    return (fr * 9 + fc) * 90 + (tr * 9 + tc)


def index_to_move(idx: int) -> str:
    idx = int(idx)
    if not 0 <= idx < 8100:
        raise ValueError(f"index out of range: {idx}")
    frm, to = divmod(idx, 90)
    return board_coords_to_algebraic(frm // 9, frm % 9, to // 9, to % 9)


def flip_board(board: np.ndarray) -> np.ndarray:
    """Vertical flip + color swap: the same position seen by the other side.

    Rows reverse (r -> 9-r) and signed piece ids negate, so a black-to-move
    position maps into the frame red enjoys (own army nearest row 9)."""
    return np.ascontiguousarray(board[::-1] * -1)


def mirror_board(board: np.ndarray) -> np.ndarray:
    """Left-right mirror (Xiangqi rules are left-right symmetric)."""
    return np.ascontiguousarray(board[:, ::-1])


def _transform_action(idx, row_map, col_map):
    if not (
        isinstance(idx, (int, np.integer))
        or np.issubdtype(np.asarray(idx).dtype, np.integer)
    ):
        raise TypeError(f"action index must be integer, got {np.asarray(idx).dtype}")
    a = np.asarray(idx, dtype=np.int64)
    if np.any(a < 0) or np.any(a >= 8100):
        raise ValueError(f"action index out of range: {idx!r}")
    frm, to = a // 90, a % 90
    fr, fc = frm // 9, frm % 9
    tr, tc = to // 9, to % 9
    out = (row_map(fr) * 9 + col_map(fc)) * 90 + (row_map(tr) * 9 + col_map(tc))
    if isinstance(idx, (int, np.integer)):
        return int(out)
    return out


def flip_action(idx: int | np.ndarray) -> int | np.ndarray:
    """Top-bottom mirror of a flat action index (rows r -> 9-r).

    Matches flip_board; accepts a python int or an int array (vectorized)."""
    return _transform_action(idx, lambda r: 9 - r, lambda c: c)


def mirror_action(idx: int | np.ndarray) -> int | np.ndarray:
    """Left-right mirror of a flat action index (cols c -> 8-c).

    Matches mirror_board; accepts a python int or an int array."""
    return _transform_action(idx, lambda r: r, lambda c: 8 - c)


def board_planes(board: np.ndarray) -> np.ndarray:
    """One position -> 14 binary planes (7 red types, then 7 black)."""
    planes = np.zeros((14, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            v = int(board[r, c])
            if v == 0:
                continue
            t = PIECE_TYPE[abs(v)]
            planes[t if v > 0 else 7 + t, r, c] = 1.0
    return planes


def encode_observation(
    boards: list,
    side_to_move: str,
    repetition_count: int,
    no_progress: int,
    history_length: int = 8,
) -> np.ndarray:
    """Stack of the last `history_length` boards (oldest first, zero-padded)
    plus side-to-move / repetition / no-progress broadcast planes."""
    assert side_to_move in ("w", "b"), side_to_move
    hist = list(boards)[-history_length:]
    stacks = [
        np.zeros((14, 10, 9), dtype=np.float32)
        for _ in range(history_length - len(hist))
    ]
    stacks = stacks + [board_planes(b) for b in hist]
    stm = np.full((1, 10, 9), 1.0 if side_to_move == "w" else 0.0, dtype=np.float32)
    rep = np.full(
        (1, 10, 9), min(max(int(repetition_count), 0), 3) / 3.0, dtype=np.float32
    )
    nop = np.full(
        (1, 10, 9), min(max(int(no_progress), 0), 100) / 100.0, dtype=np.float32
    )
    return np.concatenate(stacks + [stm, rep, nop], axis=0)


def material_balance(board: np.ndarray) -> float:
    """Red minus Black piece value (kings worth 0)."""
    total = 0.0
    for v in board.flatten():
        v = int(v)
        if v == 0:
            continue
        val = PIECE_VALUE[PIECE_TYPE[abs(v)]]
        total += val if v > 0 else -val
    return total
