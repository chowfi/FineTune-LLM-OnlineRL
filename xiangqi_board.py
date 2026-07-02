"""Xiangqi board ↔ FEN / algebraic helpers (no torch). Shared by RL training and SFT scripts."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

COLS = "abcdefghi"
COL_TO_IDX = {c: i for i, c in enumerate(COLS)}

_PIECE_TO_FEN = {
    1: "k",
    2: "a",
    3: "a",
    4: "b",
    5: "b",
    6: "n",
    7: "n",
    8: "r",
    9: "r",
    10: "c",
    11: "c",
    12: "p",
    13: "p",
    14: "p",
    15: "p",
    16: "p",
}

ALGEBRAIC_RE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])$")


def board_to_fen(state: np.ndarray) -> str:
    fen_rows: List[str] = []
    for row in state:
        empties = 0
        tokens: List[str] = []
        for cell in row:
            val = int(cell)
            if val == 0:
                empties += 1
                continue
            if empties > 0:
                tokens.append(str(empties))
                empties = 0
            base = _PIECE_TO_FEN.get(abs(val), "?")
            tokens.append(base.upper() if val > 0 else base)
        if empties > 0:
            tokens.append(str(empties))
        fen_rows.append("".join(tokens))
    return "/".join(fen_rows)


def board_to_uci_fen(state: np.ndarray, side_to_move: str = "w") -> str:
    stm = side_to_move if side_to_move in {"w", "b"} else "w"
    return f"{board_to_fen(state)} {stm} - - 0 1"


def board_to_graphic(state: np.ndarray) -> str:
    lines = ["  " + " ".join(COLS)]
    for row_idx in range(state.shape[0]):
        row_tokens: List[str] = []
        for col_idx in range(state.shape[1]):
            val = int(state[row_idx][col_idx])
            if val == 0:
                row_tokens.append(".")
                continue
            base = _PIECE_TO_FEN.get(abs(val), "?")
            row_tokens.append(base.upper() if val > 0 else base.lower())
        lines.append(f"{row_idx} " + " ".join(row_tokens))
        if row_idx == 4:
            lines.append("  ~~~~~~~~~~~~~~~~~")
    return "\n".join(lines)


def board_coords_to_algebraic(
    from_row: int, from_col: int, to_row: int, to_col: int
) -> str:
    return f"{COLS[from_col]}{from_row}{COLS[to_col]}{to_row}"


def algebraic_to_board_coords(
    move_str: str,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if not move_str:
        return None
    match = ALGEBRAIC_RE.match(move_str.strip().lower())
    if not match:
        return None
    from_col = COL_TO_IDX[match.group(1)]
    from_row = int(match.group(2))
    to_col = COL_TO_IDX[match.group(3)]
    to_row = int(match.group(4))
    return (from_row, from_col), (to_row, to_col)


def algebraic_to_engine_move(move_str: str) -> Optional[str]:
    """Internal board algebraic (rank 0 top) → Pikafish UCI (rank 0 bottom)."""
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return None
    (from_row, from_col), (to_row, to_col) = parsed
    return f"{COLS[from_col]}{9 - from_row}{COLS[to_col]}{9 - to_row}"


def engine_uci_to_algebraic(uci: str) -> Optional[str]:
    """Pikafish UCI (bottom-origin ranks) → internal algebraic ``a0a1``."""
    m = re.match(r"^([a-i])([0-9])([a-i])([0-9])$", uci.strip().lower())
    if not m:
        return None
    fc, fr_s, tc, tr_s = m.groups()
    from_row = 9 - int(fr_s)
    to_row = 9 - int(tr_s)
    from_col = COL_TO_IDX[fc]
    to_col = COL_TO_IDX[tc]
    return board_coords_to_algebraic(from_row, from_col, to_row, to_col)
