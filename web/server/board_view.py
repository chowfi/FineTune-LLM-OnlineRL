"""Board-grid rendering shared by LLM and MuZero sessions (no heavy imports)."""

from __future__ import annotations

from typing import List

import numpy as np

# gym_xiangqi piece ids (abs value) -> FEN letter:
# 1=king, 2/3=advisor, 4/5=elephant, 6/7=horse, 8/9=rook, 10/11=cannon, 12-16=pawn
PIECE_FEN = {
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


def board_grid(state: np.ndarray) -> List[List[str]]:
    """Signed int8 board -> UI letter grid (upper=red, lower=black, '.'=empty)."""
    rows: List[List[str]] = []
    for row in state:
        cells: List[str] = []
        for cell in row:
            val = int(cell)
            if val == 0:
                cells.append(".")
            else:
                base = PIECE_FEN.get(abs(val), "?")
                cells.append(base.upper() if val > 0 else base)
        rows.append(cells)
    return rows
