"""Board -> tensor planes and internal-algebraic move <-> flat action index."""

from __future__ import annotations


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
    frm, to = divmod(int(idx), 90)
    return board_coords_to_algebraic(frm // 9, frm % 9, to // 9, to % 9)
