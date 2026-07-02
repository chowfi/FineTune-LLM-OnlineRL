"""Pikafish-aligned situation labels (Xiangqi-R1 paper §3.1). Red-positive centipawns."""

from __future__ import annotations

import re
from typing import Callable, Optional, Tuple

# Paper defaults (arXiv:2507.12215v1)
SIGMA_S = 100
SIGMA_L = 800
SIGMA_GOOD = 100


def parse_situation_from_response(text: str) -> Optional[str]:
    """Parse ``Situation: ...`` line for GRPO ``R_analysis`` (3-class string)."""
    m = re.search(r"Situation:\s*([^\n<]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    rl = m.group(1).strip().lower()
    if "balanced" in rl or rl == "equal":
        return "Balanced"
    if "black" in rl and (
        "adv" in rl or "advantage" in rl or "slight" in rl or "clear" in rl
    ):
        return "Advantage_Black"
    if "red" in rl and (
        "adv" in rl or "advantage" in rl or "slight" in rl or "clear" in rl
    ):
        return "Advantage_Red"
    if "advantage_black" in rl.replace(" ", "_"):
        return "Advantage_Black"
    if "advantage_red" in rl.replace(" ", "_"):
        return "Advantage_Red"
    return None


def root_value_red_oriented(
    fen: str, cp_side_to_move: Optional[float]
) -> Optional[float]:
    """Convert engine root score to *Red-positive* centipawns (paper ``Value``).

    ``cp_side_to_move`` is Pikafish's score for the player to move in *fen*.
    """
    if cp_side_to_move is None:
        return None
    stm = fen.split()[1] if len(fen.split()) > 1 else "w"
    if stm == "w":
        return float(cp_side_to_move)
    return -float(cp_side_to_move)


def situation_5class(value_red: float) -> str:
    """Five paper classes: slight/clear adv for Red or Black, or balanced."""
    v = float(value_red)
    av = abs(v)
    if av <= SIGMA_S:
        return "Balanced"
    if v > SIGMA_S:
        if av < SIGMA_L:
            return "Slight_Adv_Red"
        return "Clear_Adv_Red"
    if av < SIGMA_L:
        return "Slight_Adv_Black"
    return "Clear_Adv_Black"


def situation_3class(value_red: float) -> str:
    """Merged slight+clear into a single advantage per side."""
    v = float(value_red)
    if abs(v) <= SIGMA_S:
        return "Balanced"
    if v > SIGMA_S:
        return "Advantage_Red"
    return "Advantage_Black"


def red_value_after_uci_move(
    fen_before: str, uci_move: str, evaluate_cp_fn
) -> Optional[float]:
    """Red-oriented value of the position *after* ``uci_move`` from ``fen_before``."""
    cp = evaluate_cp_fn(fen_before, [uci_move])
    if cp is None:
        return None
    return -float(cp)


def is_good_move(
    fen_before: str,
    played_uci: str,
    best_uci: str,
    evaluate_cp_fn: Callable[..., Optional[float]],
    sigma_good: float = SIGMA_GOOD,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """|V_red(after played) - V_red(after best)| <= sigma_good (paper Eq. 3 style)."""
    vp = red_value_after_uci_move(fen_before, played_uci, evaluate_cp_fn)
    vb = red_value_after_uci_move(fen_before, best_uci, evaluate_cp_fn)
    if vp is None or vb is None:
        return False, vp, vb
    return abs(vp - vb) <= sigma_good, vp, vb
