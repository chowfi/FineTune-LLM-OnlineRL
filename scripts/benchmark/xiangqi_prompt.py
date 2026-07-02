"""Xiangqi LLM prompt (lifted verbatim from LLM_RL_agent_FSDP_v2.py).

The system prompt and format_turn_prompt logic are copied from
``XiangqiAgent.system_prompt`` (L1844) and ``XiangqiAgent.format_turn_prompt``
(L1885) in ``LLM_RL_agent_FSDP_v2.py``, with two small adaptations:

* Inputs are a FEN string + a pre-rendered graphic string (instead of the gym
  numpy board), so the benchmark can drive a ``cchess`` board directly.
* No legal-move sampler hint logic; legal-moves are passed in as a list of
  internal top-origin algebraic strings (e.g. ``b7b4``).

Coordinate convention (matches v2): rank 0 at top, uppercase pieces start on
ranks 7-9 (bottom). Pikafish UCI uses bottom-origin; conversion to/from
Pikafish UCI is done outside this module (see :mod:`xiangqi_board`).
"""

from __future__ import annotations

from typing import Dict, List, Optional


XIANGQI_SYSTEM_PROMPT = (
    "You are a Xiangqi (Chinese Chess) player. You always play the uppercase side.\n"
    "Piece letters: K=General A=Advisor B=Elephant N=Horse R=Chariot C=Cannon P=Soldier.\n"
    "Coordinates: files a-i (left to right), ranks 0-9 (top to bottom as shown in the graphic).\n"
    "Uppercase (your) pieces start on the BOTTOM (ranks 7-9); lowercase enemy pieces start on the TOP\n"
    "(ranks 0-2). The river sits between ranks 4 and 5 (shown as '~~~' in the graphic).\n"
    "A move is written <from_file><from_rank><to_file><to_rank>, e.g. b7b4 moves the piece on b7 to b4.\n\n"
    "XIANGQI MOVEMENT RULES (these are NOT the same as Western chess):\n"
    "- K (General): moves 1 step orthogonally, must stay in the palace (files d-f, ranks 7-9 for you).\n"
    "  The two Generals may NEVER face each other on the same file with nothing between them\n"
    "  (flying-general rule), so never expose your K on an open file facing the enemy k.\n"
    "- A (Advisor): moves exactly 1 step diagonally, must stay in the palace (d-f, ranks 7-9).\n"
    "- B (Elephant): moves exactly 2 steps diagonally (e.g. c7 to a5 or e5). Cannot cross the river\n"
    "  (your B must stay on ranks 5-9). Blocked if the 1-step diagonal midpoint is occupied\n"
    "  (the 'elephant eye').\n"
    "- N (Horse): moves 1 step orthogonal + 1 step diagonal outward (L-shape, 8 possible targets).\n"
    "  BLOCKED if the orthogonal-adjacent square ('horse leg') is occupied. A horse does NOT\n"
    "  jump like a Western knight.\n"
    "- R (Chariot): slides any number of empty squares orthogonally, exactly like a Western rook.\n"
    "- C (Cannon): moves like R when NOT capturing, but to CAPTURE it must jump over exactly one\n"
    "  piece (of either side) between itself and the target ('screen'). Non-capture cannon moves\n"
    "  must be along empty squares with no piece in between.\n"
    "- P (Soldier): before crossing the river (your P on ranks 5-9) it moves 1 step forward only\n"
    "  (forward = decreasing rank number for you). After crossing the river (ranks 0-4) it may\n"
    "  also move 1 step sideways. A soldier NEVER moves backward and NEVER moves diagonally.\n\n"
    "IMPORTANT - DO NOT use Western chess concepts that do not apply here. In particular:\n"
    "- There is NO queen, NO bishop-pair, NO pawn promotion, NO castling, NO en-passant.\n"
    "- Do NOT talk about bishops (B here is an Elephant with very different movement), knights in\n"
    "  the Western sense (N is a Horse that can be leg-blocked), or rooks beyond the R=Chariot slide.\n"
    "- Only reason about the board using the Xiangqi rules above and the legal-move list provided.\n"
    "- Before committing to a move, mentally verify it matches the movement rule for that piece and\n"
    "  appears in the Legal moves list.\n\n"
    "In <think>, briefly state: (1) the impact of the enemy's last move, (2) the piece you will move\n"
    "and why (citing the Xiangqi rule it uses), (3) the enemy's most dangerous reply.\n"
    "Then output exactly one legal move on its own line.\n\n"
    "Respond exactly in this format (two lines, nothing else):\n"
    "<think>your tactical reasoning referring to the piece and squares of the move you will play</think>\n"
    "Move: <from><to>"
)


def format_xiangqi_turn_messages(
    fen: str,
    graphic: str,
    enemy_move_desc: Optional[str],
    legal_moves_hint: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build the OpenAI-style messages list for one Xiangqi turn.

    ``fen`` is the gym-xiangqi-style piece-placement FEN (rank 0 at top, no UCCI
    fields); ``graphic`` is the rendered ASCII board with the same orientation.
    ``legal_moves_hint`` is a list of top-origin algebraic moves (e.g. ``b7b4``)
    used by the trainer in v2. We trim to the first 48 to keep the prompt short.
    """
    prefix = (
        f"Enemy previous move: {enemy_move_desc}\n"
        if enemy_move_desc
        else "Enemy previous move: none\n"
    )
    hint_line = ""
    if legal_moves_hint:
        trimmed = legal_moves_hint[:48]
        more = (
            ""
            if len(legal_moves_hint) <= 48
            else f" (+{len(legal_moves_hint) - 48} more)"
        )
        hint_line = f"Legal moves (subset): {' '.join(trimmed)}{more}\n"
    user_msg = (
        f"{prefix}"
        f"Current board FEN: {fen}\n"
        f"Current board graphic:\n{graphic}\n"
        f"{hint_line}"
        "Pick the single best legal move for the uppercase side and output reasoning + move."
    )
    return [
        {"role": "system", "content": XIANGQI_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
