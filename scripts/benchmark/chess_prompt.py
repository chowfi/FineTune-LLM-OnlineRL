"""Western Chess LLM prompt (parallel shape to xiangqi_prompt).

Same output contract as the Xiangqi prompt:

    <think>...</think>
    Move: <uci>

UCI move format: ``<file><rank><file><rank>`` with files ``a-h`` and ranks
``1-8`` (rank 1 = White's first rank). Promotion suffix optional: ``e7e8q``.

The LLM always plays the uppercase (White) side; we never request black-side
play (any color-balanced sampling would require either prompt rotation or
board flipping, neither of which v2 supports). This costs ~30 Elo of
first-move advantage that applies symmetrically to both games.
"""

from __future__ import annotations

from typing import Dict, List, Optional


CHESS_SYSTEM_PROMPT = (
    "You are a Western Chess player. You always play the uppercase (White) side.\n"
    "Piece letters: K=King Q=Queen R=Rook B=Bishop N=Knight P=Pawn.\n"
    "Coordinates: files a-h (left to right from White's perspective), ranks 1-8\n"
    "(rank 1 at the bottom from White's perspective, rank 8 at the top as shown in the graphic).\n"
    "Uppercase (your) pieces start on ranks 1-2; lowercase enemy pieces start on ranks 7-8.\n"
    "A move is written in UCI: <from_file><from_rank><to_file><to_rank>, e.g. e2e4 moves the\n"
    "pawn on e2 to e4. Pawn promotion is written by appending the promoted piece letter, lowercase:\n"
    "e7e8q (promote to queen), a7a8n (promote to knight), etc. Castling is written as the King's\n"
    "two-square move: kingside e1g1, queenside e1c1.\n\n"
    "STANDARD CHESS MOVEMENT RULES:\n"
    "- K (King): moves 1 step in any direction. Castling: if neither King nor rook has moved and\n"
    "  the squares between them are empty (and not attacked, and the King is not in check), the\n"
    "  King moves two squares toward the rook and the rook jumps over the King.\n"
    "- Q (Queen): slides any number of empty squares orthogonally OR diagonally.\n"
    "- R (Rook): slides any number of empty squares orthogonally.\n"
    "- B (Bishop): slides any number of empty squares diagonally.\n"
    "- N (Knight): L-shape - 2 squares orthogonally + 1 square perpendicular. Jumps over pieces.\n"
    "- P (Pawn): forward 1 step (toward higher rank for White) on an empty square; from rank 2 it\n"
    "  may move forward 2 squares to rank 4 if both are empty. It captures DIAGONALLY one square\n"
    "  forward. On the 8th rank it MUST promote (default to Q if you do not specify a letter).\n"
    "  En-passant: if an enemy pawn just moved two squares from rank 7 to rank 5 and is now next\n"
    "  to your pawn on rank 5, your pawn may capture it by moving diagonally to the empty rank-6\n"
    "  square behind it on the SAME move (one-move-only window).\n\n"
    "IMPORTANT:\n"
    "- Only output legal moves. Verify your move appears in the legal-move list provided.\n"
    "- Do not invent extra moves, comments, or alternative options outside the format below.\n"
    "- Promotion letter is lowercase: q, r, b, n.\n\n"
    "In <think>, briefly state: (1) the impact of the enemy's last move, (2) the piece you will\n"
    "move and the tactical idea, (3) the enemy's most dangerous reply.\n"
    "Then output exactly one legal move on its own line.\n\n"
    "Respond exactly in this format (two lines, nothing else):\n"
    "<think>your tactical reasoning referring to the piece and squares of the move you will play</think>\n"
    "Move: <from><to>[promo]"
)


def format_chess_turn_messages(
    fen: str,
    graphic: str,
    enemy_move_desc: Optional[str],
    legal_moves_hint: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build the OpenAI-style messages list for one Western chess turn.

    ``fen`` is the full chess FEN (incl. side-to-move + castling/en-passant/halfmove/fullmove).
    ``graphic`` is an ASCII board with rank 8 at the top, rank 1 at the bottom.
    ``legal_moves_hint`` is a list of UCI strings (e.g. ``e2e4``, ``g1f3``, ``e7e8q``);
    we trim to the first 48 to keep the prompt short.
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
        "Pick the single best legal move for the uppercase (White) side and output reasoning + move."
    )
    return [
        {"role": "system", "content": CHESS_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
