"""Streaming PGN → (fen, move, winner) iterator for the Xiangqi SFT builder.

Tailored to the wukong-xiangqi UCI-notation PGN dump (no ``[Result]`` tag, no terminator):
we infer the winner by replaying moves through ``cchess.ChessBoard`` and checking the final
position. Games whose final state isn't checkmate / no-moves are tagged ``"unknown"``; the
SFT builder drops them per the paper's "winner-side + draw moves only" filter (§3.1).

We deliberately do not depend on ``cchess.read_from_pgn`` because it (a) reads the whole
file at once and (b) discards positions; we need per-ply state.

Output: ``Iterator[Position]`` where ``Position`` is::

    {
      "fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
      "played_uci": "h2e2",
      "ply": 0,                       # 0-indexed half-move
      "turn": "w" | "b",              # side that played `played_uci`
      "winner": "red" | "black" | "draw" | "unknown",
      "headers": {"Event": "...", "Red": "...", "Black": "..."},
    }
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Iterator, List

from cchess import RED, ChessBoard, FULL_INIT_FEN  # type: ignore[attr-defined]

HEADER_RE = re.compile(r'^\[(\w+)\s+"([^"]*)"\]\s*$')
RESULT_TOKENS = {"*", "1-0", "0-1", "1/2-1/2"}
# UCCI/ICCS: two letters a–i + two digits 0–9 → letter + digit
UCI_RE = re.compile(r"^[a-i][0-9][a-i][0-9]$")


def _result_to_winner(result: str) -> str:
    """PGN ``[Result ...]`` → ``red`` | ``black`` | ``draw`` | ``unknown``."""
    if result == "1-0":
        return "red"
    if result == "0-1":
        return "black"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def _infer_winner_from_board(board: ChessBoard) -> str:
    """No ``[Result]`` in wukong PGNs: infer from terminal position.

    ``board.move_player`` should already be flipped to the side **to move next**.
    If that side has no legal moves, the previous mover won (Xiangqi stalemate = loss).
    """
    try:
        if board.no_moves():
            return "black" if board.move_player.color == RED else "red"
    except Exception:
        pass
    return "unknown"


def _tokenize_moves(move_lines: Iterable[str]) -> List[str]:
    moves: List[str] = []
    for line in move_lines:
        for tok in line.split():
            t = tok.strip()
            if not t:
                continue
            if t in RESULT_TOKENS:
                continue
            if t.endswith("."):
                continue
            t = t.lower()
            if len(t) == 5 and t[2] in {"-", "="}:
                t = t[:2] + t[3:]
            moves.append(t)
    return moves


def _iter_game_blocks(
    path: str,
) -> Iterator[tuple[Dict[str, str], List[str], str]]:
    """Yield ``(headers, move_lines, raw_result)`` per game block in the file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        headers: Dict[str, str] = {}
        move_lines: List[str] = []
        result = "*"
        in_moves = False
        for raw in f:
            line = raw.rstrip("\n").strip()
            if not line:
                if in_moves and (headers or move_lines):
                    yield headers, move_lines, result
                    headers, move_lines, result, in_moves = {}, [], "*", False
                continue
            m = HEADER_RE.match(line)
            if m:
                if in_moves and (headers or move_lines):
                    yield headers, move_lines, result
                    headers, move_lines, result, in_moves = {}, [], "*", False
                tag, val = m.group(1), m.group(2)
                headers[tag] = val
                if tag == "Result":
                    result = val
                continue
            in_moves = True
            move_lines.append(line)
            for tok in line.split():
                if tok in RESULT_TOKENS:
                    result = tok
        if headers or move_lines:
            yield headers, move_lines, result


def iter_positions(
    path: str,
    *,
    max_games: int | None = None,
    drop_unresolved: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield one dict per legal ply across the PGN file.

    Games with an illegal move (data corruption) are *always* dropped wholesale. Games
    that finish without a ``[Result]`` tag and without a terminal checkmate are emitted
    with ``winner="unknown"`` unless ``drop_unresolved`` is set; the SFT builder treats
    those as draw-equivalent so we keep both sides' moves (per the paper's drawn-game
    branch), which is the safest fallback when the corpus has no result metadata.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    games_emitted = 0
    for headers, move_lines, raw_result in _iter_game_blocks(path):
        if max_games is not None and games_emitted >= max_games:
            return
        tokens = _tokenize_moves(move_lines)
        if not tokens:
            continue

        board = ChessBoard(FULL_INIT_FEN)
        plies: List[Dict[str, Any]] = []
        aborted = False
        for ply_idx, mv in enumerate(tokens):
            if not UCI_RE.match(mv):
                aborted = True
                break
            fen_before = board.to_full_fen()
            turn = fen_before.split()[1] if " " in fen_before else "w"
            applied = board.move_iccs(mv)
            if applied is None:
                aborted = True
                break
            plies.append(
                {
                    "fen": fen_before,
                    "played_uci": mv,
                    "ply": ply_idx,
                    "turn": turn,
                }
            )
            board.move_player.next()

        if aborted or not plies:
            # Data error in the PGN — drop the whole game.
            continue

        if raw_result in {"1-0", "0-1", "1/2-1/2"}:
            winner = _result_to_winner(raw_result)
        else:
            winner = _infer_winner_from_board(board)

        if drop_unresolved and winner == "unknown":
            continue

        games_emitted += 1
        for p in plies:
            p["winner"] = winner
            p["headers"] = headers
            yield p
