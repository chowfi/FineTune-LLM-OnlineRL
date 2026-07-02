"""Pikafish-vs-Pikafish self-play generator (fallback when PGN corpus is exhausted).

Emits the same ``Position`` dict shape as :mod:`xiangqi_pgn` so the SFT builder can pull
from either source through a single interface. Each game randomises a short legal-move
opening (so we don't always get the engine's #1 line), then both sides play Pikafish's
best move until checkmate / no-moves / ``--max-plies`` cap. Winner is annotated on every
ply of the finished game so the paper §3.1 winner-side filter still applies.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator

from cchess import RED, ChessBoard, FULL_INIT_FEN  # type: ignore[attr-defined]

from pikafish_eval import PikafishEvaluator


def _maybe_terminal_winner(board: ChessBoard) -> str | None:
    """If ``board`` is terminal (no moves for side-to-move), return winner; else None.

    ``board.move_player`` must already be flipped to the side **to move next**.
    """
    try:
        if board.no_moves():
            return "black" if board.move_player.color == RED else "red"
    except Exception:
        return None
    return None


def _random_opening(
    eng: PikafishEvaluator,
    board: ChessBoard,
    rng: random.Random,
    plies: int,
) -> bool:
    """Play ``plies`` random legal half-moves on ``board``. False if it terminates early."""
    for _ in range(plies):
        legals = eng.list_legal_moves(board.to_full_fen()) or []
        if not legals:
            return False
        mv = rng.choice(legals)
        if board.move_iccs(mv) is None:
            return False
        board.move_player.next()
    return True


def iter_selfplay(
    eng: PikafishEvaluator,
    *,
    n_games: int,
    seed: int = 0,
    random_opening_min: int = 2,
    random_opening_max: int = 6,
    max_plies: int = 200,
) -> Iterator[Dict[str, Any]]:
    """Yield ply-level positions for at most ``n_games`` completed self-play games.

    Games that hit ``max_plies`` without checkmate are emitted with ``winner="draw"``
    (treated like the paper's drawn-game branch — both sides' moves kept).
    """
    rng = random.Random(seed)
    games_played = 0

    while games_played < n_games:
        board = ChessBoard(FULL_INIT_FEN)
        opening_plies = rng.randint(
            max(0, int(random_opening_min)), max(0, int(random_opening_max))
        )
        if opening_plies > 0 and not _random_opening(eng, board, rng, opening_plies):
            continue

        plies = []
        winner: str | None = None
        for ply_idx in range(max_plies):
            fen_before = board.to_full_fen()
            turn = fen_before.split()[1] if " " in fen_before else "w"
            best_uci, _ = eng.bestmove_root_cached(fen_before)
            if not best_uci:
                winner = "black" if turn == "w" else "red"
                break
            applied = board.move_iccs(best_uci)
            if applied is None:
                winner = "unknown"
                break
            plies.append(
                {
                    "fen": fen_before,
                    "played_uci": best_uci,
                    "ply": ply_idx,
                    "turn": turn,
                }
            )
            board.move_player.next()
            terminal = _maybe_terminal_winner(board)
            if terminal is not None:
                winner = terminal
                break

        if winner is None:
            winner = "draw"
        if winner == "unknown" or not plies:
            continue

        games_played += 1
        for p in plies:
            p["winner"] = winner
            p["headers"] = {"Event": f"selfplay-{games_played}"}
            yield p
