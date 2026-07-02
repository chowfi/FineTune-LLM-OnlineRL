"""Unified game adapter for Western chess (python-chess) + Xiangqi (cchess).

Both adapters expose the same interface to the match runner / LLM player so
``run_match.py`` doesn't have to special-case the game type:

* ``fen()`` - full FEN consumed by the engine (UCI).
* ``llm_fen()`` - piece-placement FEN shown to the LLM (xiangqi only renders
  the placement; chess keeps the full FEN for castling/en-passant info).
* ``graphic()`` - ASCII board for the LLM prompt, oriented White/Red at the
  bottom in the LLM's frame.
* ``legal_moves_llm()`` - move strings in the LLM's expected dialect
  (top-origin algebraic for xiangqi, standard UCI for chess).
* ``legal_moves_engine()`` - move strings in the engine's UCI dialect.
* ``llm_to_engine(move) -> engine_uci`` and ``engine_to_llm(move) -> llm_uci``.
* ``apply_llm_move(move)`` / ``apply_engine_move(move)``: both accept their
  respective dialect, advance the position. Return True iff legal.
* ``is_terminal() -> (done, outcome)`` where outcome is one of
  ``"white_wins"`` / ``"black_wins"`` / ``"draw"`` / ``""`` (only meaningful
  if ``done``).
* ``side_to_move() -> "w" | "b"``.
* ``ply()`` - 0-based half-move counter.

Coordinate conventions:

* Western chess: standard UCI everywhere (rank 1 at White's home).
* Xiangqi: LLM speaks **top-origin** algebraic (rank 0 at top, matching v2's
  prompt), engine speaks **bottom-origin** Pikafish UCI (rank 0 at bottom).
  Translation is via :mod:`xiangqi_board` helpers.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from xiangqi_board import (
    algebraic_to_engine_move,
    engine_uci_to_algebraic,
)


_FILES = "abcdefghi"
_XIANGQI_PIECE_GLOSSARY = {
    "k": "K",
    "a": "A",
    "b": "B",
    "n": "N",
    "r": "R",
    "c": "C",
    "p": "P",
}


def _xq_placement_to_graphic(placement: str) -> str:
    """Render a Xiangqi FEN piece-placement in v2's top-origin format.

    Output matches ``xiangqi_board.board_to_graphic``: row 0 at top, river
    drawn after row 4, uppercase pieces shown as uppercase letters.
    """
    rows_fen = placement.split("/")
    if len(rows_fen) != 10:
        raise ValueError(f"Xiangqi FEN must have 10 ranks, got {len(rows_fen)}")
    lines = ["  " + " ".join(_FILES)]
    for row_idx, rank_str in enumerate(rows_fen):
        row_tokens: List[str] = []
        for ch in rank_str:
            if ch.isdigit():
                row_tokens.extend(["."] * int(ch))
            else:
                row_tokens.append(ch)  # FEN preserves uppercase/lowercase
        if len(row_tokens) != 9:
            raise ValueError(f"Xiangqi rank {row_idx} has {len(row_tokens)} cells")
        lines.append(f"{row_idx} " + " ".join(row_tokens))
        if row_idx == 4:
            lines.append("  ~~~~~~~~~~~~~~~~~")
    return "\n".join(lines)


def _move_tuple_to_engine_uci(move: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
    (fx, fy), (tx, ty) = move
    return f"{_FILES[fx]}{fy}{_FILES[tx]}{ty}"


# -----------------------------------------------------------------------------
# Xiangqi adapter (cchess-backed)
# -----------------------------------------------------------------------------


class XiangqiBoard:
    """Xiangqi game adapter wrapping :mod:`cchess`. Engine UCI is bottom-origin
    (Pikafish convention); LLM UCI is top-origin (v2 prompt convention)."""

    def __init__(self, *, max_plies: int = 300):
        from cchess import RED, ChessBoard, FULL_INIT_FEN  # type: ignore

        self._ChessBoard = ChessBoard
        self._RED = RED
        self._FULL_INIT_FEN = FULL_INIT_FEN
        self._board = ChessBoard(FULL_INIT_FEN)
        self._ply = 0
        self._max_plies = int(max_plies)
        self._game_over = False
        self._outcome = ""

    def reset(self) -> None:
        self._board = self._ChessBoard(self._FULL_INIT_FEN)
        self._ply = 0
        self._game_over = False
        self._outcome = ""

    # ------------------ Position / FEN helpers ------------------

    def fen(self) -> str:
        """Engine-consumable full FEN (Pikafish-compatible)."""
        return self._board.to_full_fen()

    def llm_fen(self) -> str:
        """Piece-placement only - what we show the LLM (matches v2)."""
        return self._board.to_full_fen().split()[0]

    def graphic(self) -> str:
        return _xq_placement_to_graphic(self.llm_fen())

    def side_to_move(self) -> str:
        full = self._board.to_full_fen()
        return full.split()[1] if " " in full else "w"

    def ply(self) -> int:
        return self._ply

    # ------------------ Legal moves ------------------

    def legal_moves_engine(self) -> List[str]:
        if self._game_over:
            return []
        return [_move_tuple_to_engine_uci(m) for m in self._board.create_moves()]

    def legal_moves_llm(self) -> List[str]:
        engine_moves = self.legal_moves_engine()
        out: List[str] = []
        for eng_uci in engine_moves:
            llm_uci = engine_uci_to_algebraic(eng_uci)
            if llm_uci is not None:
                out.append(llm_uci)
        return out

    # ------------------ Move translation ------------------

    @staticmethod
    def llm_to_engine(move: str) -> Optional[str]:
        return algebraic_to_engine_move((move or "").strip().lower())

    @staticmethod
    def engine_to_llm(move: str) -> Optional[str]:
        return engine_uci_to_algebraic((move or "").strip().lower())

    # ------------------ Apply moves ------------------

    def _apply_engine_uci(self, engine_uci: str) -> bool:
        applied = self._board.move_iccs(engine_uci)
        if applied is None:
            return False
        self._board.move_player.next()
        self._ply += 1
        self._check_terminal()
        return True

    def apply_engine_move(self, move: str) -> bool:
        return self._apply_engine_uci((move or "").strip().lower())

    def apply_llm_move(self, move: str) -> bool:
        engine_uci = self.llm_to_engine(move)
        if engine_uci is None:
            return False
        return self._apply_engine_uci(engine_uci)

    # ------------------ Terminal detection ------------------

    def _check_terminal(self) -> None:
        if self._game_over:
            return
        try:
            if self._board.no_moves():
                # In Xiangqi, no-moves (mate OR stalemate) is a loss for the
                # side to move. Pikafish + ICCS rules treat stalemate as loss.
                self._game_over = True
                if self._board.move_player.color == self._RED:
                    self._outcome = "black_wins"
                else:
                    self._outcome = "white_wins"
                return
        except Exception:
            pass
        if self._ply >= self._max_plies:
            self._game_over = True
            self._outcome = "draw"

    def is_terminal(self) -> Tuple[bool, str]:
        return self._game_over, self._outcome


# -----------------------------------------------------------------------------
# Western chess adapter (python-chess-backed)
# -----------------------------------------------------------------------------


class ChessBoard:
    """Western chess adapter wrapping :mod:`chess` (python-chess).

    LLM and engine share UCI. Termination follows python-chess's
    ``board.outcome(claim_draw=True)`` so we get 50-move + threefold without
    extra plumbing.
    """

    def __init__(self):
        try:
            import chess  # noqa: F401  (verify dep is installed)
        except ImportError as err:
            raise ImportError(
                "scripts.benchmark.boards.ChessBoard requires `python-chess`. "
                "After the SFT build finishes, run `uv sync` to install it."
            ) from err
        import chess

        self._chess = chess
        self._board = chess.Board()
        self._game_over = False
        self._outcome_str = ""

    def reset(self) -> None:
        self._board = self._chess.Board()
        self._game_over = False
        self._outcome_str = ""

    # ------------------ Position / FEN helpers ------------------

    def fen(self) -> str:
        return self._board.fen()

    def llm_fen(self) -> str:
        return self._board.fen()

    def graphic(self) -> str:
        # python-chess's default str() prints rank 8 at the top; we expand to
        # add file/rank labels for the LLM (parallel to v2's xiangqi graphic).
        lines = ["  " + " ".join("abcdefgh")]
        rows = str(self._board).split("\n")
        for i, line in enumerate(rows):
            rank_num = 8 - i
            cells = line.split()
            lines.append(f"{rank_num} " + " ".join(cells))
        return "\n".join(lines)

    def side_to_move(self) -> str:
        return "w" if self._board.turn == self._chess.WHITE else "b"

    def ply(self) -> int:
        return self._board.ply()

    # ------------------ Legal moves ------------------

    def legal_moves_engine(self) -> List[str]:
        if self._game_over:
            return []
        return [m.uci() for m in self._board.legal_moves]

    def legal_moves_llm(self) -> List[str]:
        return self.legal_moves_engine()

    # ------------------ Move translation (identity in chess) ------------------

    @staticmethod
    def llm_to_engine(move: str) -> Optional[str]:
        m = (move or "").strip().lower()
        return m or None

    @staticmethod
    def engine_to_llm(move: str) -> Optional[str]:
        m = (move or "").strip().lower()
        return m or None

    # ------------------ Apply moves ------------------

    def _apply_uci(self, uci: str) -> bool:
        try:
            mv = self._chess.Move.from_uci(uci)
        except Exception:
            return False
        if mv not in self._board.legal_moves:
            return False
        self._board.push(mv)
        self._check_terminal()
        return True

    def apply_engine_move(self, move: str) -> bool:
        return self._apply_uci((move or "").strip().lower())

    def apply_llm_move(self, move: str) -> bool:
        return self._apply_uci((move or "").strip().lower())

    # ------------------ Terminal detection ------------------

    def _check_terminal(self) -> None:
        if self._game_over:
            return
        outcome = self._board.outcome(claim_draw=True)
        if outcome is None:
            return
        self._game_over = True
        if outcome.winner is None:
            self._outcome_str = "draw"
        elif outcome.winner == self._chess.WHITE:
            self._outcome_str = "white_wins"
        else:
            self._outcome_str = "black_wins"

    def is_terminal(self) -> Tuple[bool, str]:
        return self._game_over, self._outcome_str


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def make_board(game: str):
    g = (game or "").strip().lower()
    if g == "chess":
        return ChessBoard()
    if g in {"xiangqi", "xq"}:
        return XiangqiBoard()
    raise ValueError(f"Unknown game: {game!r} (expected 'chess' or 'xiangqi')")
