"""Play full matches between an LLM and a UCI engine (or two engines).

The match runner does not care which game it is - it asks the board adapter
for FEN/graphic/legal-moves and asks the engine + LLM for moves. JSONL logs
are streamed per game under ``data/benchmark/games/``.

LLM always plays the uppercase side (White / Red). That ~30 Elo first-mover
bias applies symmetrically across chess and xiangqi (limitation documented
in the experiment log).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from .boards import XiangqiBoard, make_board
from .chess_prompt import format_chess_turn_messages
from .engines import UciEngine
from .llm_player import LLMPlayer, LLMMoveResult
from .xiangqi_prompt import format_xiangqi_turn_messages


@dataclass
class GameResult:
    game: str  # "chess" or "xiangqi"
    rung_ms: int
    n_plies: int
    outcome: str  # "white_wins" / "black_wins" / "draw"
    llm_won: bool
    llm_drew: bool
    format_fail_count: int
    parse_ok_count: int
    legal_ok_count: int
    llm_move_count: int
    engine_move_count: int
    llm_wall_sec: float
    engine_wall_sec: float
    log_path: str


def _format_messages_for_game(
    game: str,
    fen: str,
    graphic: str,
    enemy_move_desc: Optional[str],
    legal_moves_hint: List[str],
) -> List[Dict[str, str]]:
    if game == "chess":
        return format_chess_turn_messages(
            fen=fen,
            graphic=graphic,
            enemy_move_desc=enemy_move_desc,
            legal_moves_hint=legal_moves_hint,
        )
    return format_xiangqi_turn_messages(
        fen=fen,
        graphic=graphic,
        enemy_move_desc=enemy_move_desc,
        legal_moves_hint=legal_moves_hint,
    )


def play_llm_vs_engine(
    *,
    game: str,
    llm: LLMPlayer,
    engine: UciEngine,
    rung_ms: int,
    out_dir: str,
    game_index: int = 0,
    max_plies: int = 300,
) -> GameResult:
    """One game: LLM plays uppercase, engine plays lowercase.

    Engine plays after every LLM move (LLM moves first because it's the
    uppercase side; uppercase is the side to move in the standard starting
    position for both games).
    """
    board = make_board(game)
    if isinstance(board, XiangqiBoard):
        board._max_plies = max_plies  # noqa: SLF001  (test-only knob)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(
        out_dir, f"{game}_rung{rung_ms}ms_game{game_index:04d}.jsonl"
    )
    log_fp = open(log_path, "w", encoding="utf-8")

    last_engine_move_desc: Optional[str] = None
    format_fail_count = 0
    parse_ok_count = 0
    legal_ok_count = 0
    llm_move_count = 0
    engine_move_count = 0
    llm_wall_sec = 0.0
    engine_wall_sec = 0.0

    try:
        while True:
            done, outcome = board.is_terminal()
            if done:
                break
            if board.ply() >= max_plies:
                break

            side = board.side_to_move()
            fen = board.fen()
            graphic = board.graphic()

            if side == "w":
                # LLM turn.
                llm_legals = board.legal_moves_llm()
                msgs = _format_messages_for_game(
                    game=game,
                    fen=board.llm_fen(),
                    graphic=graphic,
                    enemy_move_desc=last_engine_move_desc,
                    legal_moves_hint=llm_legals,
                )
                result: LLMMoveResult = llm.generate_move(msgs, llm_legals)
                llm_wall_sec += result.wall_sec
                llm_move_count += 1
                parse_ok_count += int(result.parse_ok)
                legal_ok_count += int(result.legal_ok)
                format_fail_count += int(result.format_fail)

                applied = board.apply_llm_move(result.move) if result.move else False
                engine_uci = board.llm_to_engine(result.move) if applied else None
                _write_jsonl(
                    log_fp,
                    {
                        "ply": board.ply() - (1 if applied else 0),
                        "actor": "llm",
                        "fen_before": fen,
                        "llm_move": result.move,
                        "engine_uci": engine_uci,
                        "applied": applied,
                        "parse_ok": result.parse_ok,
                        "legal_ok": result.legal_ok,
                        "format_fail": result.format_fail,
                        "wall_sec": result.wall_sec,
                        "gen_tokens": result.gen_tokens,
                        "raw_text": result.raw_text,
                    },
                )
                if not applied:
                    # Random legal fallback already happened inside
                    # generate_move; if even that fails (no legals), we end.
                    break
                last_engine_move_desc = None
            else:
                # Engine turn.
                t0 = time.perf_counter()
                engine_uci = engine.bestmove(fen)
                engine_wall_sec += time.perf_counter() - t0
                if not engine_uci:
                    # Engine couldn't find a move - treat as terminal.
                    _write_jsonl(
                        log_fp,
                        {
                            "ply": board.ply(),
                            "actor": "engine",
                            "fen_before": fen,
                            "engine_uci": None,
                            "applied": False,
                            "note": "engine returned no bestmove",
                        },
                    )
                    break
                applied = board.apply_engine_move(engine_uci)
                llm_view = board.engine_to_llm(engine_uci) if applied else None
                _write_jsonl(
                    log_fp,
                    {
                        "ply": board.ply() - (1 if applied else 0),
                        "actor": "engine",
                        "fen_before": fen,
                        "engine_uci": engine_uci,
                        "llm_view": llm_view,
                        "applied": applied,
                    },
                )
                engine_move_count += 1
                if not applied:
                    break
                last_engine_move_desc = llm_view or engine_uci

        # Final terminal check + outcome.
        done, outcome = board.is_terminal()
        if not done and board.ply() >= max_plies:
            outcome = "draw"
        if not outcome:
            outcome = "draw"  # safety: treat aborted games as draw
        llm_won = outcome == "white_wins"
        llm_drew = outcome == "draw"
        return GameResult(
            game=game,
            rung_ms=int(rung_ms),
            n_plies=int(board.ply()),
            outcome=outcome,
            llm_won=llm_won,
            llm_drew=llm_drew,
            format_fail_count=format_fail_count,
            parse_ok_count=parse_ok_count,
            legal_ok_count=legal_ok_count,
            llm_move_count=llm_move_count,
            engine_move_count=engine_move_count,
            llm_wall_sec=llm_wall_sec,
            engine_wall_sec=engine_wall_sec,
            log_path=log_path,
        )
    finally:
        log_fp.close()


def play_engine_vs_engine(
    *,
    game: str,
    white_engine: UciEngine,
    black_engine: UciEngine,
    out_dir: str,
    game_index: int = 0,
    max_plies: int = 300,
) -> Dict[str, Any]:
    """One engine-vs-engine game (used for ladder calibration).

    Returns ``{game, n_plies, outcome, white_ms, black_ms, wall_sec, log_path}``.
    """
    board = make_board(game)
    if isinstance(board, XiangqiBoard):
        board._max_plies = max_plies  # noqa: SLF001

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(
        out_dir,
        f"{game}_cal_{white_engine.movetime_ms}vs{black_engine.movetime_ms}_g{game_index:04d}.jsonl",
    )
    log_fp = open(log_path, "w", encoding="utf-8")
    t_start = time.perf_counter()
    try:
        while True:
            done, _ = board.is_terminal()
            if done or board.ply() >= max_plies:
                break
            eng = white_engine if board.side_to_move() == "w" else black_engine
            fen = board.fen()
            uci = eng.bestmove(fen)
            if not uci:
                break
            ok = board.apply_engine_move(uci)
            _write_jsonl(
                log_fp,
                {
                    "ply": board.ply() - (1 if ok else 0),
                    "side": "w" if eng is white_engine else "b",
                    "engine_ms": eng.movetime_ms,
                    "fen_before": fen,
                    "engine_uci": uci,
                    "applied": ok,
                },
            )
            if not ok:
                break
        done, outcome = board.is_terminal()
        if not done and board.ply() >= max_plies:
            outcome = "draw"
        if not outcome:
            outcome = "draw"
        return {
            "game": game,
            "n_plies": int(board.ply()),
            "outcome": outcome,
            "white_ms": white_engine.movetime_ms,
            "black_ms": black_engine.movetime_ms,
            "wall_sec": time.perf_counter() - t_start,
            "log_path": log_path,
        }
    finally:
        log_fp.close()


def _write_jsonl(fp, row: Dict[str, Any]) -> None:
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    fp.flush()


def play_match(
    *,
    game: str,
    llm: LLMPlayer,
    engine: UciEngine,
    n_games: int,
    out_dir: str,
    rung_ms: int,
    max_plies: int = 300,
    progress_callback: Optional[Callable[[int, int, GameResult], None]] = None,
) -> List[GameResult]:
    """Play ``n_games`` LLM-vs-engine. Returns list of GameResults."""
    results: List[GameResult] = []
    games_dir = os.path.join(out_dir, "games")
    for i in range(int(n_games)):
        engine.newgame()
        result = play_llm_vs_engine(
            game=game,
            llm=llm,
            engine=engine,
            rung_ms=rung_ms,
            out_dir=games_dir,
            game_index=i,
            max_plies=max_plies,
        )
        results.append(result)
        if progress_callback:
            progress_callback(i, n_games, result)
    return results


def summarize_results(results: List[GameResult]) -> Dict[str, Any]:
    if not results:
        return {"n": 0}
    wins = sum(int(r.llm_won) for r in results)
    draws = sum(int(r.llm_drew) for r in results)
    losses = len(results) - wins - draws
    fmt_total = sum(r.format_fail_count for r in results)
    move_total = sum(r.llm_move_count for r in results)
    parse_total = sum(r.parse_ok_count for r in results)
    legal_total = sum(r.legal_ok_count for r in results)
    return {
        "n": len(results),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / max(1, len(results)),
        "format_fail_rate": fmt_total / max(1, move_total),
        "parse_ok_rate": parse_total / max(1, move_total),
        "legal_ok_rate": legal_total / max(1, move_total),
        "total_llm_moves": move_total,
        "total_llm_wall_sec": sum(r.llm_wall_sec for r in results),
        "total_engine_wall_sec": sum(r.engine_wall_sec for r in results),
        "avg_plies": sum(r.n_plies for r in results) / len(results),
    }


def gameresult_to_dict(r: GameResult) -> Dict[str, Any]:
    return asdict(r)


__all__ = [
    "GameResult",
    "play_llm_vs_engine",
    "play_engine_vs_engine",
    "play_match",
    "summarize_results",
    "gameresult_to_dict",
]
