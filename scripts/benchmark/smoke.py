"""Minimal end-to-end smoke test for the benchmark plumbing.

Runs ONE LLM-vs-engine game per game at a single fast rung (50 ms by
default). Verifies that:

* engine binaries resolve (Pikafish via ``PIKAFISH_BIN``, Stockfish via
  ``STOCKFISH_BIN`` or ``stockfish`` on PATH),
* the LLM produces parseable UCI moves under the chess / xiangqi prompts,
* the board adapters round-trip moves between LLM and engine dialects,
* JSONL per-game logs are written to ``data/benchmark/games/smoke_*``.

By default we skip games whose engine binary is missing, so a quick
chess-only or xiangqi-only run is fine when only one engine is installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.benchmark.engines import UciEngine, _resolve_binary
from scripts.benchmark.llm_player import LLMPlayer
from scripts.benchmark.run_match import play_llm_vs_engine


def _has_chess() -> bool:
    return _resolve_binary("STOCKFISH_BIN", "stockfish") is not None


def _has_xiangqi() -> bool:
    return _resolve_binary("PIKAFISH_BIN", "pikafish") is not None


def _has_python_chess() -> bool:
    return shutil.which is not None and __import_safe("chess")


def __import_safe(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def run_one(
    *,
    game: str,
    llm: LLMPlayer,
    movetime_ms: int,
    out_dir: str,
    max_plies: int,
) -> None:
    engine: UciEngine
    if game == "chess":
        engine = UciEngine.stockfish(movetime_ms=movetime_ms)
    else:
        engine = UciEngine.pikafish(movetime_ms=movetime_ms)
    engine.newgame()
    try:
        result = play_llm_vs_engine(
            game=game,
            llm=llm,
            engine=engine,
            rung_ms=movetime_ms,
            out_dir=out_dir,
            game_index=0,
            max_plies=max_plies,
        )
    finally:
        engine.close()
    print(
        f"[{game}] outcome={result.outcome} plies={result.n_plies} "
        f"fmt_fail={result.format_fail_count}/{result.llm_move_count} "
        f"parse_ok={result.parse_ok_count}/{result.llm_move_count} "
        f"log={result.log_path}"
    )


def main():
    p = argparse.ArgumentParser(description="Benchmark smoke test (1 game per game).")
    p.add_argument(
        "--game", choices=["chess", "xiangqi", "both", "auto"], default="auto"
    )
    p.add_argument("--movetime-ms", type=int, default=50)
    p.add_argument("--max-plies", type=int, default=60)
    p.add_argument("--out", type=str, default="data/benchmark/games/smoke")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()

    if args.game == "auto":
        wanted = []
        if _has_chess() and _has_python_chess():
            wanted.append("chess")
        if _has_xiangqi():
            wanted.append("xiangqi")
        if not wanted:
            print(
                "No supported game available: install stockfish (+ python-chess) "
                "or pikafish first.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.game == "both":
        wanted = ["chess", "xiangqi"]
    else:
        wanted = [args.game]

    os.makedirs(args.out, exist_ok=True)
    print(f"[smoke] running {wanted} at {args.movetime_ms} ms ...", flush=True)

    print(f"[smoke] loading {args.model} ...", flush=True)
    llm = LLMPlayer(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    for game in wanted:
        run_one(
            game=game,
            llm=llm,
            movetime_ms=args.movetime_ms,
            out_dir=args.out,
            max_plies=args.max_plies,
        )


if __name__ == "__main__":
    main()
