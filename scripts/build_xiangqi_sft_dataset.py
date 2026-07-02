#!/usr/bin/env python3
"""Build the Xiangqi-R1 §3.1 SFT JSONL (winner+draw moves, Pikafish-labeled).

Pipeline::

    [PGN corpus]  --iter_positions-->  filter winner+draw  --Pikafish-->  JSONL rows
    [self-play]   --iter_selfplay---->  filter winner+draw  --Pikafish-->  JSONL rows  (top-up)

Each row::

    {"messages": [system, user, assistant], "meta": {fen, played_uci, best_uci,
       value_red_cp, situation_3, situation_5, is_good_move, phase, ply, source, turn}}

The ``--target-move`` switch selects what the assistant must produce:

* ``best``        — engine best move (paper Stage-1 style, ~5M positions in paper). Best
                    signal-to-noise; default.
* ``human_good``  — the **human-played** move, kept iff |V(after played) - V(after best)|
                    ≤ σ_good=100 (paper Eq. 3). Smaller, expert-imitation flavor.

Requires ``PIKAFISH_BIN`` (or ``--pikafish-bin``) pointing to a real Pikafish executable
with ``pikafish.nnue`` next to it (same as the GRPO v2 training script).

Example::

    export PIKAFISH_BIN=$(which pikafish)
    uv run python scripts/download_xiangqi_pgn.py
    # Smoke test (~minutes)
    uv run python scripts/build_xiangqi_sft_dataset.py --samples 512
    # Full Stage-2-scale build (hours)
    uv run python scripts/build_xiangqi_sft_dataset.py --samples 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for _p in (ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tqdm import tqdm

from src.pikafish_eval import PikafishEvaluator
from xiangqi_pgn import iter_positions
from xiangqi_selfplay import iter_selfplay
from src.xiangqi_labels import (
    SIGMA_GOOD,
    is_good_move,
    root_value_red_oriented,
    situation_3class,
    situation_5class,
)

SYSTEM_PROMPT = """You are a Xiangqi (Chinese Chess) coach for the side to move.
Always answer in exactly two final lines:
  Situation: <Balanced | Advantage_Red | Advantage_Black>
  Move: <one legal UCCI move like h2e2>

Centipawn scale, Red-positive. Use the FEN and graphic. Prefer the engine-recommended
move when it matches the position; in commentary, briefly justify the move before the
``Situation:`` and ``Move:`` lines."""


def _phase(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 60:
        return "middlegame"
    return "endgame"


def _fen_to_graphic(fen: str) -> str:
    """Render a Xiangqi FEN piece-placement as a coordinate grid (rank 9 top, rank 0 bottom)."""
    placement = fen.split()[0]
    rows = placement.split("/")
    lines = ["  a b c d e f g h i"]
    for rank_idx, row in enumerate(rows):
        rank = 9 - rank_idx
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend(["."] * int(ch))
            else:
                cells.append(ch)
        lines.append(f"{rank} " + " ".join(cells))
        if rank == 5:
            lines.append("  ~~~~~~~~~~~~~~~~~")
    return "\n".join(lines)


def _build_messages(
    fen: str,
    played_uci: str,
    meta: Dict[str, Any],
    target_uci: str,
) -> list[Dict[str, str]]:
    turn = fen.split()[1] if " " in fen else "w"
    side = "Red" if turn == "w" else "Black"
    graphic = _fen_to_graphic(fen)
    headers = meta.get("headers", {}) or {}
    src_note = (
        f"Source: {meta.get('source', 'pgn')}; "
        f"Event: {headers.get('Event', '?')}; "
        f"Red: {headers.get('Red', '?')} vs Black: {headers.get('Black', '?')}; "
        f"winner: {meta.get('winner', '?')}."
    )
    user = (
        f"{src_note}\n"
        f"Side to move: {side} ({turn}).\n"
        f"FEN: {fen}\n"
        f"Board:\n{graphic}\n"
        f"Summarize the situation for Red and recommend the best move."
    )
    rationale = (
        f"Pikafish root score (Red-positive cp): {meta['value_red_cp']:.0f}. "
        f"Coarse class: {meta['situation_3']}; fine class: {meta['situation_5']}. "
        f"Engine best move: {meta['best_uci']}. "
        f"Human-played move: {played_uci} (good={bool(meta['is_good_move'])})."
    )
    assistant = (
        f"<think>\n{rationale}\n</think>\n"
        f"Situation: {meta['situation_3']}\n"
        f"Move: {target_uci}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _winner_matches_side(winner: str, turn: str, strict: bool) -> bool:
    """Paper §3.1: retain winner-side moves and ALL moves from drawn games.

    ``strict=False`` additionally keeps ``winner="unknown"`` games (PGN without a
    ``[Result]`` tag and no terminal mate) as draw-equivalent, so we don't reject
    almost the entire wukong-xiangqi corpus. Pass ``strict=True`` to enforce the
    paper rule exactly (requires a Result tag or an inferred mate).
    """
    if winner == "draw":
        return True
    if not strict and winner == "unknown":
        return True
    if winner == "red" and turn == "w":
        return True
    if winner == "black" and turn == "b":
        return True
    return False


def _label(
    eng: PikafishEvaluator, fen: str, played_uci: str
) -> Optional[Dict[str, Any]]:
    best_uci, root_cp = eng.bestmove_root_cached(fen)
    if best_uci is None or root_cp is None:
        return None
    value_red = root_value_red_oriented(fen, root_cp)
    if value_red is None:
        return None
    good, vp, vb = is_good_move(fen, played_uci, best_uci, eng.evaluate_cp, SIGMA_GOOD)
    return {
        "best_uci": best_uci,
        "root_cp_stm": float(root_cp),
        "value_red_cp": float(value_red),
        "situation_3": situation_3class(value_red),
        "situation_5": situation_5class(value_red),
        "is_good_move": bool(good),
        "value_red_after_played_cp": None if vp is None else float(vp),
        "value_red_after_best_cp": None if vb is None else float(vb),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pgn",
        type=str,
        default="data/xiangqi_sft/raw/xqdb_masters_40711_UCI_games.pgn",
        help="Path to the wukong-xiangqi UCI-notation PGN (see download_xiangqi_pgn.py).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/xiangqi_sft/xiangqi_sft_train.jsonl",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Target row count (paper Stage-2 scale). Use --samples 512 for a smoke test.",
    )
    ap.add_argument(
        "--target-move",
        choices=["best", "human_good"],
        default="best",
        help="Assistant target: engine-best (default) or human-played move filtered by paper Eq. 3.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-games", type=int, default=0, help="0 = no cap on PGN games.")
    ap.add_argument(
        "--max-ply",
        type=int,
        default=160,
        help="Drop positions past this ply (engine eval gets noisy in long endgames).",
    )
    ap.add_argument(
        "--pikafish-bin", type=str, default=os.environ.get("PIKAFISH_BIN", "")
    )
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--movetime-ms", type=int, default=400)
    ap.add_argument(
        "--selfplay-topup-games",
        type=int,
        default=2000,
        help="Self-play games to drive if PGN doesn't reach --samples (each game ~80 plies).",
    )
    ap.add_argument(
        "--no-selfplay-topup",
        action="store_true",
        help="Disable self-play fallback entirely.",
    )
    ap.add_argument(
        "--paper-strict-filter",
        action="store_true",
        help=(
            "Drop wukong-style games that lack a [Result] tag AND don't end in mate. "
            "Off by default because the wukong corpus has no Result tags; the relaxed "
            "default treats those as draw-equivalent (both sides kept)."
        ),
    )
    args = ap.parse_args()

    if not args.pikafish_bin.strip():
        raise SystemExit(
            "Set a real Pikafish executable, e.g.\n"
            "  export PIKAFISH_BIN=$(which pikafish)\n"
            "or:  --pikafish-bin /full/path/to/pikafish"
        )
    if "path/to" in args.pikafish_bin.lower():
        raise SystemExit(
            f"PIKAFISH_BIN looks like a placeholder ({args.pikafish_bin!r})."
        )
    if not os.path.isfile(args.pgn):
        raise SystemExit(
            f"PGN not found: {args.pgn!r}\n"
            "Run:  uv run python scripts/download_xiangqi_pgn.py  to fetch it."
        )

    eng = PikafishEvaluator(
        binary_path=args.pikafish_bin,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        verbose=False,
    )
    if not eng.enabled:
        raise SystemExit(
            "Pikafish failed to start. Check the binary path and ``pikafish.nnue`` placement."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    pgn_seen = 0
    rejected_loser = 0
    rejected_label = 0
    rejected_target = 0

    def _sources() -> Iterator[Dict[str, Any]]:
        nonlocal pgn_seen
        for p in iter_positions(
            args.pgn,
            max_games=(args.max_games if args.max_games > 0 else None),
            drop_unresolved=bool(args.paper_strict_filter),
        ):
            pgn_seen += 1
            p["source"] = "pgn"
            yield p
        if args.no_selfplay_topup:
            return
        if args.selfplay_topup_games > 0:
            for p in iter_selfplay(
                eng,
                n_games=args.selfplay_topup_games,
                seed=args.seed,
            ):
                p["source"] = "selfplay"
                yield p

    pbar = tqdm(total=args.samples, desc="xiangqi_sft", unit="row")
    try:
        with open(args.out, "w", encoding="utf-8") as fout:
            for pos in _sources():
                if written >= args.samples:
                    break
                if pos["ply"] > args.max_ply:
                    continue
                if not _winner_matches_side(
                    pos["winner"], pos["turn"], strict=bool(args.paper_strict_filter)
                ):
                    rejected_loser += 1
                    continue

                fen = pos["fen"]
                played = pos["played_uci"]
                labels = _label(eng, fen, played)
                if labels is None:
                    rejected_label += 1
                    continue

                target_uci = (
                    labels["best_uci"] if args.target_move == "best" else played
                )
                if args.target_move == "human_good" and not labels["is_good_move"]:
                    rejected_target += 1
                    continue

                meta = {
                    "fen": fen,
                    "turn": pos["turn"],
                    "ply": pos["ply"],
                    "phase": _phase(int(pos["ply"])),
                    "winner": pos["winner"],
                    "played_uci": played,
                    "source": pos["source"],
                    "headers": pos.get("headers", {}),
                    **labels,
                }
                row = {
                    "messages": _build_messages(fen, played, meta, target_uci),
                    "meta": meta,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)
    finally:
        pbar.close()
        eng.close()

    summary = {
        "written": written,
        "pgn_positions_seen": pgn_seen,
        "rejected_losing_side": rejected_loser,
        "rejected_engine_label_fail": rejected_label,
        "rejected_target_not_good": rejected_target,
        "target_move": args.target_move,
        "out": args.out,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
