"""Calibrate per-game engine-movetime rungs to within-pool Elo.

For each game (chess / xiangqi) we play a small round-robin between adjacent
rungs (e.g. 10 ms vs 50 ms, 50 ms vs 200 ms, ...) plus the top-vs-bottom
pair as a long-range anchor, alternating colors. We then fit Bradley-Terry
ratings via :mod:`elo_estimator`, anchor the top rung at ``anchor_elo``
(default 3500), and serialise to ``data/benchmark/<game>_ladder_elo.json``.

The anchor is a convention - calling Pikafish/Stockfish at 5 s "Elo 3500"
puts both ladders near published top-engine strength (Stockfish 17 ~3640
Elo CCRL, Pikafish similar in xiangqi pools) without claiming the two
ladders are identical to the digit. The cross-game evidence in the
benchmark log is the win-rate-vs-movetime curve, not the literal numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.benchmark.elo_estimator import fit_ratings
from scripts.benchmark.engines import EnginePool
from scripts.benchmark.run_match import play_engine_vs_engine


def _rung_id(game: str, ms: int) -> str:
    return f"{game}_engine_{int(ms)}ms"


def _outcome_to_result_white_pov(outcome: str) -> str:
    if outcome == "white_wins":
        return "win"
    if outcome == "black_wins":
        return "loss"
    return "draw"


def _calibration_pairs(rungs_ms: List[int]) -> List[Tuple[int, int]]:
    """Adjacent pairs + one bottom-top anchor (helps fit the full spread).

    For rungs [10, 50, 200, 1000, 5000] this yields:
        (10, 50), (50, 200), (200, 1000), (1000, 5000), (10, 5000)
    """
    sorted_ms = sorted(set(int(x) for x in rungs_ms))
    if len(sorted_ms) < 2:
        return []
    pairs = list(zip(sorted_ms[:-1], sorted_ms[1:]))
    pairs.append((sorted_ms[0], sorted_ms[-1]))
    return pairs


def run_self_gauntlet(
    *,
    game: str,
    rungs_ms: List[int],
    pool: EnginePool,
    games_per_pair: int = 30,
    out_dir: str = "data/benchmark",
    max_plies: int = 300,
    log: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Engine-vs-engine round-robin between adjacent rungs + bottom-vs-top.

    Returns ``(games_for_elo, telemetry)`` where ``games_for_elo`` is the
    list-of-game-dicts expected by :func:`fit_ratings` (player ids =
    ``"<game>_engine_<ms>ms"``), and ``telemetry`` is a per-pair summary
    used for the log.
    """
    games_dir = os.path.join(out_dir, "games")
    os.makedirs(games_dir, exist_ok=True)

    pairs = _calibration_pairs(rungs_ms)
    games_for_elo: List[Dict[str, str]] = []
    pair_summary: Dict[str, Dict[str, Any]] = {}

    for lo_ms, hi_ms in pairs:
        wins_hi = draws = wins_lo = 0
        t_start = time.perf_counter()
        for game_idx in range(int(games_per_pair)):
            # Alternate colors: even games HI=white, odd games LO=white.
            if game_idx % 2 == 0:
                white_ms, black_ms = hi_ms, lo_ms
            else:
                white_ms, black_ms = lo_ms, hi_ms
            white_eng = pool.get(game, white_ms)
            black_eng = pool.get(game, black_ms)
            white_eng.newgame()
            black_eng.newgame()
            res = play_engine_vs_engine(
                game=game,
                white_engine=white_eng,
                black_engine=black_eng,
                out_dir=games_dir,
                game_index=game_idx,
                max_plies=max_plies,
            )
            outcome = res["outcome"]
            # Tally from "hi vs lo" view.
            if outcome == "draw":
                draws += 1
            elif (outcome == "white_wins" and white_ms == hi_ms) or (
                outcome == "black_wins" and black_ms == hi_ms
            ):
                wins_hi += 1
            else:
                wins_lo += 1

            games_for_elo.append(
                {
                    "white": _rung_id(game, white_ms),
                    "black": _rung_id(game, black_ms),
                    "result": _outcome_to_result_white_pov(outcome),
                }
            )
            if log:
                print(
                    f"  [{game}/cal] {hi_ms}vs{lo_ms} g{game_idx + 1}/{games_per_pair}: "
                    f"{outcome} (white={white_ms}ms, plies={res['n_plies']}, "
                    f"wall={res['wall_sec']:.1f}s)",
                    flush=True,
                )

        pair_key = f"{lo_ms}vs{hi_ms}"
        pair_summary[pair_key] = {
            "lo_ms": lo_ms,
            "hi_ms": hi_ms,
            "n": int(games_per_pair),
            "wins_hi": wins_hi,
            "wins_lo": wins_lo,
            "draws": draws,
            "score_hi": (wins_hi + 0.5 * draws) / max(1, int(games_per_pair)),
            "wall_sec": time.perf_counter() - t_start,
        }

    telemetry = {
        "game": game,
        "rungs_ms": list(sorted(set(int(x) for x in rungs_ms))),
        "pairs": pair_summary,
        "n_games_total": len(games_for_elo),
    }
    return games_for_elo, telemetry


def calibrate(
    *,
    game: str,
    rungs_ms: List[int],
    games_per_pair: int = 30,
    out_dir: str = "data/benchmark",
    anchor_elo: float = 3500.0,
    engine_threads: int = 1,
    engine_hash_mb: int = 64,
    save: bool = True,
    max_plies: int = 300,
) -> Dict[str, Any]:
    """Top-level entry: run gauntlet, fit ratings, save to JSON.

    Returns ``{rungs_ms, rung_elo, theta, anchor_rung_ms, n_calibration_games,
    sanity_low_vs_high_elo_gap, telemetry, output_path}``.
    """
    with EnginePool(threads=engine_threads, hash_mb=engine_hash_mb) as pool:
        games_for_elo, telemetry = run_self_gauntlet(
            game=game,
            rungs_ms=rungs_ms,
            pool=pool,
            games_per_pair=games_per_pair,
            out_dir=out_dir,
            max_plies=max_plies,
        )

    sorted_rungs = sorted(set(int(x) for x in rungs_ms))
    top_rung_id = _rung_id(game, sorted_rungs[-1])
    fitted, theta, nll = fit_ratings(
        games_for_elo,
        fixed_ratings={top_rung_id: float(anchor_elo)},
        initial_theta=1.5,
    )

    rung_elo: Dict[int, float] = {}
    for ms in sorted_rungs:
        rung_elo[int(ms)] = float(fitted[_rung_id(game, ms)])

    gap_low_high = rung_elo[sorted_rungs[-1]] - rung_elo[sorted_rungs[0]]
    output_path = os.path.join(out_dir, f"{game}_ladder_elo.json")
    payload = {
        "game": game,
        "anchor_rung_ms": sorted_rungs[-1],
        "anchor_elo": float(anchor_elo),
        "rungs_ms": sorted_rungs,
        "rung_elo": {str(k): v for k, v in rung_elo.items()},
        "rung_elo_player_ids": {str(k): _rung_id(game, int(k)) for k in sorted_rungs},
        "theta": float(theta),
        "nll": float(nll),
        "n_calibration_games": len(games_for_elo),
        "sanity_low_vs_high_elo_gap": float(gap_low_high),
        "sanity_low_vs_high_warn": (
            gap_low_high < 250.0
        ),  # plan: expect ~+500; warn loudly if << that
        "telemetry": telemetry,
    }
    if save:
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        payload["output_path"] = output_path
    return payload


def load_ladder(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cli_main():
    import argparse

    p = argparse.ArgumentParser(
        description="Calibrate engine-movetime ladder to within-pool Elo."
    )
    p.add_argument("--game", choices=["chess", "xiangqi"], required=True)
    p.add_argument(
        "--rungs",
        type=str,
        default="10,50,200,1000,5000",
        help="Comma-separated movetime rungs in ms (default 10,50,200,1000,5000).",
    )
    p.add_argument("--games-per-pair", type=int, default=30)
    p.add_argument("--anchor-elo", type=float, default=3500.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--out", type=str, default="data/benchmark")
    p.add_argument("--max-plies", type=int, default=300)
    args = p.parse_args()

    rungs = [int(x.strip()) for x in args.rungs.split(",") if x.strip()]
    result = calibrate(
        game=args.game,
        rungs_ms=rungs,
        games_per_pair=args.games_per_pair,
        out_dir=args.out,
        anchor_elo=args.anchor_elo,
        engine_threads=args.threads,
        engine_hash_mb=args.hash_mb,
        max_plies=args.max_plies,
    )
    print("\n== Ladder calibration done ==")
    print(
        json.dumps(
            {
                "game": result["game"],
                "rung_elo": result["rung_elo"],
                "theta": result["theta"],
                "n_games": result["n_calibration_games"],
                "low_vs_high_gap": result["sanity_low_vs_high_elo_gap"],
                "warn": result["sanity_low_vs_high_warn"],
                "out": result.get("output_path"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _cli_main()
