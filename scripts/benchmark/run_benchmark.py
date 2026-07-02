"""Top-level CLI for the inference-only LLM Elo benchmark.

Pipeline per game::

    1. Load (or recompute) the engine-movetime ladder Elos
       (data/benchmark/<game>_ladder_elo.json).
    2. Play ``--games-per-rung`` LLM-vs-engine games at each rung,
       alternating colors is NOT supported (LLM plays uppercase only -
       see docs/logs/2026-05-13-log-llm-chess-xiangqi-elo-bench.md).
    3. Fit the LLM's Elo via Bradley-Terry MLE with the rung Elos pinned.
    4. Bootstrap 95% CI on the LLM Elo.
    5. Write data/benchmark/<game>_results.json and a win-rate plot.

The same ``LLMPlayer`` is reused across both games (one model load).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.benchmark.calibrate_ladder import (
    _rung_id,
    calibrate,
    load_ladder,
)
from scripts.benchmark.elo_estimator import (
    bootstrap_ci,
    fit_ratings,
    predicted_score,
)
from scripts.benchmark.engines import EnginePool
from scripts.benchmark.llm_player import LLMPlayer
from scripts.benchmark.run_match import (
    gameresult_to_dict,
    play_match,
    summarize_results,
)


LLM_PLAYER_ID = "llm_qwen2.5_7b_instruct_base"


def _outcome_to_result_white_pov(outcome: str) -> str:
    if outcome == "white_wins":
        return "win"
    if outcome == "black_wins":
        return "loss"
    return "draw"


def _llm_games_for_elo(
    game: str,
    rung_ms: int,
    game_results,
) -> List[Dict[str, str]]:
    """LLM always plays White (uppercase) in the current benchmark."""
    rung = _rung_id(game, rung_ms)
    out: List[Dict[str, str]] = []
    for r in game_results:
        out.append(
            {
                "white": LLM_PLAYER_ID,
                "black": rung,
                "result": _outcome_to_result_white_pov(r.outcome),
            }
        )
    return out


def _plot_winrate(
    out_path: str,
    game: str,
    rungs_ms: List[int],
    win_rates: Dict[int, float],
    score: Dict[int, float],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as err:
        print(f"[plot] matplotlib unavailable: {err}", file=sys.stderr)
        return None

    xs = sorted(rungs_ms)
    win_ys = [win_rates[r] for r in xs]
    score_ys = [score[r] for r in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, win_ys, marker="o", label="Win rate")
    ax.plot(xs, score_ys, marker="s", linestyle="--", label="Score (W=1, D=0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Engine thinking time per move (ms, log scale)")
    ax.set_ylabel("LLM result rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{game.title()}: LLM-vs-engine result rate by movetime")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def benchmark_one_game(
    *,
    game: str,
    llm: LLMPlayer,
    rungs_ms: List[int],
    games_per_rung: int,
    out_dir: str,
    ladder: Dict[str, Any],
    max_plies: int,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    games_dir = os.path.join(out_dir, "games")
    os.makedirs(games_dir, exist_ok=True)

    all_llm_games_for_elo: List[Dict[str, str]] = []
    rung_telemetry: Dict[int, Dict[str, Any]] = {}
    per_rung_game_dicts: Dict[int, List[Dict[str, Any]]] = {}

    with EnginePool() as pool:
        for ms in sorted(rungs_ms):
            engine = pool.get(game, ms)
            engine.newgame()
            print(f"\n[{game}/llm] rung {ms} ms: {games_per_rung} games", flush=True)
            t0 = time.perf_counter()

            def on_progress(i: int, n: int, r) -> None:
                print(
                    f"  game {i + 1}/{n}: {r.outcome} plies={r.n_plies} "
                    f"fmt_fail={r.format_fail_count}/{r.llm_move_count} "
                    f"wall_llm={r.llm_wall_sec:.1f}s",
                    flush=True,
                )

            results = play_match(
                game=game,
                llm=llm,
                engine=engine,
                n_games=games_per_rung,
                out_dir=out_dir,
                rung_ms=ms,
                max_plies=max_plies,
                progress_callback=on_progress,
            )
            summary = summarize_results(results)
            summary["wall_sec"] = time.perf_counter() - t0
            rung_telemetry[ms] = summary
            per_rung_game_dicts[ms] = [gameresult_to_dict(r) for r in results]
            all_llm_games_for_elo.extend(_llm_games_for_elo(game, ms, results))

    # Build the combined Elo fit: rungs pinned to calibrated Elos, LLM free.
    rung_elo_floats: Dict[str, float] = {
        _rung_id(game, int(k)): float(v) for k, v in ladder["rung_elo"].items()
    }
    fitted, theta, nll = fit_ratings(
        all_llm_games_for_elo,
        fixed_ratings=rung_elo_floats,
        initial_theta=float(ladder.get("theta", 1.5)),
        initial_ratings={LLM_PLAYER_ID: 1500.0},
    )
    llm_point = float(fitted[LLM_PLAYER_ID])

    ci = bootstrap_ci(
        all_llm_games_for_elo,
        fixed_ratings=rung_elo_floats,
        target_player=LLM_PLAYER_ID,
        n_reps=int(bootstrap_reps),
        seed=int(bootstrap_seed),
        initial_theta=float(theta),
    )

    win_rates = {
        ms: rung_telemetry[ms]["wins"] / max(1, rung_telemetry[ms]["n"])
        for ms in rungs_ms
    }
    score = {ms: rung_telemetry[ms]["score"] for ms in rungs_ms}

    # Compare model to predicted score at each rung for residual diagnostics.
    predicted_rates = {}
    for ms in rungs_ms:
        predicted_rates[ms] = predicted_score(
            llm_point, rung_elo_floats[_rung_id(game, int(ms))], theta=theta
        )

    plot_path = os.path.join(out_dir, f"{game}_winrate.png")
    plot_written = _plot_winrate(plot_path, game, rungs_ms, win_rates, score)

    result_payload = {
        "game": game,
        "llm_player_id": LLM_PLAYER_ID,
        "llm_elo": llm_point,
        "llm_elo_ci95": [ci["lo"], ci["hi"]],
        "theta": float(theta),
        "nll": float(nll),
        "rungs_ms": sorted(rungs_ms),
        "rung_elo": ladder["rung_elo"],
        "win_rates_by_rung": {str(k): float(v) for k, v in win_rates.items()},
        "score_by_rung": {str(k): float(v) for k, v in score.items()},
        "predicted_score_by_rung": {
            str(k): float(v) for k, v in predicted_rates.items()
        },
        "rung_telemetry": {str(k): v for k, v in rung_telemetry.items()},
        "n_games_total": len(all_llm_games_for_elo),
        "bootstrap_reps_used": ci["n_reps"],
        "anchor_rung_ms": ladder["anchor_rung_ms"],
        "anchor_elo": ladder["anchor_elo"],
        "plot_path": plot_written,
    }
    out_path = os.path.join(out_dir, f"{game}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    result_payload["output_path"] = out_path
    return result_payload


def main():
    p = argparse.ArgumentParser(description="LLM chess + xiangqi Elo benchmark.")
    p.add_argument("--game", choices=["chess", "xiangqi", "both"], default="both")
    p.add_argument(
        "--rungs",
        type=str,
        default="10,50,200,1000,5000",
        help="Comma-separated engine movetime rungs in ms.",
    )
    p.add_argument("--games-per-rung", type=int, default=20)
    p.add_argument("--calibration-games-per-pair", type=int, default=30)
    p.add_argument("--anchor-elo", type=float, default=3500.0)
    p.add_argument("--out", type=str, default="data/benchmark")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recompute the engine ladder even if the JSON exists.",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only run calibration; do not load the LLM or play LLM matches.",
    )
    args = p.parse_args()

    rungs = [int(x.strip()) for x in args.rungs.split(",") if x.strip()]
    games = ["chess", "xiangqi"] if args.game == "both" else [args.game]
    os.makedirs(args.out, exist_ok=True)

    # Calibration first (cheap, engine-only).
    ladders: Dict[str, Dict[str, Any]] = {}
    for game in games:
        ladder_path = os.path.join(args.out, f"{game}_ladder_elo.json")
        if (not args.recalibrate) and os.path.isfile(ladder_path):
            ladder = load_ladder(ladder_path)
            assert ladder is not None
            print(f"[{game}/cal] reusing existing ladder: {ladder_path}", flush=True)
        else:
            print(f"\n[{game}/cal] calibrating ladder ({rungs}) ...", flush=True)
            ladder = calibrate(
                game=game,
                rungs_ms=rungs,
                games_per_pair=args.calibration_games_per_pair,
                out_dir=args.out,
                anchor_elo=args.anchor_elo,
                max_plies=args.max_plies,
            )
        ladders[game] = ladder
        gap = ladder.get("sanity_low_vs_high_elo_gap", 0.0)
        warn = "WARN " if ladder.get("sanity_low_vs_high_warn") else "ok   "
        print(
            f"[{game}/cal] {warn}rung_elo={ladder['rung_elo']} "
            f"theta={ladder['theta']:.3f} top-vs-bot_gap={gap:.0f}"
        )

    if args.skip_llm:
        print("\n[--skip-llm] stopping after calibration.")
        return

    # One LLM player shared across games.
    print(f"\n[llm] loading {args.model} ...", flush=True)
    llm = LLMPlayer(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    summary: Dict[str, Any] = {"games": {}}
    for game in games:
        print(f"\n=== {game.upper()} ===", flush=True)
        result = benchmark_one_game(
            game=game,
            llm=llm,
            rungs_ms=rungs,
            games_per_rung=args.games_per_rung,
            out_dir=args.out,
            ladder=ladders[game],
            max_plies=args.max_plies,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.seed,
        )
        summary["games"][game] = {
            "llm_elo": result["llm_elo"],
            "llm_elo_ci95": result["llm_elo_ci95"],
            "win_rates_by_rung": result["win_rates_by_rung"],
            "n_games_total": result["n_games_total"],
            "output_path": result["output_path"],
            "plot_path": result["plot_path"],
        }

    if "chess" in summary["games"] and "xiangqi" in summary["games"]:
        c = summary["games"]["chess"]["llm_elo"]
        x = summary["games"]["xiangqi"]["llm_elo"]
        summary["chess_minus_xiangqi_elo_gap"] = float(c - x)

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n== Done ==")
    print(json.dumps(summary, indent=2))
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
