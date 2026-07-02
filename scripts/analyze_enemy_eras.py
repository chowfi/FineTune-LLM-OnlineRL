#!/usr/bin/env python3
"""Summarize per-episode RL metrics grouped by frozen self-play enemy generation.

Reads ``chinese_chess_episode_metrics_v2.csv`` and (optionally) enriches
``game_mean_chosen_engine_reward`` from wandb stdout logs when the CSV column
is empty for older rows.

Usage:
  uv run python scripts/analyze_enemy_eras.py
  uv run python scripts/analyze_enemy_eras.py --ep-start 5 --ep-end 32
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/metrics/chinese_chess_episode_metrics_v2.csv"

# Known sync boundaries when ``game_self_play_enemy_id`` is missing from CSV.
KNOWN_SYNC_AFTER = (15, 30)

METRICS = (
    ("game_mean_chosen_engine_reward", "mean_chosen_engine_reward"),
    ("game_mean_chosen_cp_delta_clipped", "cp_delta_clipped"),
    ("game_chosen_is_engine_argmax_in_group_rate", "engine_argmax_%"),
    ("game_median_chosen_engine_rank_in_group", "median_engine_rank"),
    ("game_median_ally_cp_after_move_red", "median_ally_cp_after"),
)


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_enemy_id(episode: int) -> int:
    if episode >= 31:
        return 2
    if episode >= 16:
        return 1
    return 0


def enemy_label(enemy_id: int) -> str:
    labels = {
        0: "Enemy A (pre ep15 sync)",
        1: "Enemy B (ally@ep15 .. pre ep30 sync)",
        2: "Enemy C (ally@ep30 sync)",
    }
    return labels.get(enemy_id, f"Enemy {enemy_id}")


def parse_engine_reward_from_logs(log_globs: Iterable[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    pat = re.compile(
        r"^\[Ep (\d+)\] ally_return=.*? mean_chosen_engine_reward=([\d.]+) "
    )
    for pattern in log_globs:
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        out[int(m.group(1))] = float(m.group(2))
    return out


def load_rows(csv_path: Path, log_globs: Iterable[str]) -> List[Dict[str, Any]]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    log_rewards = parse_engine_reward_from_logs(log_globs)
    out: List[Dict[str, Any]] = []
    for row in rows:
        ep = int(float(row["episode"]))
        enemy_id_raw = row.get("game_self_play_enemy_id", "")
        enemy_id = (
            int(float(enemy_id_raw))
            if str(enemy_id_raw).strip()
            else infer_enemy_id(ep)
        )
        eng_rew = _f(row.get("game_mean_chosen_engine_reward"))
        if eng_rew is None:
            eng_rew = log_rewards.get(ep)
        out.append(
            {
                "episode": ep,
                "enemy_id": enemy_id,
                "rounds": int(float(row["rounds"])),
                "ally_turns": int(float(row["ally_turns_episode"])),
                "global_step_end": _f(row.get("game_global_train_step_end")),
                "outcome": row["outcome"],
                "mean_chosen_engine_reward": eng_rew,
                "cp_delta_clipped": _f(row.get("game_mean_chosen_cp_delta_clipped")),
                "engine_argmax_%": _f(
                    row.get("game_chosen_is_engine_argmax_in_group_rate")
                ),
                "median_engine_rank": _f(
                    row.get("game_median_chosen_engine_rank_in_group")
                ),
                "median_ally_cp_after": _f(
                    row.get("game_median_ally_cp_after_move_red")
                ),
            }
        )
    return out


def steps_in_episode(rows: List[Dict[str, Any]], idx: int) -> int:
    cur = rows[idx]["ally_turns"]
    prev_end = rows[idx - 1]["global_step_end"] if idx > 0 else None
    end = rows[idx]["global_step_end"]
    if prev_end is not None and end is not None:
        return int(end - prev_end)
    return int(cur)


def summarize_group(name: str, items: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}")
    print(name)
    if not items:
        print("  (no episodes)")
        return
    eps = [x["episode"] for x in items]
    steps = sum(x["ally_turns"] for x in items)
    print(f"  Episodes {min(eps)}–{max(eps)} (n={len(items)}), ally_turns={steps}")
    outcomes: Dict[str, int] = defaultdict(int)
    for x in items:
        outcomes[x["outcome"]] += 1
    print(f"  Outcomes: {dict(outcomes)}")
    wins = sum(1 for x in items if x["outcome"] == "ally_win")
    losses = sum(1 for x in items if x["outcome"] == "enemy_win")
    if wins + losses:
        print(
            f"  Win rate (decided): {100 * wins / (wins + losses):.0f}% ({wins}/{wins + losses})"
        )

    def avg(key: str) -> Optional[float]:
        vals = [x[key] for x in items if x.get(key) is not None]
        return stats.mean(vals) if vals else None

    print("  ALL episodes:")
    for _, label in METRICS:
        v = avg(label)
        if v is not None:
            print(f"    {label}: {v:.2f}")

    win_items = [x for x in items if x["outcome"] == "ally_win"]
    if win_items:
        print(f"  WINS only (n={len(win_items)}):")
        print(f"    rounds_to_win: {stats.mean([x['rounds'] for x in win_items]):.1f}")
        for _, label in METRICS:
            vals = [x[label] for x in win_items if x.get(label) is not None]
            if vals:
                print(f"    {label}: {stats.mean(vals):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ep-start", type=int, default=5)
    parser.add_argument("--ep-end", type=int, default=10_000)
    parser.add_argument(
        "--log-glob",
        action="append",
        default=[str(ROOT / "wandb" / "run-*" / "files" / "output.log")],
        help="Glob(s) for wandb output.log episode summaries.",
    )
    args = parser.parse_args()

    rows = [
        r
        for r in load_rows(args.csv, args.log_glob)
        if args.ep_start <= r["episode"] <= args.ep_end
    ]
    rows.sort(key=lambda r: r["episode"])

    print("ENEMY SYNC TIMELINE (known / inferred)")
    for ep in KNOWN_SYNC_AFTER:
        print(f"  After ep {ep}: enemy generation increments (ally weights copied)")
    print()
    print(
        "ep | enemy | steps | rounds | outcome      | eng_rew | cp_Δclip | "
        "argmax% | med_rank | med_cp"
    )
    print("-" * 96)
    for idx, r in enumerate(rows):
        steps = steps_in_episode(rows, idx)
        er = r["mean_chosen_engine_reward"]
        er_s = f"{er:.3f}" if er is not None else "n/a"
        print(
            f"{r['episode']:2d} | {r['enemy_id']:5d} | {steps:5d} | "
            f"{r['rounds']:3d}    | {r['outcome']:12s} | {er_s:>5} | "
            f"{r['cp_delta_clipped']:8.1f} | {r['engine_argmax_%']:5.1f} | "
            f"{r['median_engine_rank']:8.1f} | {r['median_ally_cp_after']:7.1f}"
        )

    by_enemy: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_enemy[r["enemy_id"]].append(r)
    for enemy_id in sorted(by_enemy):
        summarize_group(enemy_label(enemy_id), by_enemy[enemy_id])


if __name__ == "__main__":
    main()
