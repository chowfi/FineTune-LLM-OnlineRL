"""Checkpoint arena: adjacent-pair matches + relative Elo.

Spec: docs/superpowers/specs/2026-07-07-checkpoint-elo-arena-design.md.
Run offline: `uv run python -m muzero.arena` (needs PIKAFISH_BIN, like the
gate). Ratings are relative (oldest checkpoint anchored at 0) and only
comparable within a single --sims setting."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, replace

import torch

from muzero.config import MuZeroConfig
from muzero.encoding import absolute_visits, index_to_move
from muzero.env import XiangqiEnv
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet
from muzero.selfplay import canonical_root
from scripts.benchmark.elo_estimator import fit_ratings
from src.xiangqi_board import engine_uci_to_algebraic

_ITER_RE = re.compile(r"iter_?(\d+)")


@dataclass
class Checkpoint:
    label: str
    path: str
    iteration: int


def discover_checkpoints(
    archive_dir: str, extras: list[str] | None = None
) -> list[Checkpoint]:
    """archive/iter_*.pt plus extras, sorted NUMERICALLY by training iteration.

    Iteration comes from the checkpoint's own "iteration" key when present,
    else from an iter<NNNN> pattern in the filename. Extras are relabelled
    to canonical iter_NNNN form so labels stay stable across runs (e.g. a
    moving latest.pt can't accumulate games under a drifting identity);
    entries sharing an iteration are the same weights by construction and
    are deduped (archive entry kept)."""
    paths: list[tuple[str, bool]] = []
    if os.path.isdir(archive_dir):
        paths = [
            (os.path.join(archive_dir, f), False)
            for f in sorted(os.listdir(archive_dir))
            if f.startswith("iter_") and f.endswith(".pt")
        ]
    paths += [(p, True) for p in extras or []]
    out = []
    for p, is_extra in paths:
        label = os.path.splitext(os.path.basename(p))[0]
        ckpt = torch.load(p, map_location="cpu")
        iteration = ckpt.get("iteration")
        if iteration is None:
            m = _ITER_RE.search(label)
            if not m:
                raise ValueError(f"cannot determine iteration for {p}")
            iteration = int(m.group(1))
        if is_extra:
            label = f"iter_{int(iteration):04d}"
        out.append(Checkpoint(label=label, path=p, iteration=int(iteration)))
    out.sort(key=lambda c: c.iteration)
    deduped: list[Checkpoint] = []
    for c in out:
        if deduped and deduped[-1].iteration == c.iteration:
            continue  # same weights archived under another name
        deduped.append(c)
    return deduped


def games_needed(
    rows: list[dict], a: str, b: str, *, sims: int, games_per_pair: int
) -> int:
    played = sum(
        1 for r in rows if {r["white"], r["black"]} == {a, b} and r.get("sims") == sims
    )
    return max(0, games_per_pair - played)


def _load_player(cfg: MuZeroConfig, path: str):
    net = MuZeroNet(cfg).to(cfg.device)
    ckpt = torch.load(path, map_location=cfg.device)
    net.load_state_dict(ckpt["ally"])
    net.eval()
    return NetRunner(net, cfg.device), MCTS(cfg)


def _play_game(cfg: MuZeroConfig, evaluator, players: dict, opening_uci: str) -> str:
    """players: side ("w"/"b") -> (runner, mcts). Returns win/loss/draw
    from WHITE's (red's) perspective — the elo_estimator convention."""
    env = XiangqiEnv(cfg, evaluator)
    env.reset()
    opening = engine_uci_to_algebraic(opening_uci)
    if opening is not None and opening in env.legal_moves():
        env.step(opening)
    while env.result is None:
        runner, mcts = players[env.side_to_move]
        obs, legal = canonical_root(env)
        if len(legal) == 0:
            # Unreachable with a live engine: a side with no legal moves
            # would have ended the game on the PREVIOUS ply (mover wins).
            # An empty list here means the engine died mid-game.
            raise RuntimeError(
                "engine returned no legal moves mid-game (Pikafish failure?) "
                "— aborting instead of recording a fake draw"
            )
        ((visits, _, _),) = mcts.run(runner, [(obs, legal)], add_noise=False)
        visits = absolute_visits(visits, env.side_to_move)
        env.step(index_to_move(max(visits, key=visits.get)))
    if env.result == "red_win":
        return "win"
    if env.result == "black_win":
        return "loss"
    return "draw"


def play_pair(
    cfg: MuZeroConfig, evaluator, a: tuple, b: tuple, n_games: int, start: int = 0
) -> list[dict]:
    """a/b: (label, path). Plays n_games alternating colors, cycling the
    opening book every two games. `start` continues the opening/color
    rotation where a previous run's jsonl left off, so idempotent top-ups
    don't replay identical games. Returns jsonl-ready row dicts."""
    (label_a, path_a), (label_b, path_b) = a, b
    player_a = _load_player(cfg, path_a)
    player_b = _load_player(cfg, path_b)
    rows = []
    for g in range(start, start + n_games):
        opening = cfg.opening_book[(g // 2) % len(cfg.opening_book)]
        a_is_white = g % 2 == 0
        players = (
            {"w": player_a, "b": player_b}
            if a_is_white
            else {"w": player_b, "b": player_a}
        )
        result = _play_game(cfg, evaluator, players, opening)
        rows.append(
            {
                "white": label_a if a_is_white else label_b,
                "black": label_b if a_is_white else label_a,
                "result": result,
                "sims": cfg.num_simulations,
                "opening": opening,
            }
        )
    return rows


def fit_arena_elo(rows: list[dict], order: list[str]) -> dict:
    """Relative Elo with the oldest player (order[0]) anchored at 0."""
    games = [
        {"white": r["white"], "black": r["black"], "result": r["result"]} for r in rows
    ]
    ratings, _theta, _nll = fit_ratings(games, fixed_ratings={order[0]: 0.0})
    return {label: float(ratings.get(label, 0.0)) for label in order}


def main() -> None:
    from src.pikafish_eval import PikafishEvaluator

    ap = argparse.ArgumentParser(description="Checkpoint arena + relative Elo")
    ap.add_argument("--archive-dir", default="checkpoints/muzero_xiangqi/archive")
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional checkpoint .pt files (repeatable), e.g. latest.pt",
    )
    ap.add_argument("--games-per-pair", type=int, default=20)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="data/arena")
    args = ap.parse_args()

    # self_play_mode="latest" pins symmetric truncation adjudication
    # regardless of future config defaults.
    cfg = replace(
        MuZeroConfig(),
        num_simulations=args.sims,
        device=args.device,
        self_play_mode="latest",
    )
    checkpoints = discover_checkpoints(args.archive_dir, extras=args.extra)
    if len(checkpoints) < 2:
        raise SystemExit(
            f"Need >= 2 checkpoints, found {len(checkpoints)} "
            f"(archive: {args.archive_dir}; use --extra to add more)"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    games_path = os.path.join(args.out_dir, "games.jsonl")
    rows: list[dict] = []
    if os.path.exists(games_path):
        with open(games_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    evaluator = PikafishEvaluator(
        binary_path=cfg.pikafish_bin,
        depth=cfg.pikafish_depth,
        timeout_sec=cfg.pikafish_timeout_sec,
        movetime_ms=cfg.pikafish_movetime_ms,
        verbose=False,
    )
    if not getattr(evaluator, "enabled", True):
        raise SystemExit(
            "Pikafish is not available (set PIKAFISH_BIN) — refusing to play "
            "arena games without a legality source."
        )
    for prev, curr in zip(checkpoints, checkpoints[1:]):
        need = games_needed(
            rows,
            prev.label,
            curr.label,
            sims=args.sims,
            games_per_pair=args.games_per_pair,
        )
        if need == 0:
            print(f"[arena] {prev.label} vs {curr.label}: complete, skipping")
            continue
        print(f"[arena] {prev.label} vs {curr.label}: playing {need} games ...")
        played = args.games_per_pair - need
        new_rows = play_pair(
            cfg,
            evaluator,
            (prev.label, prev.path),
            (curr.label, curr.path),
            need,
            start=played,
        )
        with open(games_path, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
        rows.extend(new_rows)

    order = [c.label for c in checkpoints]
    fit_rows = [r for r in rows if r.get("sims") == args.sims]
    skipped = len(rows) - len(fit_rows)
    if skipped:
        print(f"[arena] ignoring {skipped} rows from other --sims settings")
    anchor = order[0]
    if not any(anchor in (r["white"], r["black"]) for r in fit_rows):
        raise SystemExit(
            f"anchor {anchor!r} has no games at --sims={args.sims}; "
            "cannot fit identified ratings (play games first)"
        )
    ratings = fit_arena_elo(fit_rows, order)
    print(f"\n{'checkpoint':<28}{'iter':>6}{'Elo':>8}{'games':>7}")
    for c in checkpoints:
        n = sum(1 for r in fit_rows if c.label in (r["white"], r["black"]))
        elo = f"{ratings[c.label]:>8.0f}" if n or c.label == anchor else f"{'--':>8}"
        print(f"{c.label:<28}{c.iteration:>6}{elo}{n:>7}")
    print(
        "\nNote: ~20 games/pair => roughly +-80 Elo per step; read the curve's"
        " shape across several checkpoints, not neighbor differences."
    )
    with open(os.path.join(args.out_dir, "ratings.json"), "w") as f:
        json.dump(
            {
                c.label: {"iteration": c.iteration, "elo": ratings[c.label]}
                for c in checkpoints
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
