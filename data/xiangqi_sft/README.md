# Xiangqi strategy SFT data

This directory holds the **Xiangqi-R1 §3.1**-aligned SFT corpus the model trains on
before / alongside GRPO. The pipeline is paper-aligned: real master-game PGNs
(positions across opening / middle / endgame), winner-side and draw moves only
(per paper §3.1), each labeled by Pikafish with best move, root cp, situation
class, and a "good move" indicator. Self-play tops the JSONL up to the target size
when the PGN corpus runs short.

See [`../../docs/XIANGQI_R1_METRICS.md`](../../docs/XIANGQI_R1_METRICS.md) for the metrics
each row enables and benchmark bands.

## Files

- `raw/xqdb_masters_40711_UCI_games.pgn.zip` — cached source corpus (downloaded by
  `scripts/download_xiangqi_pgn.py`).
- `raw/xqdb_masters_40711_UCI_games.pgn` — extracted PGN (UCCI/ICCS notation, no
  `[Result]` tags — winners are inferred from terminal mate, otherwise treated as
  draw-equivalent so we keep both sides; use `--paper-strict-filter` on the builder
  to enforce the paper rule exactly).
- `xiangqi_sft_train.jsonl` — built shard, one row per labeled position. Default
  `--samples 50000` to match paper Stage-2 scale.
- `xiangqi_sft_eval_holdout.jsonl` — optional smaller shard for offline eval (e.g.
  separate build with `--no-selfplay-topup` / different row cap). Measured
  `eval_xiangqi_metrics.py` results for train vs this file: see
  [`../../docs/logs/2026-05-13-log-xiangqi-r1-sft-rebuild.md`](../../docs/logs/2026-05-13-log-xiangqi-r1-sft-rebuild.md) §4.

## JSONL schema

```jsonc
{
  "messages": [
    {"role": "system",    "content": "You are a Xiangqi …"},
    {"role": "user",      "content": "Source: pgn; Event: …; …\nFEN: …\nBoard: …"},
    {"role": "assistant", "content": "<think>…</think>\nSituation: Balanced\nMove: h2e2"}
  ],
  "meta": {
    "fen":  "rnbakabnr/… w - - 0 1",
    "turn": "w",                        // "w" = Red, "b" = Black
    "ply":  0,                          // 0-indexed half-move
    "phase": "opening" | "middlegame" | "endgame",
    "winner": "red" | "black" | "draw" | "unknown",
    "played_uci": "h2e2",
    "best_uci":   "h2e2",
    "root_cp_stm": 29.0,                // Pikafish raw cp for side-to-move
    "value_red_cp": 29.0,               // Red-positive cp (paper Value)
    "situation_3": "Balanced" | "Advantage_Red" | "Advantage_Black",
    "situation_5": "Balanced" | "Slight_Adv_Red" | "Clear_Adv_Red" | "Slight_Adv_Black" | "Clear_Adv_Black",
    "is_good_move": true,               // |V(after played) - V(after best)| ≤ σ_good
    "value_red_after_played_cp": 29.0,
    "value_red_after_best_cp":   29.0,
    "source": "pgn" | "selfplay",
    "headers": {"Event": "…", "Red": "…", "Black": "…"}
  }
}
```

## End-to-end build

```bash
export PIKAFISH_BIN=$(which pikafish)

# 1) Fetch + extract the PGN corpus (~8 MB compressed, cached).
uv run python scripts/download_xiangqi_pgn.py

# 2) Smoke test (~minutes; ensures Pikafish is wired correctly).
uv run python scripts/build_xiangqi_sft_dataset.py --samples 512 \
  --out data/xiangqi_sft/xiangqi_sft_smoke.jsonl

# 3) Full Stage-2-scale build (hours of Pikafish search).
uv run python scripts/build_xiangqi_sft_dataset.py --samples 50000

# 4) Train.
uv run python scripts/train_sft_xiangqi.py \
  --dataset data/xiangqi_sft/xiangqi_sft_train.jsonl \
  --output-dir checkpoints/xiangqi_sft

# 5) Evaluate (paper-aligned).
uv run python scripts/eval_xiangqi_metrics.py \
  --shard data/xiangqi_sft/xiangqi_sft_train.jsonl \
  --adapter checkpoints/xiangqi_sft --max-positions 64 --k 3

# Optional: eval on a different JSONL shard (see experiment log §4 for recorded numbers).
uv run python scripts/eval_xiangqi_metrics.py \
  --shard data/xiangqi_sft/xiangqi_sft_eval_holdout.jsonl \
  --adapter checkpoints/xiangqi_sft --max-positions 64 --k 3
```

## Builder knobs worth knowing

- `--target-move best` (default) — assistant target = engine best move (paper Stage-1 style).
- `--target-move human_good` — assistant target = the human-played move, **kept only when it is "good"** (paper Eq. 3 σ_good = 100). Smaller yield, expert-imitation flavor.
- `--paper-strict-filter` — drop wukong games that lack a `[Result]` tag and don't end at mate (paper-exact rule). Default off because the wukong corpus has no Result tags; off keeps the corpus usable.
- `--no-selfplay-topup` — disable the Pikafish self-play fallback.
- `--max-games 0` / `--max-ply 160` — PGN sampling caps; lower these for fast smoke tests.
- `--depth 12 --movetime-ms 400` — Pikafish settings. Match these to the eval and online-RL runs for apples-to-apples cp comparisons.

## Sources

- **Primary corpus:** [`maksimKorzh/wukong-xiangqi`](https://github.com/maksimKorzh/wukong-xiangqi/tree/main/xqdb/xqdb)
  — `xqdb_masters_40711_UCI_games.pgn.zip` (40,711 master games scraped from
  `wxf.ca`).
- **Parser:** [`cchess`](https://pypi.org/project/cchess/) (PyPI, `walker8088/cchess`)
  — same library used by the Xiangqi-R1 paper for PGN → FEN.
- **Self-play engine:** Pikafish (UCI Chinese chess engine), driven through the
  in-repo `pikafish_eval.PikafishEvaluator` wrapper.
