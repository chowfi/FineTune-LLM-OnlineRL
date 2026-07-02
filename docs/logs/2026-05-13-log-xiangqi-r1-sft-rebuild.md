# Experiment Log: Xiangqi-R1 SFT pipeline rebuild (PGN + self-play)

**Date:** 2026-05-13
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

The base Qwen2.5-7B-Instruct has limited exposure to Xiangqi (Chinese chess) compared to Western chess, and the existing GRPO loop (`LLM_RL_agent_FSDP_v2.py`) shows no episode-level learning over a 26 h run ([2026-05-11 log](2026-05-11-log-legal-move-sampler.md)). The previous SFT pipeline sampled positions via **random rollouts from the start position** (`scripts/build_xiangqi_strategy_sft.py`), producing unrealistic boards the model would never face in a real game.

**Hypothesis:** Replacing random-rollout positions with **real master-game PGNs**, with the paper's winner-side + draw-move filter (Xiangqi-R1 §3.1, arXiv:2507.12215), and labeling each position with Pikafish's best move + cp + situation class, will produce SFT examples that transfer to actual play. We target **50 k rows** (paper Stage-2 scale).

## 2. Configuration Changes

### Deleted (random-rollout pipeline + stale log + JSONL)

- `scripts/build_xiangqi_strategy_sft.py`
- `scripts/train_sft_xiangqi_strategy.py`
- `scripts/eval_xiangqi_metrics.py`
- `data/xiangqi_sft/strategy_train.jsonl`
- `data/xiangqi_sft/strategy_train_smoke.jsonl`
- `data/xiangqi_sft/README.md`
- `docs/logs/2026-05-12-log-strategy-sft-metrics.md`

### New files

- `scripts/download_xiangqi_pgn.py` — caches the [`maksimKorzh/wukong-xiangqi`](https://github.com/maksimKorzh/wukong-xiangqi/tree/main/xqdb/xqdb) `xqdb_masters_40711_UCI_games.pgn.zip` (40,711 master games, UCCI notation).
- `scripts/xiangqi_pgn.py` — streaming `iter_positions(path)` over the PGN with per-ply state via `cchess.ChessBoard`, winner inference from terminal mate (paper §3.1).
- `scripts/xiangqi_selfplay.py` — `iter_selfplay(eng, ...)` Pikafish-vs-Pikafish fallback with a 2–6 half-move random opening for diversity.
- `scripts/build_xiangqi_sft_dataset.py` — main builder. CLI:
  - `--samples 50000` (default, paper Stage-2 scale; smoke test with `--samples 512`)
  - `--target-move {best,human_good}` — engine best (default) vs Eq. 3 σ_good = 100 cp filter on the human-played move.
  - `--paper-strict-filter` — drop wukong games with no `[Result]` and no terminal mate (off by default because wukong PGNs lack Result tags).
  - `--depth 12 --movetime-ms 400` — Pikafish settings (kept aligned with GRPO v2 defaults).
- `scripts/train_sft_xiangqi.py` — Unsloth + TRL LoRA SFT on the new JSONL.
- `scripts/eval_xiangqi_metrics.py` — paper-aligned `legal@k`, `good@k`, `best@k`, `3-class@k`, `5-class@k` on a held-out shard.

### Updated

- `pyproject.toml`: added `cchess>=1.25` (PyPI; same library the paper uses for PGN → FEN).
- `docs/XIANGQI_R1_METRICS.md`: expanded with an "Online RL training signals" section documenting `game/mean_ally_cp_after_move_red`, `game/median_ally_cp_after_move_red`, `game/ally_cp_after_move_red_ema`, and the `metrics/ally_cp_after_ema_alpha` hyperparam (already wired in `LLM_RL_agent_FSDP_v2.py`). Added benchmark bands per metric, Pikafish-settings caveats vs paper Table 1 (depth-12 here vs depth-25 there), σ constants, and "what to expect" SFT vs SFT+GRPO target ranges.
- `data/xiangqi_sft/README.md`: rewritten to describe the new schema and end-to-end build commands.
- `docs/ARCHITECTURE.md` §3c: refreshed file references.
- `docs/AGENT_TODO.md`: marked the rebuild complete; added an active "run pipeline end-to-end" task and a backlog "per-phase ally-cp split" task.

### Not changed (at time of this log)

- `LLM_RL_agent_FSDP_v2.py` — the ally-cp metric (`game/mean_ally_cp_after_move_red` etc.) and `xiangqi_r1` reward mode already existed. **Follow-up (2026-05-14):** GRPO-from-SFT runs hit a first-step `backward` crash; see [2026-05-14 GRPO backward / adapter fix](2026-05-14-log-grpo-backward-sft-adapter.md) for the trainer-side fix (`enable_input_require_grads`, restore `train()` after the legal-move sampler, PEFT `disable_adapter()` for the reference forward).
- `pikafish_eval.py`, `xiangqi_board.py`, `xiangqi_labels.py` — kept as shared helpers (reused by both online RL and the new SFT pipeline).

## 3. Run Command

```bash
export PIKAFISH_BIN=$(which pikafish)

# 1) Fetch + extract the PGN corpus (~8 MB compressed, cached).
uv run python scripts/download_xiangqi_pgn.py

# 2) Smoke test (no GPU needed, ~minutes; sanity-checks Pikafish + cchess).
uv run python scripts/build_xiangqi_sft_dataset.py --samples 512 \
  --out data/xiangqi_sft/xiangqi_sft_smoke.jsonl

# 3) Full Stage-2-scale build (hours of Pikafish search).
uv run python scripts/build_xiangqi_sft_dataset.py --samples 50000

# 4) Train.
uv run python scripts/train_sft_xiangqi.py \
  --dataset data/xiangqi_sft/xiangqi_sft_train.jsonl \
  --output-dir checkpoints/xiangqi_sft

# 5) Evaluate.
uv run python scripts/eval_xiangqi_metrics.py \
  --shard data/xiangqi_sft/xiangqi_sft_train.jsonl \
  --adapter checkpoints/xiangqi_sft --max-positions 64 --k 3

# Optional: second JSONL shard (recorded numbers in §4).
uv run python scripts/eval_xiangqi_metrics.py \
  --shard data/xiangqi_sft/xiangqi_sft_eval_holdout.jsonl \
  --adapter checkpoints/xiangqi_sft --max-positions 64 --k 3
```

## 4. Quantitative Results

**Offline eval** (`scripts/eval_xiangqi_metrics.py`): base `Qwen/Qwen2.5-7B-Instruct` + LoRA adapter `checkpoints/xiangqi_sft`, `--max-positions 64`, `--k 3`, default Pikafish `--depth 12` / `--movetime-ms 400` in the eval script. Output is JSON with `n` = number of positions scored; `@k` = fraction of positions where **at least one** of *k* samples passed (see `docs/XIANGQI_R1_METRICS.md`).

**A) Train shard (not withheld — same file as SFT):** `--shard data/xiangqi_sft/xiangqi_sft_train.jsonl`

```json
{
  "n": 64,
  "legal@3": 0.90625,
  "good@3": 0.515625,
  "best@3": 0.140625,
  "3class@3": 0.96875,
  "5class@3": 0.640625
}
```

**B) Separate small JSONL (`xiangqi_sft_eval_holdout.jsonl`) — closer to a holdout check than (A), still same PGN pipeline / different row file:** `--shard data/xiangqi_sft/xiangqi_sft_eval_holdout.jsonl`

```json
{
  "n": 64,
  "legal@3": 0.90625,
  "good@3": 0.453125,
  "best@3": 0.1875,
  "3class@3": 0.921875,
  "5class@3": 0.640625
}
```

**Build / train:** full 50 k build + SFT completed locally (see §3 commands); Pikafish + GPU were available for the follow-up run.

**Still to verify in GRPO:** after loading the SFT adapter into `LLM_RL_agent_FSDP_v2.py` with `metrics/ally_cp_after_ema_alpha=0.2`, whether `game/ally_cp_after_move_red_ema` drifts up over episodes (online signal; not measured in this log).

## 5. Conclusion & Next Steps

- The pipeline change directly addresses the position-realism gap the previous random-rollout SFT had. PGN positions + paper-aligned labels are the right inputs.
- Offline numbers above show strong **3-class@3** and solid **legal@3** / **good@3** vs an untuned base; train-shard (A) is optimistic; use (B) or a time-split / shuffled JSONL split for stricter generalisation checks.
- **Next:** GRPO from SFT init (`checkpoint/load_adapter_path`, `reward/format_mix_mode=xiangqi_r1` optional) and watch W&B ally-cp mean / EMA. If GRPO fails on the first backward, see [2026-05-14 log](2026-05-14-log-grpo-backward-sft-adapter.md) (trainer fixes, not SFT data).
- Backlog: per-phase ally-cp split (opening / middlegame / endgame) so we can locate *where* the policy improves.
