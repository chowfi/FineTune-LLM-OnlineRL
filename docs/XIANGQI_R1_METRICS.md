# Xiangqi metrics and benchmarks (Xiangqi-R1–aligned)

Reference: [Xiangqi-R1 (Chen et al., arXiv:2507.12215)](https://arxiv.org/html/2507.12215v1), especially §3.1 (data + labels), §3.4 (rewards), §4.2 (evaluation metrics), and Table 1.

This document is the **single source of truth** for which numbers to track and how to read them. It is referenced from `docs/ARCHITECTURE.md` and from the SFT scripts under `scripts/`.

## TL;DR — what to look at first

1. **`game/mean_ally_cp_after_move_red`** (W&B, `LLM_RL_agent_FSDP_v2.py`): trend after each ally move. If this rises across episodes, Red is genuinely getting better at picking moves — this is the **primary "is it learning" signal**.
2. **`good@1`** (offline, `scripts/eval_xiangqi_metrics.py`): how often the model picks a Pikafish-good move on held-out positions. SFT alone should push this from ~0.004 (base 7B) to ~0.35 (paper SFT-Stage2); GRPO on top to ~0.46.
3. **`3-class@1`**: situation comprehension (Balanced / Advantage_Red / Advantage_Black). Independent of move quality — diagnoses whether the model is "reading" the board.

## Online RL training signals (`LLM_RL_agent_FSDP_v2.py`)

These metrics are emitted per episode to both W&B and `chinese_chess_episode_metrics_v2.csv`. None require new code — they're already implemented; this section documents how to read them.

| W&B key | What it measures | Why it matters |
|---------|------------------|----------------|
| `game/mean_ally_cp_after_move_red` | Pikafish **Red-positive** centipawn score *after every ally move*, averaged across the episode. Higher = Red improved its position. | Direct "is the policy learning to pick better moves" signal. Independent of episodic win/loss noise. Should trend up across runs. |
| `game/median_ally_cp_after_move_red` | Median version of the above. | Robust to single-move spikes from forced-mate searches (which Pikafish reports as ±10000 cp); use this when the mean is unstable. |
| `game/ally_cp_after_move_red_ema` | Exponential moving average of the per-episode mean, controlled by hyperparam `metrics/ally_cp_after_ema_alpha` (default `0.0` = off). | Smoothed learning curve. **Recommended: set `metrics/ally_cp_after_ema_alpha=0.2`** so this metric actually populates. |
| `game_mean_chosen_cp_delta_raw` / `_clipped` | Per-step Pikafish reward signal used by GRPO. | Sanity-check that the reward function still produces gradient. |
| `train/chosen_cp_delta`, `train/chosen_cp_before` | Pre/post move cp from the chosen-move accounting. | Cross-check against the ally-cp metric above. |
| `game_legal_move_rate` / `game_parsed_move_rate` / `game_format_compliance_rate` | Parsing / legality / template health. | If any of these drops below ~0.9 the chosen-cp metric becomes meaningless (we fell back to random moves). |

**How to interpret the ally-cp trend (initial behaviour):** At t=0 with a base 7B that has not seen Xiangqi, expect Red to play near-random moves; Pikafish often responds to give Red a position around 0 to −300 cp on Red's perspective. As training progresses, this should drift up toward 0 and above (positive = Red is ahead). A run that stays flat for >10 k episodes (like the 2026-05-11 legal-move-sampler run logged at [`docs/logs/2026-05-11-log-legal-move-sampler.md`](logs/2026-05-11-log-legal-move-sampler.md)) is a strong signal that the policy is not learning — the SFT-first pipeline in this repo (see "Strategy SFT pipeline" below) is the recommended counter-measure.

### Reward modes (paper §3.4)

`reward/format_mix_mode` controls how the per-step reward is composed:

- `mix` (default historical): legacy weighted mix of move quality + format compliance.
- `gate`: format compliance acts as a gate; non-compliant → 0; compliant → Pikafish cp-based reward + small format bonus.
- **`xiangqi_r1`** (paper §3.4 implementation): discrete `R_move + R_analysis + R_format`, where `R_move = r_legal + r_good + r_best`, `R_analysis = 1{predicted Situation == Pikafish Situation}`, `R_format = 1{think/answer blocks well-formed}`. **Requires the completion to include a parseable `Situation:` line**, which the SFT data does emit.

## Offline move-suggestion metrics (`scripts/eval_xiangqi_metrics.py`)

Used on any JSONL emitting the same `meta` schema (e.g. `data/xiangqi_sft/xiangqi_sft_train.jsonl` or a separate eval shard such as `data/xiangqi_sft/xiangqi_sft_eval_holdout.jsonl`). Prefer a shard **not** used in SFT for unbiased `good@k` / `best@k`.

| Metric | Definition | Why it matters |
|--------|------------|----------------|
| **legal@k** | In *k* independent samples, at least one is a rule-legal move. | Without legality, strategy is meaningless. Human players are ~100% here. |
| **good@k** | At least one sample is a **good** move: engine value after the move is within **σ_good = 100** cp of the value after the engine's **best** move (paper Eqs. 2–3). | Captures **practical strength** short of perfect play. |
| **best@k** | At least one sample equals the engine's **best** move. | Hardest move-matching target; stays well below 1.0 even for strong specialised models. |
| **3-class@k** | Predicted Situation (Balanced / Advantage_Red / Advantage_Black) matches the Pikafish root evaluation with **σ_s = 100** (paper Eq. 4 collapsed to three classes). | Tests whether the model **reads the board** in engine space, not only produces moves. |
| **5-class@k** | Same with **slight** vs **clear** advantage split at **σ_l = 800**. | Finer diagnosis (opening vs decisive advantage). The default SFT only emits 3-class strings, so this is partial-credit. |

**Paper note:** Situation metrics are evaluated **only when a legal move exists** in the sample being judged (paper §4.2).

**Recorded offline eval (2026-05-14, local):** `Qwen/Qwen2.5-7B-Instruct` + adapter `checkpoints/xiangqi_sft`, `n=64`, `k=3`, `eval_xiangqi_metrics.py` defaults. **Train shard** (`xiangqi_sft_train.jsonl`): `legal@3=0.90625`, `good@3=0.515625`, `best@3=0.140625`, `3class@3=0.96875`, `5class@3=0.640625`. **Eval JSONL** (`xiangqi_sft_eval_holdout.jsonl`): `legal@3=0.90625`, `good@3=0.453125`, `best@3=0.1875`, `3class@3=0.921875`, `5class@3=0.640625`. Full commands and caveats: [`docs/logs/2026-05-13-log-xiangqi-r1-sft-rebuild.md`](logs/2026-05-13-log-xiangqi-r1-sft-rebuild.md) §4.

## Benchmark bands (Xiangqi-R1, 7B-scale)

These are paper Table 1 numbers (depth-25 Pikafish, k=1 / k=3 reported). They should be read as **bands**, not magic thresholds.

| Model (paper) | legal@1 | good@1 | best@1 | 3-class@1 | 5-class@1 |
|----------------|---------|--------|---------|-----------|-----------|
| Qwen2.5-7B-Instruct (untuned) | ~0.012 | ~0.004 | ~0.001 | ~0.003 | ~0.001 |
| SFT-Stage1-7B | ~0.922 | ~0.449 | ~0.194 | — | — |
| SFT-Stage2-7B | ~0.864 | ~0.353 | ~0.119 | ~0.431 | ~0.389 |
| Xiangqi-R1-7B (full pipeline) | ~0.952 | ~0.461 | ~0.181 | ~0.629 | ~0.610 |

**Pikafish-settings caveat (read this).** This repo's defaults are **`depth ≈ 12` and `movetime_ms ≈ 400–500`** (the GRPO loop can't wait for depth-25 every step). Paper Table 1 uses depth-25 in the engine. With shallower search:

- `best@k` is the most sensitive — shallow search may pick a different "best", inflating mismatches.
- `good@k` is more stable because the σ_good = 100 cp window is large vs typical depth-12 noise.
- `3-class@k` / `5-class@k` are very stable; the cp window for class boundaries (±100, ±800) dwarfs depth-induced jitter.

Bump `--depth` / `--movetime-ms` in `eval_xiangqi_metrics.py` when directly comparing to Table 1.

**What to expect for *this* repo (rough targets):**

- After SFT alone on 50 k rows (Stage-2-style): `good@1 ≈ 0.30–0.40`, `best@1 ≈ 0.10–0.15`, `3-class@1 ≈ 0.40–0.55`.
- After SFT + GRPO with `reward/format_mix_mode=xiangqi_r1`: `good@1 ≈ 0.40–0.50`, `best@1 ≈ 0.15–0.20`, `3-class@1 ≈ 0.55–0.65`.
- **Online RL signal**: `game/mean_ally_cp_after_move_red` rising from initially negative (Red loses cp every move) toward 0 and above is the *fastest* learning indicator — visible within a few hundred episodes if the SFT base is decent.

## Strategy SFT pipeline (Xiangqi-R1 §3.1)

| Stage | Script | Role |
|-------|--------|------|
| Fetch corpus | [`scripts/download_xiangqi_pgn.py`](../scripts/download_xiangqi_pgn.py) | Cache the wukong-xiangqi 40,711-game UCCI-notation PGN under `data/xiangqi_sft/raw/`. |
| Build SFT JSONL | [`scripts/build_xiangqi_sft_dataset.py`](../scripts/build_xiangqi_sft_dataset.py) | Walk PGN → winner+draw filter → Pikafish best+cp+good labels → JSONL. Self-play top-up if PGN doesn't reach `--samples`. |
| Train | [`scripts/train_sft_xiangqi.py`](../scripts/train_sft_xiangqi.py) | Unsloth + TRL LoRA SFT on `messages`. |
| Evaluate | [`scripts/eval_xiangqi_metrics.py`](../scripts/eval_xiangqi_metrics.py) | legal@k / good@k / best@k / 3-class@k / 5-class@k on a held-out JSONL shard. |
| Data | `data/xiangqi_sft/xiangqi_sft_train.jsonl` | One row per labeled position; `meta` has `fen`, `played_uci`, `best_uci`, `value_red_cp`, `situation_3/5`, `is_good_move`, `phase`, `ply`, `source`. |

**Position sources:**

- **PGN primary**: `maksimKorzh/wukong-xiangqi` `xqdb/xqdb/xqdb_masters_40711_UCI_games.pgn.zip` (40,711 master games from wxf.ca; moves in UCCI/ICCS, no `[Result]` tag — winner inferred from terminal mate or treated as draw-equivalent).
- **Self-play fallback**: Pikafish-vs-Pikafish with a 2–6 half-move random opening for diversity (`scripts/xiangqi_selfplay.py`).
- **Parsing library**: `cchess` (PyPI; same library the paper uses for PGN → FEN).

## σ constants (paper defaults)

Mirrored in [`xiangqi_labels.py`](../xiangqi_labels.py):

- **σ_good = 100 cp** — "good move" vs best move.
- **σ_s = 100 cp** — balanced vs advantage (root Value).
- **σ_l = 800 cp** — slight vs clear advantage (5-class only).

## Tracking these in W&B

The relevant keys are emitted by `LLM_RL_agent_FSDP_v2.py` (look up the names in the table at the top of this file). For ad-hoc analyses, `chinese_chess_episode_metrics_v2.csv` contains the same metrics in flat tabular form so you can sanity-check W&B traces with `pandas`.
