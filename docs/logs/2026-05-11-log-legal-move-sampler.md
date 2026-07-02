# Experiment Log: Legal move sampler, gate reward, grounded reasoning

**Date:** 2026-05-11 (config + code); **run stopped:** 2026-05-12 (manual interrupt — same session below)  
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal
*What are we trying to achieve or test in this run?*

Stabilize online GRPO for Xiangqi by: (1) sampling **distinct legal** candidate moves from the **policy** (not free-form generation + regen/dedupe), (2) scoring candidates with **Pikafish** only **after** the group is fixed, (3) using **engine-first** rewards with a **format gate + small reasoning bonus**, and (4) **conditioning** rationale on a **fixed** move via a short second generate plus stricter rubric on the sampler path—reducing collapse (illegal/format drift) and reducing format reward masking engine signal.

## 2. Configuration Changes
*List any hyperparameter or code changes made before running the experiment.*

- **Script:** `LLM_RL_agent_FSDP_v2.py` (primary training entry for this iteration).
- `grpo/use_legal_move_sampler`: `True` — policy log-prob on `Move: <uci>` per legal action; sample `k` distinct moves without replacement (`temperature`, `epsilon` mix).
- `grpo/legal_move_sample_temperature`: `1.0`; `grpo/legal_move_sample_epsilon`: `0.05`.
- `game/play_best_candidate`: `False` — env move from sampled policy group, not argmax Pikafish among candidates.
- `grpo/entropy_coef_move`: `0.01` (was `0.05`); `generate/temperature`: `1.0` (was `1.2`); `generate/regen_temperature`: `1.2` (was `1.6`) for legacy free-form path.
- `grpo/legal_anchor_count`: `0` when using sampler (anchors redundant for distinct legal sampling).
- **Reward:** `reward/format_mix_mode`: `gate` — fail template / low `reasoning_quality` downscales engine reward; pass adds small `format_weight * reasoning_quality * 2` (capped). Legacy `mix` still configurable.
- `reward/format_weight`: `0.08` (reduced from `0.2` for weaker format-vs-engine competition under gate).
- `reward/grounding_quality_min`, `reward/format_gate_fail_scale`, `reward/format_soft_fail_scale`: gate thresholds (see code defaults).
- `sampler/generate_grounded_reasoning`: `True` — `format_grounded_move_prompt` + short generate; fallback template if parse fails.
- `reward/grounding_strict_when_sampler`: `True` — reasoning rubric caps unless exact UCI appears in `<think>` (with sampler evaluation path).
- **Logging / metrics:** `game/parsed_move_rate`, `game/legal_move_diversity`, `game/mean_legal_anchor_count`, W&B `train/legal_action_policy_*`, `train/sampler_grounding_wall_sec` where applicable.
- **Validation (no full train):** `python -m py_compile LLM_RL_agent_FSDP_v2.py`; repo-wide `uv run ruff check . --fix && uv run ruff format .` per `AGENTS.md` (see `pyproject.toml` `[tool.ruff]` — excludes generated caches / `wandb` / `checkpoints` / `*.ipynb`; `LLM_RL_agent_FSDP.py` ignores `E402` for the intentional PEFT patch import order).

## 3. Run Command
*What exact command was run?*

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
```

W&B recorded the same run as `python LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision` (single process, `WORLD_SIZE=1`, FSDP skipped).

## 4. Quantitative Results
*After a run, paste a few numbers (e.g. last episode or mean over last N). Episode-level keys are logged in W&B as `game/*`, `grpo/*`, `mfu/episode_*`; per-turn keys include `train/*`. `chinese_chess_episode_metrics_v2.csv` mirrors the main game + GRPO columns—use W&B for anything missing from the CSV.*

**Key metrics to record**

| What | W&B keys (primary) | CSV (if present) |
| --- | --- | --- |
| Who is winning | `game/ally_win_rate`, `game/enemy_win_rate`, `game/truncated_rate` | `outcome` |
| Game length | `game/episode_length` | `rounds` |
| Legal sampler working | `game/legal_move_rate`, `game/legal_move_diversity` | `game_legal_move_rate`, `game_legal_move_diversity` |
| Policy not collapsing on one move | `train/legal_action_policy_entropy` (per-step) | — |
| Format / gate | `game/format_compliance_rate`, `game/mean_chosen_format_reward` | `game_format_compliance_rate` |
| Engine signal on played move | `game/mean_chosen_engine_reward` | — |
| GRPO health | `grpo/loss`, `grpo/mean_kl`, `grpo/mean_reward`, `grpo/pg_clip_frac` | `grpo_loss`, `grpo_mean_kl`, `grpo_mean_reward`, `grpo_pg_clip_frac` |
| Degenerate fallback | `game/random_move_rate_episode` | `random_move_rate_episode` |

### Run outcome (stopped manually)

The training run for this experiment was **stopped manually** because metrics did **not** show **episode-level learning** (ally stayed at **0%** wins; returns and engine statistics did not trend clearly upward). **Scope:** seven **completed** episodes plus **episode 8 in progress** at interrupt (~**26.3 h** wall, `train/global_step` 703 at last W&B summary).

**Environment / provenance:** git `0253e1f5a7782316e64f5dc5ae45b37a8a2331c3`, host `fchow-gpu`, RTX 5090, Pikafish `/home/fchow/bin/pikafish` (depth 12), opponent `GreedyEnemy`.

**Per-episode summaries** (stdout `ally_return` line after each episode, episodes 1–7)

| Ep | ally_return | enemy_return | ally_win_rate (cumulative) | enemy_win_rate (cumulative) | mean_chosen_engine_reward | outcome (this ep) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.00 | 148.00 | 0.0% | 100.0% | 4.844 | enemy_win |
| 2 | 5.50 | 51.00 | 0.0% | 50.0% | 5.040 | truncated_cap |
| 3 | 5.00 | 150.00 | 0.0% | 66.7% | 4.789 | enemy_win |
| 4 | 9.00 | 149.00 | 0.0% | 75.0% | 4.387 | enemy_win |
| 5 | 12.00 | 50.00 | 0.0% | 60.0% | 5.063 | truncated_cap |
| 6 | 4.50 | 51.00 | 0.0% | 50.0% | 4.963 | truncated_cap |
| 7 | 8.50 | 149.00 | 0.0% | 57.1% | 4.566 | enemy_win |

**Observations**

- **Ally wins:** `ally_win_rate` stayed **0%** on every completed-episode summary line.
- **Returns:** `ally_return` bounced ~4.5–12 with **no clear upward trend** (not treated as learning signal).
- **Sampler / format:** `legal_rate`, `parsed_rate`, `legal_diversity` **1.0** throughout; `format_rate` ~0.997–1.0; `random_move_rate_episode` **0** — failure mode was **not** illegal-move collapse or random fallback.
- **W&B at stop** (`wandb/latest-run/files/wandb-summary.json`): `episode` 8, `_runtime` ~94 550 s, `game/ally_win_rate` **0.0**, `game/enemy_win_rate` ~57.1%, `game/truncated_rate` ~42.9%, `grpo/mean_reward` ~5.59, `grpo/mean_kl` ~0.0043, `grpo/loss` ~2.09e-4; `grpo/batch_reward_std` **NaN** (worth verifying — degenerate batch or logging).

**Artifacts:** `wandb/latest-run/files/output.log`; `wandb/latest-run/files/wandb-summary.json`; episode CSV if synced: `chinese_chess_episode_metrics_v2.csv` (not opened in this note per repo guidance on heavy files).

**Full metric list:** `episode_stats`, `csv_row` / `_EPISODE_METRICS_FIELDNAMES`, and per-turn `step_payload` in `LLM_RL_agent_FSDP_v2.py`.

**Lint / format (repository procedure per `AGENTS.md`):** from repo root (passes with current `pyproject.toml`):

```bash
uv run ruff check . --fix && uv run ruff format .
```

## 5. Conclusion & Next Steps
*Did it work? What should the next agent or run do?*

- **Conclusion:** Implementation in `LLM_RL_agent_FSDP_v2.py` behaved as intended on **legality and format** for this run, but **episode-level skill** (wins, returns, clear engine-quality trend) **did not improve**; stopping was reasonable without a new hypothesis.
- **Next steps:** (1) Inspect **GRPO advantage variance** / `grpo/batch_reward_std` (NaN at final summary — verify degenerate batch vs logging). (2) Check **reward spread** among legal candidates and whether **gate + engine** scaling washes out differences. (3) Consider **weaker opponent**, **curriculum**, **`play_best_candidate` ablation**, or **KL/LR** schedule after reviewing `grpo/*` in W&B. (4) Keep watching `train/legal_action_policy_entropy` and format rates. (5) Optional: low-frequency “score all legal moves with Pikafish” diagnostic. (6) Optional: sampler + gate smoke test (`docs/AGENT_TODO.md` backlog).
