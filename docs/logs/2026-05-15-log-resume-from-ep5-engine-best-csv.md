# Experiment Log: Resume from `ep_5` + engine-best agreement in CSV

**Date:** 2026-05-15
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

The current xiangqi GRPO run (`pomdy3p6`, started 2026-05-14 23:17) shows
**no learning signal in `grpo/mean_kl_move`** and the chosen-vs-group engine
reward gap is actually *shrinking* over Ep1-4 (+0.32 → +0.09 → -0.03 → -0.05).
The May 14 doc spec added engine-best agreement metrics in code but they were
landed *after* this run started, so W&B is missing them and there is no way
to track whether GRPO is teaching the policy's argmax to align with
Pikafish's best legal move *inside* the 32-candidate group.

**Goal:** Make the *next* run (resumed from end-of-episode-5) actually log
the engine-best agreement series in both W&B and the per-episode CSV, and
preserve cumulative scoreboard / win-rate state across the restart so the
charts read continuously.

## 2. Configuration Changes

### Hyperparam defaults (`LLM_RL_agent_FSDP_v2.py`)

| Key | Old | New | Why |
|---|---|---|---|
| `checkpoint/load_adapter_path` | `checkpoints/xiangqi_sft` | `checkpoints/xiangqi_grpo_v2/ep_5` | Resume from end-of-episode-5 instead of restarting from SFT. |
| `checkpoint/start_episode` | *(new)* | `6` | 1-indexed episode the loop continues at. Must equal loaded `ep_N + 1`. |
| `metrics/clear_csv_on_start` | `True` | `False` | Append to `chinese_chess_episode_metrics_v2.csv` instead of truncating it. |

### New CLI args

```text
--resume-from PATH        # overrides checkpoint/load_adapter_path
--start-episode N         # overrides checkpoint/start_episode (1-indexed)
```

### CSV schema additions

Seven new columns in `_EPISODE_METRICS_FIELDNAMES` (and `csv_row` populates
them from the per-episode aggregates that already exist in `episode_stats`):

- `game_engine_best_known_rate` — % of ally turns where Pikafish returned a `bestmove`.
- `game_engine_best_in_group_rate` — % of those turns where Pikafish's best is also one of the 32 sampled.
- `game_chosen_is_engine_argmax_in_group_rate` — % where the played move == argmax-`engine_reward` move inside the group.
- `game_chosen_is_engine_best_overall_rate` — joint: in-group **and** picked.
- `game_mean_chosen_engine_rank_in_group` — average rank of the played move within the 32 (1 = best).
- `game_median_chosen_engine_rank_in_group` — median equivalent.
- `game_mean_chosen_minus_argmax_cp_delta` — avg cp left on the table vs in-group best (negative = blunder).

### CSV schema migration

A new rank-0 startup helper `ensure_episode_metrics_csv_schema` runs in the
resume path (when `metrics/clear_csv_on_start=False`). It:

1. Reads the existing CSV header.
2. If it doesn't already match `_EPISODE_METRICS_FIELDNAMES`, writes a single
   `.bak` and rewrites the file in place with the new header, padding prior
   rows with empty strings for the new columns.
3. Logs a one-line summary of how many columns were added.

This keeps every prior episode's per-row data intact while making the file
appendable under the new schema.

### Resume state preload

New helper `preload_run_state_from_csv(filepath, up_to_episode_exclusive)`
sums `[1, start_episode)` rows of the existing CSV to reconstruct:

- `season_ally_return`, `season_enemy_return` (stdout scoreboard).
- `ally_wins`, `enemy_wins`, `truncated_games` (W&B win-rate %).
- `cp_saturation_truncations` (W&B saturation rate).
- `lifetime_ally_turns`, `lifetime_random_fallback` (lifetime random-move rate).

Outcome strings (`ally_win` / `enemy_win` / `truncated_*`) are mapped back to
the counter buckets. Missing or malformed cells are tolerated — the
counter just misses that one row. The reconstructed counters are logged
once at startup so the user can sanity-check before training kicks in.

### Loop start point

```python
for episode in range(start_episode, episodes + 1):
```

with `start_episode = max(1, int(hyperparams["checkpoint/start_episode"]))`.
When the value is `1` (fresh run), behavior is unchanged.

### What did **NOT** change

- GRPO hyperparams (`grpo/lr`, `grpo/beta`, `grpo/max_grad_norm`,
  `grpo/ppo_epochs`, `grpo/clip_eps_low/high`) are deliberately unchanged so
  this resumed run is a **clean diagnostic baseline** for engine-best
  agreement. The optimizer-knob bump and `adaptive_cp_scale` tightening
  (recs 1 + 2 from the analysis chat) are intentionally split out to a
  follow-up log so we can isolate which lever is moving which metric.
- Action-selection rule stays `greedy`; the GRPO group composition is
  unchanged.
- Reward function is unchanged.
- Engine-best computation in `act_and_train` is unchanged — the code was
  already there from the May 14 doc; this log only adds the **CSV plumbing**
  and the **resume mechanics** required to see it in a continuous run.

## 3. Run Command

**Prerequisite:** the still-running `pomdy3p6` process must reach the end of
episode 5 so `checkpoints/xiangqi_grpo_v2/ep_5/` exists. As of writing,
heartbeat shows Ep 5 Rd 57 (≈ 43 plies + 1 train step + 1 checkpoint left).
Verify with:

```bash
ls -la checkpoints/xiangqi_grpo_v2/ep_5/
```

Once `ep_5/` is present, kill the old `pomdy3p6` process (it'll write an
`interrupted_ep5/` snapshot via the `except` path — that's fine) and launch:

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision
```

Defaults already point at `ep_5` and `start_episode=6`. If you instead want
to resume from `ep_4` (already on disk now, no waiting), override at the
CLI:

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_4 \
  --start-episode 5
```

## 4. Quantitative Targets (next run)

W&B series to watch from Ep 6 onward (none of these exist in `pomdy3p6`):

- `game/chosen_is_engine_argmax_in_group_rate` — primary learning indicator.
  Baseline is whatever Ep 6 lands at; want a positive slope over 10-20
  episodes. If flat, the optimizer-step recommendations are needed.
- `game/engine_best_in_group_rate` — sampling coverage of Pikafish's best
  move. If this stays ≤ 50 %, the GRPO group is structurally missing the
  best target and even a perfect optimizer can't learn it (push
  `legal_move_sample_temperature` up or `legal_move_sample_epsilon` up).
- `game/chosen_is_engine_best_overall_rate` — joint condition (`≤ in_group_rate`).
- `game/mean_chosen_engine_rank_in_group` — should trend toward 1.0.
- `game/mean_chosen_minus_argmax_cp_delta` — should trend toward 0 from
  negative.

Stdout sanity per ally turn:

```text
Engine-best comparison: engine_best_overall=<uci> in_group=<0|1>
  argmax_in_group=<uci> argmax_engine_reward=… argmax_cp_delta=…
  chosen=<uci> chosen_is_argmax_in_group=<0|1>
  chosen_is_engine_best_overall=<0|1> chosen_rank_in_group=<int>
  chosen_minus_argmax_cp_delta=…
```

CSV sanity (`chinese_chess_episode_metrics_v2.csv`):

- Header upgraded once at startup (look for the `[csv-schema] upgraded` line
  in stdout) and `.bak` file present.
- Rows for Ep 1-5 keep their original cell values; the seven new columns
  are empty for those rows.
- Rows for Ep 6+ populate the new columns.

Resume sanity (one-shot at startup):

```text
[resume] start_episode=6 | preloaded ally_wins=0 enemy_wins=0
  truncated=5 cp_sat_trunc=0 season_ally=22.00 season_enemy=255.00
  lifetime_ally_turns=250 lifetime_random_fallback=0
```

(Exact numbers depend on what's in CSV when resuming.)

## 5. Conclusion & Next Steps

- This change is **purely additive plumbing + resume mechanics**. No
  optimizer or reward behavior is altered, so any change in learning
  trajectory after the restart is attributable only to (a) the loaded
  adapter and (b) the freshly logged diagnostics.
- After 5-10 resumed episodes, decide whether to land the optimizer-knob
  bump (`lr 3e-6 → 1e-5`, `max_grad_norm 0.1 → 0.5`, `ppo_epochs 2 → 4`)
  and / or the `adaptive_cp_scale` cap tighten — those each need their own
  log per the AGENTS.md template.
- **Followups (backlog):**
  - Optional: skip GRPO update when `grpo/batch_reward_std < 0.3` to avoid
    the KL-regularizer pulling the policy back toward SFT on uninformative
    batches.
  - Optional: position curriculum reset when `cp_before < -8000` to spend
    training compute on positions with non-saturated reward.
