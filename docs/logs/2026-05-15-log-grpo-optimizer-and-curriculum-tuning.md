# Experiment Log: GRPO optimizer + reward-shape + curriculum tuning

**Date:** 2026-05-15
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

After Ep1-5 of `pomdy3p6` and resumed Ep6 of the next run, the diagnostic
analysis (see prior chat + `docs/logs/2026-05-15-log-resume-from-ep5-engine-best-csv.md`)
established that:

- `grpo/mean_kl_move` was hovering at 1e-7 -- 1e-5 per step (i.e. effectively
  float-rounding noise). The reasoning region was drifting (`mean_kl_think
  ~ 0.5--0.8` per token), so the optimizer was active in general, just not
  applying meaningful pressure to the ``Move: <uci>`` tokens specifically.
- `chinese_chess_episode_metrics_v2.csv` Ep6 row showed
  `game_engine_best_known_rate = 0.0` (bug since fixed in the UCI-FEN patch),
  `mean_chosen_engine_rank_in_group = 6.74` (median = 1.5), and
  `mean_chosen_minus_argmax_cp_delta = -59.82` cp. The policy was leaving
  cp on the table inside its own sampled group.
- `game_ally_cp_after_move_red_ema` was drifting from -3617 (Ep1) to -5343
  (Ep4) to -5443 (Ep6) -- training distribution is biased toward saturated
  lost positions where adaptive_cp_scale collapses the reward range.

**Goal:** unblock the move-token policy gradient so `grpo/mean_kl_move`
moves into the 1e-3 -- 1e-1 range per step and the engine-best-in-group
agreement rates start trending up over 5-15 episodes. Three independent
levers, each tied to a specific failure mode in the analysis.

## 2. Configuration Changes

### A. Optimizer knobs unlock the move-token update (`LLM_RL_agent_FSDP_v2.py`)

| Hyperparam               | Old   | New   | Mechanism |
|--------------------------|-------|-------|-----------|
| `grpo/lr`                | 3e-6  | 1e-5  | ~3.3x larger AdamW step per gradient. Near-zero ``KL_move`` should track ``lr`` linearly. |
| `grpo/max_grad_norm`     | 0.1   | 0.5   | The previous 0.1 cap was almost certainly clipping every step (typical 7B LoRA + GRPO grad norms are O(1)). Lifting the cap lets the actual gradient through to AdamW. |
| `grpo/ppo_epochs`        | 2     | 4     | ``ratio_mean`` was at 1.001 and ``pg_clip_frac`` at 0.3% after 2 epochs -- huge unused PPO headroom. Two more inner epochs per group exploit it. |
| `grpo/beta`              | 0.05  | 0.01  | The KL-to-ref term was the dominant loss component on uninformative batches; lowering it lets advantages move the policy further before the SFT-anchor pulls back. |

### B. Reward-spread fix in saturated positions (`adaptive_cp_scale`)

- New module-level constants `_CP_SCALE_WIDEN_CAP_MULT = 1.2` (was 2.0) and
  `_CP_SCALE_WIDEN_SLOPE = 0.2` (unchanged).
- Updated docstring documents the tightening rationale + worked example.

What it does: for `cp_before = -8000`, `cp_delta = -100` the candidate
reward is now `5.5 + 4.5 * tanh(-100/300) ~= 4.07` instead of `~4.6` at
the old scale=500. Across a 32-candidate group with `cp_delta in [-500, 0]`
the reward span widens from `[2.0, 5.5]` to `[1.0, 5.5]`, doubling
`grpo/batch_reward_std` and the advantage gradient on the same cp data.

### C. Skip optimizer step on uninformative batches (GRPO trainer)

New hyperparam `grpo/min_batch_reward_std: 0.3`.

Mechanism: in `GRPOTrainerOnline.train_group`, after computing
`reward_std = rewards_t.std()`, if `reward_std < min_batch_reward_std`
return a stats dict with `grpo/samples_skipped = N`,
`grpo/skipped_low_reward_std = 1.0`, and zero/identity values for the
gradient-driven metrics. The env still rolls forward (the GRPO group is
already collected); we just don't call `optimizer.step()`. This prevents
`beta * KL` from dominating and pulling the policy back toward SFT on
near-zero-advantage batches. Stdout marker: `[GRPO] Skipping optimizer
step: reward_std=...`.

The new `grpo/samples_skipped` and `grpo/skipped_low_reward_std` keys are
*always* present in the returned stats (skip path and normal path both
populate them) so the per-step W&B time series stays well-defined.

### D. Curriculum via early-truncation in hopeless positions

| Hyperparam                          | Old     | New    | Reason |
|-------------------------------------|---------|--------|--------|
| `game/cp_saturation_threshold`      | 8000.0  | 4000.0 | At |cp_before| >= 4000 the engine's cp range across candidates is already collapsing toward the same value. |
| `game/cp_saturation_consecutive`    | 0       | 3      | Truncate the episode after 3 consecutive ally turns at the saturation threshold (instead of letting the episode pad out to round 100). |

The next `env.reset()` then starts a fresh game whose opening / early
middlegame positions have moderate `cp_before`, where
`adaptive_cp_scale` returns the base 250 -- the regime with maximum
`reward_std` per batch and the strongest GRPO learning signal.

### E. Pre-clip grad norm diagnostic

New stat `grpo/grad_norm_pre_clip` exposes the **largest L2 gradient norm
across PPO epochs in this step, before** `clip_grad_norm_` rescales it.
`torch.nn.utils.clip_grad_norm_` already returns this value; we just track
the running max across inner epochs and surface it in the stats dict.

Why: confirms whether the previous 0.1 cap was throttling. After the bump
to 0.5, we should see `grpo/grad_norm_pre_clip` regularly in the 0.1 -- 1.0
range (i.e. mostly *not* clipped). If it's still pinned at the cap, then
either grads are genuinely large (need to lower LR) or there's a
gradient-explosion path we haven't addressed.

### What did **NOT** change

- Action-selection rule stays `greedy` (analysis #6: not a learning lever).
- GRPO group composition, regen rules, dedupe, anchors all unchanged.
- Reward function logic outside of `adaptive_cp_scale` is unchanged
  (`evaluate_candidate_response`, format-weight annealing, grounding
  strict-mode, etc. all preserved).
- Engine-best metrics from the previous patch are unchanged; the UCI-FEN
  fix from the same day is preserved.

## 3. Run Command

Same as the previous resumed run -- defaults already point at the Ep5
checkpoint and `start_episode = 6`. After killing the current process
(it'll save an `interrupted_ep<N>/` snapshot via the `except` path),
relaunch:

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision
```

To resume from a different checkpoint or episode, use the CLI overrides
from the previous log: `--resume-from PATH` and `--start-episode N`.

## 4. Quantitative Targets (per analysis acceptance criteria)

Watch within ~50 GRPO steps of the changes:

- **Primary:** `grpo/mean_kl_move` jumps from ~1e-7 to >= 1e-3 per step.
  If it stays at 1e-7 the optimizer is still throttled and we need a
  follow-up (likely `beta` even lower or `grad_norm` even higher).
- `grpo/grad_norm_pre_clip` lives mostly in `[0.05, 0.5]` (i.e. the new
  cap rarely binds). If it's pinned at 0.5 we've just moved the clip
  location; lower `grpo/lr` or raise `grpo/max_grad_norm` again.
- `grpo/ratio_mean` drifts further from 1.0 (range ~[0.95, 1.05]) and
  `grpo/pg_clip_frac` rises from 0.3% to 5--15%.
- `grpo/batch_reward_std` rises ~1.5--2x on average (rec D alone).
- `grpo/samples_skipped` is non-zero on saturated turns -- this is the
  expected behavior, not a bug. We *want* those steps gated.

Watch over 5--15 episodes:

- `game/chosen_is_engine_argmax_in_group_rate` trends up (now correctly
  logged thanks to the UCI-FEN fix).
- `game/mean_chosen_engine_rank_in_group` trends down toward 1.
- `game/mean_chosen_minus_argmax_cp_delta` trends up toward 0.
- `game/engine_best_known_rate` should land near 100% (verifies UCI-FEN
  fix is taking effect under the new run).
- `game_ally_cp_after_move_red_ema` starts moving in the positive
  direction now that hopeless tails are truncated.

`ally_return` will lag -- game-outcome improvement is a downstream effect
of per-turn policy improvement and needs many episodes of compounding.

## 5. Conclusion & Next Steps

- This is the bundled implementation of the optimizer / reward / curriculum
  recommendations from the analysis chat. Resume mechanics and engine-best
  diagnostics (separate logs from earlier today) are unchanged.
- After the next 5--10 resumed episodes, decide:
  - If KL_move moves but the engine-agreement rates stay flat -> the
    advantages are pointing the wrong way (re-examine reward shaping).
  - If KL_move is still pinned near zero -> further loosen `beta`
    (e.g. 0.01 -> 0.005) or further lift `max_grad_norm` (0.5 -> 1.0).
  - If `samples_skipped` is consuming > 70% of steps -> `min_batch_reward_std`
    is too aggressive; lower to 0.2 or 0.15.
- **Followups (backlog):**
  - Aggregate `grpo/grad_norm_pre_clip` and skipped-step rate into the
    per-episode CSV.
  - Position curriculum from a curated middlegame FEN set (instead of just
    truncating to the env's default starting position).
  - Consider further annealing of `format_weight` once compliance is at
    99%+ for several episodes.
