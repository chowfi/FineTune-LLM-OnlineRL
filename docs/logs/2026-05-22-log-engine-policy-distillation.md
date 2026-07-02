# Experiment Log: Engine-Policy Distillation and Smoother GRPO Updates

**Date:** 2026-05-22
**Agent/Author:** AI Coding Agent

## 1. Hypothesis / Goal
The resumed self-play run showed that the model's highest-probability legal move often did not match what Pikafish rated as a good move. GRPO was validly learning from relative rewards inside the sampled group, but it did not directly optimize the legal-move policy distribution to put its top probability on engine-good moves. The run also made episode-level `grpo_mean_kl_move` look like `0.0` for some episodes because the CSV stored the final turn's GRPO stat; if the final turn was skipped by the low reward-std gate, the whole episode appeared to have zero movement.

Goal: keep GRPO as the main online RL objective, but add a direct engine-policy distillation signal and make optimizer diagnostics reflect episode means instead of the last turn only.

## 2. Configuration Changes
- Added `grpo/engine_policy_align_coef = 0.05`.
- Added `grpo/engine_policy_align_temperature = 0.5`.
- Changed `grpo/lr` from `1e-5` to `5e-6` to reduce bursty update episodes where `grpo_mean_kl_move` exceeded 1.0.
- Changed `grpo/ppo_epochs` from `4` to `3` so the added alignment loss does not stack on top of the previous full update pressure.
- Changed `grpo/min_batch_reward_std` from `0.3` to `0.15` so fewer turns are GRPO-skipped.
- Kept `game/play_best_candidate = False`; the agent still plays its own policy-greedy legal move, not a Pikafish action oracle.

Code changes:
- `GRPOTrainerOnline.train_group(...)` now accepts `engine_policy_scores`.
- For candidate rows with successful engine evals, the trainer converts `engine_reward` into a soft target distribution and adds `KL(engine_target || policy_distribution)` over the candidate move-region log-probs.
- If `reward_std < grpo/min_batch_reward_std`, the GRPO policy-gradient term is skipped but the engine-policy alignment update still runs when at least two valid engine-scored legal candidates exist.
- Episode CSV GRPO fields now aggregate over all turns in the episode rather than storing only the final turn's stats.
- Added CSV diagnostics: `grpo_update_rate`, `grpo_skip_low_reward_std_rate`, `grpo_engine_align_loss`, `grpo_engine_align_kl`, `grpo_engine_align_entropy`, `grpo_engine_align_target_entropy`, and `grpo_engine_align_valid_count`.

## 3. Run Command
For a fresh run:

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
```

To resume from the latest pre-change checkpoint, use an explicit adapter and start episode. Example:

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_10 \
  --start-episode 11
```

Keep `metrics/clear_csv_on_start = False` for resumed runs.

## 4. Quantitative Results

### Long run before this patch (eps 5–22, `lr=1e-5`, no alignment)

Code landed 2026-05-22 but was **not exercised** in the long run; CSV alignment
columns are empty for eps 5–22.

| Metric | Value |
|--------|-------|
| Record (eps 5–22) | 9W / 1L / 8T |
| Win episodes with GRPO | **0 / 9** (`grpo_mean_kl_move=0.0` on every win) |
| Trunc episodes with GRPO | 8 / 8 active; mean `grpo_mean_kl_move=4.32` |
| High-KL trunc episodes | ep 18: 5.26; ep 20: **14.0**; ep 22: 9.40 |

Wandb `grpo/mean_kl_move` over the run: mostly low (&lt;5) for steps 0–~440, then
bursty (15–40 common, spike to ~100 around step 270). Win rate still improved
and self-play enemy synced after eps 13–15 — learning happened **despite** hot
KL, not because spikes were ideal.

Engine alignment proxies (gate-only, no distillation loss):

- Win-episode `game_chosen_is_engine_argmax_in_group_rate`: ~7–40% (mostly 12–18%).
- `game_chosen_is_engine_best_overall_rate` on wins: mostly 3–14%.
- Policy–engine agreement did not track win rate; wins came from broader play
  quality and self-play curriculum, not argmax-matching Pikafish.

### Post-patch run

Not started. Planned resume: adapter from `ep_22`, **cold optimizer** (move
`optimizer.pt` aside) so `lr=5e-6` and alignment coef `0.05` actually apply.
See `docs/logs/2026-05-26-log-long-run-grpo-analysis-and-resume-plan.md`.

Metrics to watch after launch:

- `grpo_update_rate` — alignment should update even when GRPO PG term is skipped.
- `grpo_engine_align_kl`, `grpo_engine_align_valid_count` — first live signal.
- `game_chosen_is_engine_argmax_in_group_rate` — should rise if alignment works.
- `grpo_mean_kl_move` — target calmer regime than old 1e-5 bursty trunc updates.

## 5. Conclusion & Next Steps

- Zero `grpo_mean_kl_move` on wins was confirmed across all 9 win episodes in
  the long run; terminal-win GRPO (2026-05-25 patch) addresses that separately.
- High trunc-episode KL under `1e-5` coexisted with real learning; for the next
  run, prefer **adapter-only resume + `5e-6`** over reloading hot Adam from
  `ep_22/optimizer.pt`.
- The auxiliary alignment loss directly targets policy-top legal moves matching
  engine-good moves; it has not been validated live yet.
- If alignment is weak after several episodes, raise
  `grpo/engine_policy_align_coef` (`0.05 -> 0.08`) or lower temperature
  (`0.5 -> 0.35`).
- If KL remains too large on trunc episodes, lower `grpo/lr` (`5e-6 -> 3e-6`)
  before increasing `grpo/beta`.
