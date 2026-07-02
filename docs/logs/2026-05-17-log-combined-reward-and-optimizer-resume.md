# Experiment Log: Combined gate + R_best reward & optimizer-state resume

**Date:** 2026-05-17
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Two independent improvements landed before the next RL restart:

1. **Reward shape (Xiangqi-R1 §3.4 on top of `gate`).** Through episode 18
   the run showed mechanical learning (`grpo/mean_kl_move` non-zero,
   `policy_entropy_move` non-zero, `grad_norm_pre_clip ~0.8-1.4` vs
   `max_grad_norm=0.5`) and the first **engine-alignment signal** in the
   project (Ep 17: `chosen_is_engine_argmax_in_group_rate = 31%`,
   `mean_chosen_engine_rank_in_group = 5.2`; Ep 18: 52% / 4.7). But
   `ally_return` / `enemy_return` are still 0/~50, `cp_saturation_truncation_rate`
   is climbing (33% -> 56% across Ep12-18) and `ally_cp_after_red_ema`
   is flat at ~-2000.
   Diagnosis: the smooth tanh(cp_delta) reward (`gate` mode) gives a small
   gradient on every move's *degree* of cp advantage but no sharp signal on
   the *identity* of Pikafish's preferred move. The Xiangqi-R1 paper's
   `R_move = r_legal + r_good + r_best` is the **identity** signal but,
   used alone, collapses advantage variance on most positions (all 32
   candidates score the same `{0,1,1,1}` outcome).
   **Hypothesis:** adding the discrete tactical bonus *on top of* the
   existing dense `gate` reward gets us both behaviors -- dense gradient
   everywhere + a sharp pull toward Pikafish's bestmove when it lands in
   the candidate group.

2. **Optimizer state preservation across restarts.** Previously
   `save_lora_checkpoint` saved only the LoRA `state_dict`. Adam's `m`/`v`
   moments live on the optimizer and are *not* serialized; every
   `--resume-from` builds a fresh `AdamW8bit` and starts cold (`m=v=0`).
   That accounts for the ~50-200 spiky steps observed right after the
   Ep 11 resume (`grpo/mean_kl_move = 3.66` on the first scored step of
   Ep 13). **Hypothesis:** saving + restoring Adam moments removes that
   transient and lets resumed runs start at the per-parameter step-size
   scaling the previous run was already at.

## 2. Configuration Changes

### A. Combined reward (paper §3.4 on top of `gate`)

- New `evaluate_candidate_response` params:
  `combine_gate_with_r_best: bool`, `position_best_uci_engine: Optional[str]`,
  `position_vb_best: Optional[float]`, `tactical_weight: float`,
  `sigma_good_cp: float`.
- `CandidateEval` gains `r_best: float` and `r_good: float` (per-candidate
  indicators; both 0.0 when combined reward is disabled or position-best
  precomputation failed).
- In the `gate` branch, after the existing `reward = engine + bonus`
  computation, when the format gate passed and we have a Pikafish bestmove:

  ```
  if engine_move == position_best_uci_engine:
      r_best = 1.0; r_good = 1.0
  elif |vp_played - position_vb_best| <= sigma_good_cp:
      r_good = 1.0
  reward = min(reward + tactical_weight * (r_good + r_best), 13.0)
  ```

  `vp_played` reuses `-cp_after_raw` so **no extra Pikafish call per
  candidate**.
- `XiangqiAgent.__init__` precomputes `position_best_uci_engine` and
  `position_vb_best` **once per turn** before scoring the 32 candidates,
  reusing the already-cached `bestmove_root_cached` call that the
  engine-best comparison metric also reads. Extra Pikafish cost per
  turn: 1 `bestmove` (cached, ~free on the second read) + 1 `evaluate_cp`
  of the best-move position.
- New hyperparams:
  - `reward/combine_gate_with_r_best`: **True** (was: not present; default
    feature-off behavior)
  - `reward/tactical_weight`: **1.5**
  - `reward/sigma_good_cp`: **100.0** (`SIGMA_GOOD` from `xiangqi_labels`)
- New per-turn metrics (`game/...`):
  `mean_r_best_in_group`, `mean_r_good_in_group`, `r_best_in_group_any`,
  `chosen_r_best`, `chosen_r_good`.
- New episode aggregates (CSV + W&B):
  `game_mean_r_best_in_group_rate`, `game_mean_r_good_in_group_rate`,
  `game_chosen_r_best_rate`, `game_chosen_r_good_rate`.
  `ensure_episode_metrics_csv_schema` automatically upgrades the existing
  CSV header in place (4 new columns; backup written to `*.bak`).

### B. Optimizer-state resume

- `save_lora_checkpoint` gains optional `optimizer` and `global_train_step`
  args. When provided, rank 0 additionally writes `optimizer.pt` next to
  the adapter directory containing `{"optimizer": state_dict,
  "global_train_step": int}`. All three save sites (per-episode,
  final-normal, interrupted-from-except) pass these in.
- After `grpo_trainer = GRPOTrainerOnline(...)` is constructed and *before*
  the training loop, when an `--resume-from` adapter dir is set, the script
  looks for `optimizer.pt` and calls `grpo_trainer.optimizer.load_state_dict`
  on it. The resumed `global_train_step` seeds `global_train_steps` so the
  W&B step counter and MFU accounting also continue.
- Failures (missing file, mismatched param groups, bnb 8-bit corner cases)
  are caught with a `[checkpoint] WARN` and the run proceeds with cold
  Adam moments -- no regression vs the previous behavior.

## 3. Run Command

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish

uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_18 \
  --start-episode 19
```

The Ep 18 checkpoint was saved before this patch, so it has no
`optimizer.pt`. The first run after this commit will print
`[checkpoint] No optimizer.pt at ...; Adam moments will start cold this run`
once and then write `optimizer.pt` at Ep 19 onward. After that, future
restarts from any later checkpoint will load Adam moments cleanly.

## 4. Quantitative Results

Pre-restart baseline (Ep 17-18, before this patch):

- `game/chosen_is_engine_argmax_in_group_rate`: 31.0% (Ep17) -> 52.3% (Ep18)
- `game/chosen_is_engine_best_overall_rate`: 13.8% -> 15.9%
- `game/mean_chosen_engine_rank_in_group`: 5.21 -> 4.73
- `game/ally_return`: 10.0 -> 5.5; **enemy_return**: 43 -> 51
- `game/cp_saturation_truncation_rate`: 52.9% -> 55.6%
- `game/ally_cp_after_red_ema`: -2031 -> -2145

Watch from Ep 19 onward (this run):

- **Reward-shape signal:** `game/mean_r_best_in_group_rate` (should jump
  off ~0 immediately) and `game/chosen_r_best_rate` (proxy for "how often
  did GRPO see a tactical bonus on the played move"). If
  `mean_r_best_in_group_rate` stays around 3-5% per turn we expect the
  total reward signal to be ~5-15% sharper, with `grpo/batch_reward_std`
  rising correspondingly on tactical positions.
- **Optimizer-resume:** Ep 19's first few `grpo/mean_kl_move` and
  `grpo/grad_norm_pre_clip` values **once you restart from a checkpoint
  saved by this version** should be calmer than the Ep 13 first-step
  spike (`mean_kl_move = 3.66` on the cold-optimizer restart).
- **Existing primary metric:** `game/chosen_is_engine_argmax_in_group_rate`
  should continue trending up from the 30-50% band; the new tactical
  bonus should also pull `chosen_is_engine_best_overall_rate` up off
  the ~15% ceiling because `r_best` directly rewards matching the
  deep-search bestmove (which the previous tanh reward did not).

## 5. Conclusion & Next Steps

Implementation only; no quantitative comparison yet. Open follow-ups:

- After ~10 episodes of the combined reward, compare
  `chosen_is_engine_best_overall_rate` and `mean_chosen_engine_rank_in_group`
  against the Ep 17-18 baseline. If `mean_r_best_in_group_rate` is
  high (>5%) but `chosen_is_engine_best_overall_rate` doesn't move,
  raise `reward/tactical_weight` from 1.5 -> 3.0.
- Optimizer save adds ~LoRA-sized bytes to each checkpoint (Adam moments
  are 2x the trainable-param count). For LoRA-only training that's ~tens
  of MB per `ep_*` directory; acceptable. If disk pressure becomes an
  issue, prune to keeping `optimizer.pt` only on every 3rd checkpoint.
- If `R_analysis` becomes interesting later (paper Eq. 6), the
  `xiangqi_r1` reward branch already implements it; switching to
  `reward/format_mix_mode = "xiangqi_r1"` is one config change away.

## 6. Curriculum (2026-05-17 follow-up)

Bundled into the same restart so the reward-shape + optimizer-resume
changes can be A/B'd against a less-hostile environment:

- **Enemy ε-random:** `GreedyEnemyAgent.move(env, epsilon=...)` now picks
  a uniformly random legal action with probability `epsilon`. New
  hyperparams: `enemy/epsilon_start = 0.5`, `enemy/epsilon_end = 0.0`,
  `enemy/epsilon_anneal_episodes = 25`, `enemy/epsilon_anchor_episode = None`
  (defaults to `checkpoint/start_episode` at launch so resumed runs start
  at "stage 0"). New helper `_current_enemy_epsilon` computes a linear
  decay from start to end across the anneal window. Per-episode value is
  printed in the `[Ep N] Opponent:` line and logged to W&B / CSV as
  `game/enemy_epsilon_current` / `game_enemy_epsilon_current`.
- **Truncation loosened:** `game/cp_saturation_threshold` 4000 -> **6000**,
  `game/cp_saturation_consecutive` 3 -> **5**. Reasoning: at the 4000 / 3
  setting, `cp_saturation_truncation_rate` was climbing 33% -> 56% across
  Ep 12-18 and killing half the episodes before the policy could see a
  full game. Wider window gives the new tactical bonus + weaker early
  opponent room to actually produce comeback turns.

These two are paired on purpose: with the easier opponent, fewer games
spiral into deep cp deficits at all, and the looser truncation means the
games that *do* survive longer can resolve naturally (terminal or full
truncation_cap). If `game/cp_saturation_truncation_rate` drops below
~25% over the next 5-10 episodes, the curriculum is doing what we want.
