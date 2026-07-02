# Experiment Log: Terminal-win GRPO update

**Date:** 2026-05-25 (analysis updated 2026-05-26)
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal

Win episodes were often recording `grpo_mean_kl=0.0` because the normal
candidate GRPO update runs before `env.step`, so it cannot see the environment's
`ally_reward == 100` terminal win reward. When Pikafish also fails to evaluate
the terminal board, all candidates can collapse to the neutral `5.5` engine
reward and trigger the low-std skip gate.

Goal: add a post-step GRPO update on the same sampled candidate group when the
ally receives the terminal win reward, so winning moves get a learning signal
even when pre-step GRPO was skipped or engine-blind.

## 2. Configuration Changes

- Added `reward/terminal_win_grpo_reward=10.0`.
- Added a post-step terminal-win update path in `LLM_RL_agent_FSDP_v2.py`.
- `TurnResult` now carries the sampled query/response/reward group and chosen
  candidate index for one immediate terminal update.
- When `ally_reward_terminal == 100.0`, the chosen candidate's reward is boosted
  to at least `10.0` and `train_group(...)` is called once more on the same
  sampled group.
- New method: `XiangqiAgent.train_terminal_win_update()`.
- Main loop logs: `[terminal-win] ally_reward=100.0; ran terminal GRPO update ...`

## 3. Run Command

Planned resume (adapter-only so new `lr=5e-6` applies; see
`docs/logs/2026-05-26-log-long-run-grpo-analysis-and-resume-plan.md`):

```bash
mv checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt.bak

export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_22 \
  --start-episode 23
```

## 4. Quantitative Results

### Pre-patch baseline (long run eps 5–22, `lr=1e-5`)

- **9 wins, all with `grpo_mean_kl_move=0.0` and `grpo_ppo_epochs_completed=0.0`.**
- GRPO updates occurred only on **truncated** episodes (mean `grpo_mean_kl_move=4.32`
  on 8 trunc games; spikes to 14.0 on ep 20).
- Win rate reached **50%** over eps 5–22 despite zero GRPO on wins — confirms
  the missing win-update path was a real gap, not the sole driver of wins.

### Post-patch run

Not run yet.

Expected signals:

- Ally-win episodes show **nonzero** `grpo_mean_kl_move` when terminal update runs.
- Stdout marker on wins: `[terminal-win] ally_reward=100.0; ran terminal GRPO update ...`
- `terminal_win/update_applied` in per-turn train stats (if logged to wandb).

## 5. Conclusion & Next Steps

- Root cause confirmed in CSV: all 9 win episodes had zero GRPO stats; trunc
  episodes carried the full update load including high-KL bursts.
- Terminal-win update is implemented but **not yet validated** in a live run.
- Next run should use **adapter-only resume** from `ep_22` (cold optimizer) so
  terminal-win updates stack on smoother `5e-6` / alignment defaults rather than
  hot Adam state from `1e-5`.
- If terminal updates cause KL spikes, lower `reward/terminal_win_grpo_reward`
  (e.g. `7.0`) before disabling the path.
