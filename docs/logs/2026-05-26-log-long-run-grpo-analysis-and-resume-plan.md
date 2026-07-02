# Experiment Log: Long-run GRPO analysis and ep 23 resume plan

**Date:** 2026-05-26
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal

Summarize the longest recent RL run (episodes 5–22, resumed from `ep_4` with
gate-only reward and `grpo/lr=1e-5`) before restarting with the May 22–25 code
changes (engine-policy alignment, smoother GRPO defaults, terminal-win GRPO
update). Decide whether to resume adapter + optimizer or adapter-only.

## 2. Configuration Changes

No new code in this session. Planned **next-run** settings (already in
`LLM_RL_agent_FSDP_v2.py` defaults, not yet validated in a live run):

- `grpo/lr`: `5e-6` (was `1e-5` in the long run)
- `grpo/ppo_epochs`: `3` (was `4`)
- `grpo/min_batch_reward_std`: `0.15` (was `0.3`)
- `grpo/engine_policy_align_coef`: `0.05` (not active in the long run)
- `reward/terminal_win_grpo_reward`: `10.0` (new post-step win update)

**Resume decision:** load **adapter only** from `checkpoints/xiangqi_grpo_v2/ep_22`;
**do not** load `ep_22/optimizer.pt` so the new LR and cold Adam actually apply.
Rename or move `optimizer.pt` aside before launch.

## 3. Run Command

```bash
mv checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt.bak

export PIKAFISH_BIN=/home/fchow/bin/pikafish
cd /home/fchow/Documents/FineTune-LLM-OnlineRL
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_22 \
  --start-episode 23
```

Self-play enemy loads automatically from `checkpoints/self_play_enemy` (synced
after the ep 13–15 win streak). CSV appends with `metrics/clear_csv_on_start=False`.

## 4. Quantitative Results

### Long run (episodes 5–22, `chinese_chess_episode_metrics_v2.csv`)

| Metric | Value |
|--------|-------|
| Record | **9W / 1L / 8T** (50% win rate over 18 episodes) |
| Last checkpoint | `ep_22` (`global_train_step=926`, optimizer saved) |
| Self-play enemy sync | After ep 15 (3 consecutive wins eps 13–15) |

**GRPO on wins vs truncations (pre terminal-win patch):**

| Outcome | n | Mean `grpo_mean_kl_move` | Mean `grpo_ppo_epochs_completed` |
|---------|---|--------------------------|-------------------------------------|
| `ally_win` | 9 | **0.0** | **0.0** |
| `truncated_cap` | 8 | **4.32** | **3.0** (full 4-epoch updates on active trunc eps) |

All 9 win episodes had zero GRPO episode stats because the normal update runs
**before** `env.step()` and cannot see the terminal `+100` reward; flat Pikafish
groups on the final turn also hit the `min_batch_reward_std=0.3` skip gate.

**High `grpo_mean_kl_move` trunc episodes (still under `lr=1e-5`):**

| Episode | Outcome | `grpo_mean_kl_move` |
|---------|---------|---------------------|
| 18 | trunc | 5.26 |
| 20 | trunc | **14.0** |
| 22 | trunc | 9.40 |

Wandb step chart for the run shows two phases: stable low KL for steps 0–~440,
then bursty spikes (often 15–40, one spike near 100 around step 270). Learning
still improved (win rate rose, self-play enemy synced) despite these spikes.

**Engine alignment (proxy from CSV, gate-only run):**

- `game_chosen_is_engine_argmax_in_group_rate` on wins: roughly **7–40%**
  (ep 7 high at 40%; most wins **12–18%**).
- `game_chosen_is_engine_best_overall_rate` on wins: mostly **3–14%**.
- Alignment did not track win rate strongly; wins were driven more by self-play
  dynamics and move quality variance than by policy matching engine argmax.

**Pikafish reliability on wins (from `output.log`, not in CSV):**

- Engine eval success on win episodes was mostly **84–97%**; ep 7 outlier **67.5%**.
- Wins did **not** correlate with engine-blind play; trunc and win episodes had
  similar per-turn eval failure rates (~15%).

### Post-patch run

Not started yet. Watch after ep 23:

- `[terminal-win]` stdout on ally wins
- Win episodes with **nonzero** `grpo_mean_kl_move` / `grpo_ppo_epochs_completed`
- `grpo_engine_align_*` columns (first run with alignment enabled)
- Whether `grpo/mean_kl_move` settles lower than the old bursty regime

## 5. Conclusion & Next Steps

**Does resuming from `ep_22` make sense?** Yes for the **adapter** — it is the
best recent policy. **No for the old optimizer** given the KL chart and new loss
terms: cold Adam at `5e-6` is the safer restart.

**Reconciling high KL with learning:** `grpo/mean_kl_move` is move-segment KL,
not a tiny per-token target. Values of 5–20 can coexist with improvement when
updates are sparse, clipped (`max_grad_norm=0.5`), and mostly on trunc episodes.
Sustained 15–30+ with spikes to 50–100 indicates **hot, noisy updates**, not
proof that aggressive settings are optimal. The run learned **despite** the spikes,
not because they were healthy.

**Next steps:**

1. Launch ep 23 with adapter-only resume (command above).
2. Confirm terminal-win GRPO fires on the first ally win.
3. If KL still spikes above ~15 on early trunc episodes, consider
   `grpo/lr=3e-6` before raising alignment coef.
4. If alignment stays flat after several wins, try
   `grpo/engine_policy_align_coef=0.08` or lower temperature `0.35`.

See also: `docs/logs/2026-05-25-log-terminal-win-grpo-update.md`,
`docs/logs/2026-05-22-log-engine-policy-distillation.md`.
