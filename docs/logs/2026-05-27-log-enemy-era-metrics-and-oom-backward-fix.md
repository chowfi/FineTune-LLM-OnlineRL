# Experiment Log: Enemy-era metrics, CSV engine reward, OOM backward fix

**Date:** 2026-05-27
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal

1. Add `game_mean_chosen_engine_reward` to episode CSV.
2. Track frozen self-play enemy generation (`game_self_play_enemy_id`) for per-opponent analysis.
3. Fix ep 33 crash: CUDA OOM on batched GRPO **backward** (engine-policy alignment + deferred graph).
4. Make W&B episode metrics use `episode` as x-axis and log `enemy/sync_marker` spikes at sync.

## 2. Configuration Changes

- CSV columns: `game_mean_chosen_engine_reward`, `game_self_play_enemy_id`, `game_global_train_step_end`.
- `checkpoints/self_play_enemy/enemy_meta.json` tracks enemy generation (seeded `enemy_id=2` after ep30 sync).
- W&B: `setup_wandb_episode_metric_axes()` binds `game/*` and `enemy/*` to `episode` step.
- GRPO: OOM on combined backward → retry GRPO-only; if still OOM → sequential fallback.
- New helper: `scripts/analyze_enemy_eras.py`.

## 3. Run Command

Resume after ep33 mid-episode OOM:

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_33 \
  --start-episode 33
```

Analyze enemy eras:

```bash
uv run python scripts/analyze_enemy_eras.py --ep-start 5 --ep-end 32
```

## 4. Quantitative Results

- Ep 33 crashed after ~480 GRPO steps with OOM at `(epoch_group_loss + align_loss).backward()` (28.28 GiB allocated).
- Ep 5–32 enemy-era summaries unchanged from prior analysis; script reproduces them.

## 5. Conclusion & Next Steps

- Resume ep 33 from `ep_33` checkpoint; expect `[GRPO] CUDA OOM during batched backward` → GRPO-only or sequential fallback instead of hard crash.
- In W&B: plot `game/mean_chosen_engine_reward` vs **Episode** (not global step); overlay `enemy/sync_marker` for vertical sync spikes at ep 16/31 boundaries.
- Use `game/self_play_enemy_id` to color/filter lines by opponent generation.
