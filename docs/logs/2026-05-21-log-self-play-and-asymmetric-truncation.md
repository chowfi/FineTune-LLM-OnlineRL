# Experiment Log: Self-Play and Asymmetric CP-Saturation Truncation

**Date:** 2026-05-21
**Agent/Author:** AI Coding Agent

## 1. Hypothesis / Goal
Our goal is to significantly stabilize the training of the Xiangqi GRPO model and prevent the high volatility observed in previous runs. We address three primary root causes of volatility:
1. **Double Non-Stationarity:** Training against a continuously updating opponent creates non-stationary environments. By freezing the opponent (the `enemy` PEFT adapter) and only updating it once the learning policy (the `default` adapter) demonstrates a clear, statistically significant superiority (winning 3 consecutive episodes), we convert the multi-agent problem back into a series of stationary, solvable single-agent MDPs.
2. **Endpoints Practice:** Asymmetric truncation ensures that the model only truncates early when in overwhelmingly hopeless/losing positions (CP <= -4000). Winning positions are played out, allowing the policy to practice checkmating and capturing under real material advantages, receiving final checkmate feedback.
3. **Loss of Signal on Truncation:** Truncating early previously left advantage signals flattened or incomplete. By awarding a clear terminal reward of `+100.0` to the winning side upon CP-saturation truncation, we ground the advantage calculations and stabilize training gradients.

## 2. Configuration Changes
We introduced and modified the following hyperparameters/logic in `LLM_RL_agent_FSDP_v2.py`:
- `game/self_play`: Set to `True` (enables dual-adapter self-play in the collective FSDP loop).
- `game/self_play_wins_to_sync`: `3` consecutive ally wins before the frozen enemy adapter is updated.
- `game/cp_saturation_threshold`: Keep at `4000.0` but apply **asymmetrically** (only trigger streak increment when `cp_before <= -4000.0` for the ally agent).
- Added `game_consecutive_self_play_wins` field in the per-episode metrics CSV schema to track state and maintain seamless resume/crash safety.
- Implemented `sync_enemy_adapter_with_default` to dynamically synchronize parameters from `default` to `enemy` in-memory across distributed GPU shards when the ally wins enough games in a row.
- **Logging:** startup and per-episode stdout/W&B lines now describe self-play (frozen enemy adapter, win streak, asymmetric truncation) instead of GreedyEnemy ε-anneal. `game_enemy_epsilon_current` is omitted from CSV/W&B when self-play is active.

## 3. Run Command
Use the following command to launch a **fresh** self-play training run (SFT adapter, episode 1, CSV cleared):

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
```

Defaults: `checkpoint/load_adapter_path=checkpoints/xiangqi_sft`, `checkpoint/start_episode=1`, `metrics/clear_csv_on_start=True`. The frozen self-play enemy at `checkpoints/self_play_enemy` is reset from the loaded ally adapter on a fresh start.

To resume a prior RL run instead:

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_29 --start-episode 30
```

When resuming, pass `--resume-from` and `--start-episode`; the script keeps the existing CSV and reuses the saved `checkpoints/self_play_enemy` adapter unless you start from episode 1 with a cleared CSV.

## 4. Quantitative Results
*Awaiting run execution and metrics collection.*
- **Win Rate:** TBD
- **Average Episode Length:** TBD
- **GRPO Loss:** TBD
- **Consecutive Self-Play Wins / Sync Rate:** TBD

## 5. Conclusion & Next Steps
- **Next Step:** Execute the run, monitor Wandb for the `game/consecutive_self_play_wins` metric, and verify that the `default` policy adapter successfully improves and synchronizes with the `enemy` adapter.
- Ensure that the saved checkpoints under `checkpoints/self_play_enemy` are preserved and can be loaded upon preloading or starting a fresh resume session.
