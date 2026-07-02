# Experiment Log: stdout episode / round scoreboard

**Date:** 2026-05-14  
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Make W&B-captured `output.log` and the board sync file easier to scan by printing **cumulative environment returns** at episode boundaries and at each round (after each enemy move and before/after each ally move), both **within the current episode** and **across all completed episodes plus the current one** (`all_episodes`).

## 2. Configuration Changes

- Code only: `LLM_RL_agent_FSDP_v2.py` — helpers `format_episode_open_scoreboard` and `format_round_scoreboard`; rolling sums `season_ally_return` / `season_enemy_return` updated at the end of each episode (rank 0).
- **Episode cap:** `max_rounds_per_episode` default **100** (was 200) — shorter games, `truncated_cap` sooner if no terminal.
- **Checkpoints:** `checkpoint/every_n_episodes` default **1** (was 25) — one LoRA save per finished episode under `checkpoint/dir` (`ep_1`, `ep_2`, …) plus `final` on clean exit.
- **Run heartbeat (JSON):** default file `xiangqi_v2_run_heartbeat.json` (repo root). Rank 0 atomically rewrites it after each listed **phase** so abrupt stops (OOM / `kill -9`) still leave the **last completed** round on disk. Disable with hyperparam `training/run_heartbeat_path: ""`. Status values include `running`, `signal:SIGINT`, `signal:SIGTERM`, `python_exception` (with `traceback_excerpt`), `training_loop_complete`, and `process_exit_unclean` (atexit fallback when neither signal nor Python wrote a terminal reason).

## 3. Run Command

No new run required for this change. Example (unchanged):

```bash
uv run python LLM_RL_agent_FSDP_v2.py
```

## 4. Quantitative Results

N/A (logging-only change).

**Log line shapes**

- Episode open: `[Ep N] scoreboard: episode ally=0.00 enemy=0.00 | all_episodes ally=… enemy=…`
- Each round: `[Ep N Rd R] scoreboard: episode ally=… enemy=… | all_episodes ally=… enemy=…` (also appended to the enemy-move `log_board_sync` block so `xiangqi_v2_board_sync.log` stays aligned).
- Episode summary line: adds `all_episodes_ally=… all_episodes_enemy=…` next to existing `ally_return` / `enemy_return`.

## 5. Conclusion & Next Steps

- **Episode** fields match `ally_return` / `enemy_return` from the gym step rewards for the in-progress game.
- **all_episodes** is the sum of finished episodes’ returns plus the current episode’s returns so far (terminal partial episodes on crash are not folded into `season_*` until the episode completes on rank 0).
- Optional follow-up: mirror the same lines inside `_generate_policy_sampled_legal_candidates` if we want the scoreboard on the same line as `Policy-sampled` (currently a dedicated line immediately before ally `act_and_train`).

## 6. Heartbeat phases (debugging silent stops)

Phases written on each flush (with `episode`, `round_idx`, `ally_return`, `enemy_return`, `global_train_step`):

| `phase` | When |
|--------|------|
| `before_episode_for_loop` | Once after env/agent setup, before episode 1 |
| `episode_reset` | After env reset at each new episode |
| `after_enemy_env_step` | After each black step and board sync (same `Rd` as the enemy line) |
| `before_ally_grpo` | Immediately before `act_and_train` (heavy GRPO + generation) |
| `after_ally_env_step` | After ally `env.step` and scoreboard print |
| `episode_csv_appended` | After episode row appended to `chinese_chess_episode_metrics_v2.csv` |
| `saved_final_checkpoint` | After successful `final` LoRA save |

**Interpreting a stop:** If `output.log` ends mid-line with no Python traceback, open `xiangqi_v2_run_heartbeat.json`. If `status` is still `running` and `phase` is `before_ally_grpo`, the process died inside forward/generation (SIGKILL, CUDA assert, unplug, etc.). If `process_exit_unclean` appears, the interpreter exited without our handlers (again often SIGKILL). `dmesg` / `journalctl -k` may show OOM killer lines.

## 7. Limitations

- **SIGKILL** cannot run Python handlers; the JSON reflects the **last** round that finished writing.
- Multi-GPU rank ≠ 0 does not write the heartbeat file (only rank 0 runs the env loop in this script).
