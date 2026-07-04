# Project Architecture

This document provides a high-level overview of the `FineTune-LLM-OnlineRL` project architecture.

## Overview
This project fine-tunes a Large Language Model (specifically `Qwen/Qwen2.5-7B-Instruct`) to play Chinese Chess (Xiangqi) using Online Reinforcement Learning. It employs a Group Relative Policy Optimization (GRPO) algorithm to optimize the LLM's decision-making directly from environment interactions without requiring a separate critic model. The model is trained using LoRA (Low-Rank Adaptation) to maintain efficiency and minimize memory overhead.

## Repository Map

- `LLM_RL_agent_FSDP_v2.py` — active Xiangqi GRPO v2 training entrypoint; owns the current Unsloth/LoRA training loop, legal-move sampler, Pikafish reward paths, self-play enemy flow, checkpointing, W&B/CSV metrics, and run heartbeat.
- `LLM_RL_agent.py`, `LLM_RL_agent_FSDP.py`, `FSDP.py` — earlier RL/FSDP training paths kept for reference and legacy experiments.
- `pikafish_eval.py` — Pikafish engine wrapper and cached board evaluation helpers used by training, analysis, and play flows.
- `xiangqi_board.py`, `xiangqi_labels.py` — Xiangqi board utilities and move-label/classification helpers shared by data and evaluation scripts.
- `scripts/` — runnable utilities for SFT data preparation, SFT training, offline evaluation, enemy-era analysis, benchmark runs, and the local web play server launcher.
- `scripts/benchmark/` — chess and Xiangqi benchmark harness, engine ladder calibration, prompt adapters, player wrappers, match runner, and Elo estimator.
- `muzero/` — MuZero/EfficientZero-style Xiangqi agent (tensor world model + MCTS self-play); Pikafish-only legality; independent of the LLM pipeline. Entrypoint `python -m muzero.train`. Spec: `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`.
- `web/` — FastAPI server and static board UI for human Red vs LoRA-backed Black play.
- `docs/` — canonical project documentation, including this repo map, Xiangqi metrics notes, the agent task queue, and dated experiment/feature logs.
- `docs/logs/` — dated logs for feature changes, experiment starts, experiment completions, failures, interruptions, conclusions, and follow-up plans. New logs follow `docs/logs/template.md`.
- `docs/AGENT_TODO.md` — active handoff queue for agents; update it when starting or finishing work, discovering blockers, or creating follow-up tasks.
- `data/` — SFT datasets, benchmark outputs, calibrated ladder data, and other generated inputs/outputs that are not model checkpoints.
- `checkpoints/` — LoRA adapters, tokenizer artifacts, optimizer state, enemy adapters, and interrupted-run snapshots from training experiments.
- `wandb/` — local W&B run artifacts and logs.
- `.cursor/` — Cursor project rules and hooks (legacy; project now uses Claude Code).
- `.claude/` — Claude Code project settings. `settings.json` contains a `Stop` hook that injects the 3-step handoff checklist into Claude's context at the end of every turn.

## Components

### 1. Training Loop (`LLM_RL_agent.py`)
- **Description**: The core RL loop integrates the `gym_xiangqi` environment with the LLM. It manages the formatting of board states into text prompts, parses the LLM's text outputs into valid game moves, and handles episodic reward collection.
- **Agent Roles**: The `ChineseChessAgent` provides game-specific system prompts and action extraction, while opponents (like `GreedyEnemyAgent` or Random agents) provide the opposing gameplay.

### 2. Group Relative Policy Optimization (GRPO)
- **Description**: A lightweight, online RL trainer (`GRPOTrainerOnline`). It collects queries (board states), responses (chosen moves), and rewards across episodes.
- **Mechanism**: Instead of a value network, GRPO normalizes rewards within a batch to compute advantages. It penalizes divergence from the reference policy using KL divergence, calculated by temporarily disabling the LoRA adapters.

### 3. Distributed Training (FSDP)
- **Description**: Manages Fully Sharded Data Parallelism for scaling up model sizes across multiple GPUs if needed.
- **Files**: `FSDP.py`, `LLM_RL_agent_FSDP.py`

### 3b. Xiangqi GRPO v2 (Unsloth + optional legal-move sampler)
- **Description**: `LLM_RL_agent_FSDP_v2.py` is the active **Xiangqi / GRPO** training script (Qwen2.5 via Unsloth, LoRA, optional FSDP). Candidate moves can be built by a **policy-scored legal-move sampler** (distinct legal actions without replacement), then scored with **Pikafish** per candidate; rewards may use **`gate`** mode (engine-first, format as gate + small bonus) and optional **grounded** rationale for a fixed `Move:` line. See `docs/logs/2026-05-11-log-legal-move-sampler.md` for the experiment log, including a **~26 h** run (started 2026-05-11, stopped 2026-05-12) with **no visible episode-level learning** (ally win rate stayed 0%).
- **Action selection in the legal-move sampler:** controlled by `grpo/legal_move_action_selection` (`"greedy"` default | `"first_sample"`). Under `"greedy"`, the argmax-policy legal move is forced into slot 0 of the GRPO group so the agent plays its own most-confident legal move (matches inference-time greedy decoding). Pikafish is **never** used as the action-selection oracle (`game/play_best_candidate=False`); it is only the reward source. See `docs/logs/2026-05-14-log-greedy-action-selection.md`.
- **Engine-best agreement tracking:** every ally turn now calls `pikafish_evaluator.bestmove_root_cached(fen)` once and emits, per turn (in `output.log` + W&B): whether Pikafish's overall best legal move is inside the 32 sampled (`game/engine_best_in_group`), whether the played move equals the in-group engine argmax (`game/chosen_is_engine_argmax_in_group`), whether it equals Pikafish's overall best (`game/chosen_is_engine_best_overall`), the chosen move's engine-reward rank inside the group, and the cp gap vs the in-group argmax. Episode aggregates use the `_rate` (%) suffix for the boolean flags. See `docs/logs/2026-05-14-log-greedy-action-selection.md` §2 for the metric list.
- **Engine-policy alignment loss:** GRPO is augmented with an optional legal-move distillation term (`grpo/engine_policy_align_coef`, `grpo/engine_policy_align_temperature`). For each sampled legal group, candidate engine rewards are converted into a soft target distribution and the trainer adds `KL(engine_target || policy_move_distribution)` over the candidate move region. This directly teaches the model's policy-top legal move to align with engine-good moves, and it still runs on low-reward-spread turns where the GRPO policy-gradient term is skipped. Episode CSV GRPO fields are episode means rather than the final turn's stat, with additional `grpo_engine_align_*`, `grpo_update_rate`, and `grpo_skip_low_reward_std_rate` diagnostics. See `docs/logs/2026-05-22-log-engine-policy-distillation.md`.
- **Terminal-win GRPO update:** When the ally receives the environment's terminal `+100` win reward after `env.step`, a second `train_group()` runs on the same pre-step sampled candidate group with the chosen move boosted to `reward/terminal_win_grpo_reward` (default `10.0`). This closes the gap where win episodes previously showed `grpo_mean_kl=0.0` because pre-step GRPO could not see the terminal reward. See `docs/logs/2026-05-25-log-terminal-win-grpo-update.md` and the long-run analysis in `docs/logs/2026-05-26-log-long-run-grpo-analysis-and-resume-plan.md`.
- **Self-play opponent (default):** when `game/self_play=True`, the enemy is a **frozen copy** of the ally LoRA (`enemy` adapter at `checkpoints/self_play_enemy`). The ally trains on the `default` adapter; after `game/self_play_wins_to_sync` consecutive ally wins, weights sync to the frozen enemy. **Asymmetric CP-saturation truncation** ends hopeless lost games early (`cp_before <= -4000` for 3 ally turns) with +100 to the winner; winning saturated positions play out. GreedyEnemy ε-curriculum is legacy-only (`game/self_play=False`). See `docs/logs/2026-05-21-log-self-play-and-asymmetric-truncation.md`.
- **Metrics**: `chinese_chess_episode_metrics_v2.csv` (and W&B) for episode aggregates, including **mean/median Red-oriented Pikafish value after ally moves** (`game/mean_ally_cp_after_move_red`, …) and optional EMA (`metrics/ally_cp_after_ema_alpha`). Reward mode **`xiangqi_r1`** implements discrete move + analysis + format signals (paper §3.4). **Stdout / W&B `output.log`:** per-episode and per-round **scoreboard** lines for cumulative env returns (`episode` vs `all_episodes`). **Run diagnostics:** rank-0 JSON heartbeat (`xiangqi_v2_run_heartbeat.json` by default, `training/run_heartbeat_path` / `""` to disable) updated each round + on SIGINT/SIGTERM/Python exceptions; defaults `max_rounds_per_episode=100`, `checkpoint/every_n_episodes=1`. See `docs/logs/2026-05-14-log-stdout-episode-scoreboard.md`.

### 3c. Xiangqi strategy SFT (offline, Xiangqi-R1 §3.1)
- **Description:** Real master-game PGNs (winner-side + draw moves) + Pikafish labels feed an Unsloth LoRA SFT. Pipeline scripts live under `scripts/`:
  - `download_xiangqi_pgn.py` — cache the `wukong-xiangqi` 40,711-game UCCI PGN under `data/xiangqi_sft/raw/`.
  - `xiangqi_pgn.py`, `xiangqi_selfplay.py` — streaming position iterators (PGN + Pikafish self-play fallback).
  - `build_xiangqi_sft_dataset.py` — main builder; default `--samples 50000` rows; `--target-move {best,human_good}` selects the assistant target.
  - `train_sft_xiangqi.py` — LoRA SFT on the resulting JSONL.
  - `eval_xiangqi_metrics.py` — paper-aligned `legal@k / good@k / best@k / 3-class@k / 5-class@k`.
- **Data:** `data/xiangqi_sft/` (see `data/xiangqi_sft/README.md`). Schema: per-row `messages` (system + user + assistant) and `meta` with FEN, played/best UCI, Red-positive cp, 3/5-class labels, is_good_move, phase, ply, source.
- **Metrics + benchmark bands:** `docs/XIANGQI_R1_METRICS.md`.

### 3e. Local web play (human Red vs LoRA engine)
- **Description:** `web/` + `scripts/serve_xiangqi_play.py` — FastAPI server and static board UI. Human plays Red; default LoRA checkpoint (e.g. `checkpoints/xiangqi_grpo_v2/ep_40`) plays Black via greedy legal-move logprob scoring on a flipped board. Pikafish validates all moves. See `web/README.md` and `docs/logs/2026-05-28-log-xiangqi-web-play-ui.md`.

### 3f. MuZero Xiangqi (tensor world model + MCTS)
- **Description:** `muzero/` implements an EfficientZero-style agent per
  `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`: 115×10×9 board
  tensors, 8100-action space masked to Pikafish-legal moves, 800-sim pUCT MCTS,
  K=8 unrolled training with policy/value/reward/moves-left/material/SimSiam
  losses (~22.1M params at the default 192-channel config), repetition-draw +
  hopeless-truncation adjudication, Pikafish warm start (MultiPV soft targets),
  and a periodic fixed-Pikafish gate. Self-play defaults to stock MuZero
  (`self_play_mode="latest"`): the newest network plays both sides, Dirichlet
  noise at every root, one batched MCTS group per lockstep round, symmetric
  hopeless-truncation, no enemy net. The original frozen-enemy scheme
  (promotion after 3 consecutive ally wins, asymmetric ally-only truncation)
  remains available as a research ablation via `self_play_mode="frozen_enemy"`.
  Spec: `docs/superpowers/specs/2026-07-03-muzero-latest-selfplay-design.md`. Value head uses a 601-bin categorical
  support over [−3, 3] h-transformed units (sized to the hybrid reward scale).
  Entrypoint: `python -m muzero.train` (`--smoke` for a tiny end-to-end run,
  `--resume checkpoints/muzero_xiangqi/latest.pt` to continue; checkpoints are
  written atomically each iteration). Tests: `uv run pytest muzero/tests`
  (engine-gated tests need `PIKAFISH_BIN`). Module map: `config.py` (all
  hyperparameters), `encoding.py`, `transforms.py`, `network.py`, `env.py`,
  `mcts.py`, `selfplay.py`, `replay_buffer.py`, `warmstart.py`, `train.py`,
  `metrics.py`. `selfplay.py`'s `_Game` tracks per-move root-policy
  entropy and (root value, engine cp) pairs (every move in latest mode,
  ally moves only in frozen mode) at no extra engine-call cost
  (the cp already comes back on `env.step`'s `info`); `metrics.py` rolls
  these into `selfplay/mean_root_entropy`, `selfplay/mean_ally_cp_auc`,
  `selfplay/value_cp_correlation`, and `selfplay/games_per_promotion`.
  `replay_buffer.py` stamps each `GameHistory` with a `buffer_index` on
  `add()` and `sample_batch()` returns a `mean_buffer_age` scalar, which
  `train.py`'s `MuZeroTrainer.train_batch` pops before tensorizing and
  reports back as `buffer_age` (aggregated into `loss/buffer_age` by the
  main loop). Deferred §10 metrics needing extra engine calls or GPU
  introspection are tracked in `docs/AGENT_TODO.md`.

### 3d. Inference-only Elo bench (chess + xiangqi)
- **Description:** `scripts/benchmark/` benchmarks any LLM (default
  `Qwen/Qwen2.5-7B-Instruct`, no LoRA) on both Western chess vs Stockfish
  and Xiangqi vs Pikafish, using a shared `go movetime` ladder per game
  (default rungs `10,50,200,1000,5000` ms) self-calibrated to within-pool
  Elo (top rung anchored at 3500). The LLM's Elo per game is fit with
  Bradley-Terry MLE (`elo_estimator.py`) using a Rao-Kupper draw extension,
  with 95% CI via case bootstrap. Reuses the v2 Xiangqi prompt verbatim
  (`xiangqi_prompt.py`) and a parallel-shape chess prompt
  (`chess_prompt.py`).
- **Outputs:** `data/benchmark/<game>_ladder_elo.json`,
  `<game>_results.json`, `<game>_winrate.png`, `summary.json`, plus
  per-game JSONL move logs under `data/benchmark/games/`.
- **Methodology + limitations:** see
  `docs/logs/2026-05-13-log-llm-chess-xiangqi-elo-bench.md`.

### 4. Experiment Logging & Evaluation
- **Description**: Tracks training progress, episodic returns, and GRPO loss metrics.
- **Location**: `docs/logs/` for qualitative iteration logs, and `chinese_chess_episode_metrics.csv` / `chinese_chess_episode_metrics_v2.csv` for quantitative metrics (v2 matches `LLM_RL_agent_FSDP_v2.py`).

## Data Flow

The training cycle follows a continuous generation and update loop:

1. **State Observation (Generation)**: The Xiangqi board state and legal moves are translated into a textual prompt.
2. **Action Selection**: The LLM processes the prompt and generates a move description (e.g., `Action: 8, (9, 0), (5, 0)`). If the format is invalid, a random legal move is chosen.
3. **Environment Step**: The environment executes the move and returns the next state and a reward (e.g., capturing a piece, winning, or losing).
4. **Reward Assignment**: Rewards are collected for each step and aggregated per episode.
5. **Model Update (GRPO)**: 
   - Once a batch is full, rewards are normalized to calculate advantages.
   - The current policy log-probabilities are calculated.
   - The reference policy log-probabilities are calculated (by disabling LoRA).
   - The LoRA weights are updated using an optimizer step based on the advantage and KL penalty.
