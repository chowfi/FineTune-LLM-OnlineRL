# Project Architecture

This document provides a high-level overview of the `FineTune-LLM-OnlineRL` project architecture.

## Overview
This project fine-tunes a Large Language Model (specifically `Qwen/Qwen2.5-7B-Instruct`) to play Chinese Chess (Xiangqi) using Online Reinforcement Learning. It employs a Group Relative Policy Optimization (GRPO) algorithm to optimize the LLM's decision-making directly from environment interactions without requiring a separate critic model. The model is trained using LoRA (Low-Rank Adaptation) to maintain efficiency and minimize memory overhead.

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

### 4. Experiment Logging & Evaluation
- **Description**: Tracks training progress, episodic returns, and GRPO loss metrics.
- **Location**: `docs/logs/` for qualitative iteration logs, and `chinese_chess_episode_metrics.csv` for quantitative metrics.

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
