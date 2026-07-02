# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Final project for NYU DS-GA 3001.005 (Reinforcement Learning, Spring 2024) by Fiona Chow, Christine Gao, and Xinyue Ma.

**Hypothesis:** Using a pre-trained LLM as the starting policy and fine-tuning it with online RL should produce more sample-efficient learning than training from scratch, due to the LLM's prior knowledge.

**Environment:** [gym-xiangqi](https://github.com/zhaobin74/gym-xiangqi) (Chinese Chess / Xiangqi), with a fallback experiment on FrozenLake-v1.

## Primary Notebook

`RL_Project.ipynb` — the single deliverable notebook. It was developed and run on **Google Colab** (GPU required; training spikes ~40 GB VRAM for the 7B-parameter Llama 2 run).

`Other notebooks/` contains incremental drafts (numbered iterations). `RL_Project_Submission.ipynb` is the submission copy.

## Environment Setup

```bash
pip install gym-xiangqi
pip install peft trl llamagym
pip install transformers==4.38.2
pip install torch
```

For the LLM fine-tuning cells, a Hugging Face login is required:
```python
from huggingface_hub import login
login()  # prompts for HF token
```

Google Drive mounting (`drive.mount('/content/drive')`) is used for persisting logs and checkpoints — only relevant inside Colab.

## Architecture / Experiment Structure

The notebook progresses through six test cases:

| # | Agent | Notes |
|---|-------|-------|
| 1 | Random vs Random | Baseline; uses built-in `RandomAgent` from gym-xiangqi |
| 2 | Greedy | Custom `GreedyAgent` — picks move with highest immediate piece-capture reward |
| 3 | DQN | PyTorch `DQNModel` (CNN) + `ReplayMemory`; epsilon-greedy exploration |
| 4 | DDQN | `DDQN` extends `DQNAgent`; two Q-value estimators to reduce maximization bias |
| 5 | AlphaZero | Attempted integration with alpha-zero-general library; abandoned due to API incompatibility |
| 6 | LLM + PPO + LoRA | Core contribution — fine-tunes OPT-125M (and briefly Llama 2 7B) via PPO using the LlamaGym framework |

## Key Classes

- **`GreedyAgent`** — greedy piece-capture heuristic, used as the opponent throughout LLM training.
- **`ReplayMemory`** — experience replay buffer (default 50 K capacity, 10 K burn-in).
- **`DQNModel`** — PyTorch `nn.Module` taking `env.observation_space.shape` as input.
- **`Agent`** (abstract, adapted from LlamaGym) — wraps any HuggingFace causal LM; handles prompt construction, token truncation, and PPO updates.
- **`ChineseChessAgent`** / **`FrozenLakeAgent`** — concrete subclasses that implement `get_system_prompt()` for each environment.

## Evaluation Metrics

Four metrics tracked per experiment: win rate, average reward, episode length, and reward variability. Helper functions `calculate_win_rate`, `average_reward`, `average_episode_length`, and associated `plot_*` functions are defined early in the notebook and reused throughout.

## Known Constraints / Findings

- OPT-125M is too small to generate coherent chess moves; responses were mostly unintelligible.
- Llama 2 7B produces sensible output but requires ~40 GB GPU RAM — beyond the A100 (40 GB) available on Colab.
- PPO loss did not decrease during training, indicating the LLM was not learning effectively at this scale.
- The `transformers==4.38.2` pin is load-bearing — later versions broke compatibility with the gym-xiangqi + trl stack at the time of the project.
