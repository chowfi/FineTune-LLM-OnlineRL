# CLAUDE.md

## Project Overview

This project fine-tunes `Qwen/Qwen2.5-7B-Instruct` to play Xiangqi (Chinese Chess) using GRPO + LoRA via Unsloth. The active training entrypoint is `LLM_RL_agent_FSDP_v2.py`. Supporting pipelines cover SFT data preparation, offline evaluation, Elo benchmarking against Pikafish/Stockfish, and a local web play UI. Package management is `uv` (`pyproject.toml`).

## Quick Orientation

| What | Where |
|---|---|
| Full file map + component descriptions | `docs/ARCHITECTURE.md` |
| Agent rules and conventions | `AGENTS.md` |
| Active task queue | `docs/AGENT_TODO.md` |
| Experiment log template | `docs/logs/template.md` |

## Session Start Protocol

At the start of every session, read `docs/AGENT_TODO.md` and pick up from the active task queue. Mark tasks in-progress when you start them, and completed or updated when you finish.

## Coding Conventions

Before finishing any task that modifies Python files, run:

```bash
ruff check . --fix
ruff format .
```

Do not edit files under `unsloth_compiled_cache/` — they are auto-generated.

## Experiment Logging & Handoff Accounting

Before any session is considered complete, do all three:

1. Write a dated log to `docs/logs/YYYY-MM-DD-log-<description>.md` using `docs/logs/template.md`
2. Update `docs/ARCHITECTURE.md` if the repo map or any component description changed
3. Update `docs/AGENT_TODO.md` — close finished tasks, add discovered follow-ups

## Heavy File Guardrails

Do **not** read the following unless explicitly instructed:

- `*.csv` metric files (`chinese_chess_episode_metrics*.csv`)
- `RL_Project.ipynb`, `deliverables/RL_Project_Submission.ipynb`
- Anything under `checkpoints/`, `wandb/`, `unsloth_compiled_cache/`
