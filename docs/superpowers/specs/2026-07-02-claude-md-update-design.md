---
title: CLAUDE.md Update Design
date: 2026-07-02
status: approved
---

## Goal

Replace the stale Spring 2024 class-project CLAUDE.md with a lightweight entry-point doc that accurately describes the current active codebase and enforces the same handoff accounting rules that existed in Cursor, now for Claude Code.

## Approach

Option A — thin pointer doc. CLAUDE.md stays concise (~60–80 lines), contains the always-visible rules Claude needs, and delegates detail to existing canonical docs (`docs/ARCHITECTURE.md`, `AGENTS.md`). No duplication.

## Sections

### 1. Project Overview
Three to four sentences describing the current state:
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Algorithm: GRPO + LoRA via Unsloth
- Active entrypoint: `LLM_RL_agent_FSDP_v2.py`
- Supporting pipelines: SFT data prep, offline eval, Elo benchmarking, local web play UI
- Package management: `uv` / `pyproject.toml`

No mention of original class project (OPT-125M, Llama 2, `RL_Project.ipynb`).

### 2. Quick Orientation
Four pointer lines only:
- Full file map + component descriptions → `docs/ARCHITECTURE.md`
- Agent rules + conventions → `AGENTS.md`
- Active task queue → `docs/AGENT_TODO.md`
- Experiment log template → `docs/logs/template.md`

### 3. Session Start Protocol
Single rule: read `docs/AGENT_TODO.md` at the start of every session; pick up from the active queue; mark tasks in-progress when starting, completed/updated when finishing.

### 4. Coding Conventions
Single rule: before finishing any task that modifies Python files, run `ruff check . --fix` then `ruff format .`. Exclude `unsloth_compiled_cache/` (auto-generated, do not edit).

### 5. Experiment Logging & Handoff Accounting
Three-step checklist that mirrors the Cursor stop hook, to be completed before any session is considered done:
1. Write a dated log to `docs/logs/YYYY-MM-DD-log-<description>.md` using `docs/logs/template.md`
2. Update `docs/ARCHITECTURE.md` if the repo map or any component description changed
3. Update `docs/AGENT_TODO.md` — close finished tasks, add discovered follow-ups

### 6. Heavy File Guardrails
Never read the following unless explicitly instructed:
- `*.csv` metric files (`chinese_chess_episode_metrics*.csv`)
- `RL_Project.ipynb`, `deliverables/RL_Project_Submission.ipynb`
- Anything under `checkpoints/`, `wandb/`, `unsloth_compiled_cache/`

## Companion: `.claude/settings.json` Stop Hook

Create `.claude/settings.json` with a `Stop` event hook that echoes the three-step handoff checklist into Claude's context whenever a session ends. This provides enforcement in addition to the CLAUDE.md rule.

Hook command echoes:
```
[Handoff Checklist]
1. Write a dated log to docs/logs/YYYY-MM-DD-log-<description>.md (use docs/logs/template.md)
2. Update docs/ARCHITECTURE.md if the repo map or any component changed
3. Update docs/AGENT_TODO.md — close finished tasks, add follow-ups
```

## Files Changed

| File | Action |
|---|---|
| `CLAUDE.md` | Overwrite with new content |
| `.claude/settings.json` | Create with Stop hook |

## Out of Scope

- Changes to `AGENTS.md` or `docs/ARCHITECTURE.md` content
- Adding new rules beyond what was agreed above
- Any code changes
