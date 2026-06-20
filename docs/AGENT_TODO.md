# Agent Task Queue

This document tracks prioritized tasks for AI agents. When starting a new session, check this list first.

## Active Tasks (In Progress)
*(Add `[ ]` and move tasks here when an agent starts working on them)*

- `[ ]` Implement `scripts/dry_run.sh` to test the RL loop with a tiny model without requiring massive GPU resources.

## Backlog (To Do)
*(Add new ideas, bug fixes, or feature requests here)*

- `[ ]` Add unit tests for `ChineseChessAgent.extract_action()` to guarantee regex parsing stability.
- `[ ]` Verify deterministic behavior for `GreedyEnemyAgent` using static boards.

## Completed
*(Move finished tasks here)*

- `[x]` Set up initial agentic test harness and directory structure.
- `[x]` Establish `docs/logs/template.md` and logging conventions.
