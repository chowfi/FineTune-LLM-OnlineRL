# Agent Directives

This document outlines the guidelines, goals, and constraints for any AI agent interacting with the `FineTune-LLM-OnlineRL` codebase. Please adhere to these instructions when contributing.

## 1. Project Context
This project focuses on Fine-Tuning Large Language Models using Online Reinforcement Learning. 

## 2. Directory Structure & Conventions
- `docs/`: Contains all project documentation. Check `docs/ARCHITECTURE.md` for system design before making structural changes.
- `docs/logs/`: Used for logging different iterations and experiment runs. 
  - **Rule**: All new log files must be placed in `docs/logs/` and follow the naming convention `YYYY-MM-DD-log-<description>.md`.
- **Scripts**: Training scripts like `FSDP.py`, `LLM_RL_agent.py`, etc., reside in the root directory. Ensure any modifications to them are tracked.

## 3. Communication and Action Rules
- **Task Queue & Handover**: Always check `docs/AGENT_TODO.md` at the start of a session to see what tasks are prioritized. Update the list when you start or finish a task.
- **Modifications & Logging**: Log any substantial changes, experiments, or testing iterations in the `docs/logs/` directory. You MUST use the `docs/logs/template.md` format for all experiment logs.
- **Testing**: Before finalizing a feature, ensure that you provide or update a corresponding test script or harness.
- **Linting & Formatting**: Before completing any task that modifies Python files, you MUST run `ruff check . --fix` and `ruff format .` to adhere to the repository style constraints.

## 4. Context Boundaries
- To avoid wasting context window, **DO NOT** read the contents of heavy files like `.csv` metrics, `RL_Project.ipynb`, or anything inside `video/` or `checkpoints/` unless explicitly instructed to debug them.

## 5. Specific Skills
*(To be populated with local skills and tools available for agents in this project)*
