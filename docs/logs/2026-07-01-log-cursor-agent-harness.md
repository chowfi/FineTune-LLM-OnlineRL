# Cursor Agent Harness

**Date:** 2026-07-01
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Add a Cursor-only harness so future agents keep the repo goal in context and cannot finalize covered feature work or experiment-run outcomes without accounting for:

- `docs/ARCHITECTURE.md` as the repo map.
- Dated feature/experiment logs under `docs/logs/`.
- `docs/AGENT_TODO.md` as the task queue and handoff list.

## 2. Configuration Changes

- Added `.cursor/rules/agent-handoff.mdc`, an always-applied project rule for Xiangqi GRPO context and required handoff updates.
- Added `.cursor/hooks.json` with a project `stop` hook.
- Added `.cursor/hooks/agent_handoff_stop.py`, a lightweight Python hook that asks the agent to revise a final response if the handoff checklist is missing.
- Added a repository map section to `docs/ARCHITECTURE.md`.
- Expanded `docs/logs/template.md` to support feature logs and experiment outcomes.
- Added design and implementation plan docs under `docs/superpowers/`.

## 3. Run Command

```bash
chmod +x .cursor/hooks/agent_handoff_stop.py
printf '{"response":"done"}' | .cursor/hooks/agent_handoff_stop.py
printf '{"response":"Handoff updated:\n- Architecture/repo map: yes, updated\n- Log: yes, docs/logs/example.md\n- Agent TODO: yes, updated"}' | .cursor/hooks/agent_handoff_stop.py
uv run ruff check .cursor/hooks/agent_handoff_stop.py --fix
uv run ruff format .cursor/hooks/agent_handoff_stop.py
```

## 4. Quantitative Results

- Hook missing-checklist case: emitted a `followup_message` asking for the handoff checklist.
- Hook completed-checklist case: returned `{}` and allowed completion.
- Ruff check/format: `uv run ruff check .cursor/hooks/agent_handoff_stop.py --fix` passed; `uv run ruff format .cursor/hooks/agent_handoff_stop.py` left the file unchanged.

## 5. Qualitative Outcome

The harness is intentionally Cursor-only. The project rule provides persistent context and handoff requirements, while the stop hook makes final responses include a concrete handoff statement.

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md`: updated with repository map and Cursor harness location.
- `docs/AGENT_TODO.md`: updated with completed harness work after verification.
- Related logs/docs: this log, `docs/logs/template.md`, `docs/superpowers/specs/2026-07-01-cursor-agent-harness-design.md`, and `docs/superpowers/plans/2026-07-01-cursor-agent-harness.md`.

## 7. Conclusion & Next Steps

- Confirm Cursor loads the project rule and hook in the Hooks settings/output UI after file creation.
