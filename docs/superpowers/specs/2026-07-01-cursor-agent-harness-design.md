# Cursor Agent Harness Design

**Date:** 2026-07-01
**Status:** Approved and implemented

## Goal

Create a Cursor-only agent harness that keeps future agents oriented around this repository's purpose and forces a handoff update before they claim feature work or experiment runs are complete.

The project goal agents must keep in mind is: **train an LLM-backed reinforcement-learning agent using GRPO to play Xiangqi**.

## Scope

The harness covers:

- Code, documentation, script, training-behavior, data-flow, and repo-structure feature work.
- Experiment or training runs that finish, fail, or are interrupted.
- Agent handoff updates to the repo map, experiment logs, and task queue.

The harness does not add a repo-owned validation script. Enforcement lives in Cursor rules and hooks only, per user preference.

## Canonical Files

- `docs/ARCHITECTURE.md` is the canonical repo map and project architecture document. It should explain the overall goal, major files and folders, active training path, data flow, and durable workflow changes.
- `docs/AGENT_TODO.md` is the canonical task queue and handoff list. Agents must move completed tasks, add follow-ups, and record blockers there.
- `docs/logs/YYYY-MM-DD-log-<description>.md` files are the canonical feature and experiment logs. New logs must follow `docs/logs/template.md`.

## Cursor Rule Design

Add a project Cursor rule that tells agents to do the following before starting substantive work:

1. Read `docs/AGENT_TODO.md`.
2. Read the relevant section of `docs/ARCHITECTURE.md`.
3. Keep the project goal in context: train an LLM-backed GRPO agent for Xiangqi.

The rule also tells agents to complete a handoff before finishing any covered task:

1. Update `docs/ARCHITECTURE.md` if the repo map, module responsibilities, scripts, commands, data flow, training behavior, or durable workflow changed.
2. Create or update a dated file in `docs/logs/` for feature changes, experiment starts, experiment finishes, failures, interruptions, and conclusions.
3. Update `docs/AGENT_TODO.md` by moving completed work, adding next steps, or recording blockers.

The rule must explicitly say agents may not claim "done", "complete", "fixed", or "experiment finished" until they have accounted for these handoff updates.

## Cursor Hook Design

Add a Cursor hook that acts as a final completion gate inside Cursor. The hook should inject or require a handoff checklist whenever an agent is about to produce a final response after covered work.

Required checklist:

```text
Handoff updated:
- Architecture/repo map: yes/no, reason
- Log: yes/no, path
- Agent TODO: yes/no, reason
```

The hook should remind the agent to ask:

- Did this task change code, docs, scripts, training behavior, data flow, repo structure, or experiment procedure?
- Did an experiment or training run finish, fail, or get interrupted?
- If yes, were `docs/ARCHITECTURE.md`, `docs/AGENT_TODO.md`, and the relevant dated `docs/logs/` file updated?

The hook should stay lightweight. It should not try to fully parse repository state, judge experiment quality, or infer every required documentation change. Its job is to make the agent explicitly account for handoff obligations before finishing.

## Documentation Updates

As part of implementation:

- Strengthen `docs/ARCHITECTURE.md` with a clear repo map section that explains what each major file and folder does.
- Refine `docs/logs/template.md` if the current template does not clearly support both feature logs and experiment outcomes.
- Keep `AGENTS.md` aligned if needed, but the enforcement mechanism should be Cursor rules/hooks, not a repo script.

## Testing

Verification should be manual and focused:

1. Inspect the Cursor rule to confirm it names the project goal and canonical files.
2. Inspect the hook configuration to confirm it triggers the final handoff reminder/gate.
3. Run or simulate a simple agent completion path and confirm the required handoff checklist appears before final completion.

## Open Decisions Resolved

- Enforcement approach: Cursor hooks/rules only.
- Covered work: all code/doc feature work plus experiment run completion, failure, or interruption.
- Repo map location: `docs/ARCHITECTURE.md`.
- Experiment log location: dated files under `docs/logs/`, not a central `logs.md`.
