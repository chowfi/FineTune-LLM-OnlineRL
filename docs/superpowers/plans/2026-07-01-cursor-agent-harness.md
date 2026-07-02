# Cursor Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Cursor-only rule and hook harness that forces agents to account for repo-map, dated-log, and task-queue handoff updates before finishing feature work or experiment runs.

**Architecture:** Cursor project rules provide persistent instructions at session/task time. A project-level `stop` hook checks final agent output for a handoff checklist and sends a follow-up prompt if it is missing. Existing docs remain canonical: `docs/ARCHITECTURE.md` for the repo map, `docs/AGENT_TODO.md` for tasks, and dated files under `docs/logs/` for feature and experiment logs.

**Tech Stack:** Cursor `.mdc` rules, Cursor project hooks JSON, Python 3 standard library, Markdown docs.

---

## File Structure

- Create `.cursor/rules/agent-handoff.mdc`: always-applied Cursor rule with project goal, startup context, and completion handoff requirements.
- Create `.cursor/hooks.json`: project hook configuration for the agent `stop` event.
- Create `.cursor/hooks/agent_handoff_stop.py`: lightweight command hook that asks the agent to revise final output if the handoff checklist is missing.
- Modify `docs/ARCHITECTURE.md`: add a repo map section that explains major files and folders.
- Modify `docs/logs/template.md`: make the template cover both feature changes and experiment outcomes.
- Create `docs/logs/2026-07-01-log-cursor-agent-harness.md`: dated feature log for this harness implementation.
- Modify `docs/AGENT_TODO.md`: record the completed harness work and any follow-up.

### Task 1: Cursor Rule

**Files:**
- Create: `.cursor/rules/agent-handoff.mdc`

- [x] **Step 1: Add always-applied handoff rule**

```markdown
---
description: Keep agents oriented around Xiangqi GRPO and enforce handoff updates
alwaysApply: true
---

# Xiangqi GRPO Agent Handoff

This repository's goal is to train an LLM-backed reinforcement-learning agent using GRPO to play Xiangqi.

Before substantive work, read `docs/AGENT_TODO.md` and the relevant section of `docs/ARCHITECTURE.md`.

Before claiming feature/code/doc work is done, or before claiming an experiment/training run finished, failed, or was interrupted, account for:

- `docs/ARCHITECTURE.md`: update when repo map, module responsibilities, scripts, commands, data flow, training behavior, or durable workflow changed.
- `docs/logs/YYYY-MM-DD-log-<description>.md`: create or update a dated log for feature changes, experiment starts, finishes, failures, interruptions, and conclusions.
- `docs/AGENT_TODO.md`: move completed work, add next steps, or record blockers.

Final responses after covered work must include:

```text
Handoff updated:
- Architecture/repo map: yes/no, reason
- Log: yes/no, path
- Agent TODO: yes/no, reason
```

Do not say "done", "complete", "fixed", or "experiment finished" until the handoff is complete or explicitly marked not applicable with a reason.
```

### Task 2: Cursor Stop Hook

**Files:**
- Create: `.cursor/hooks.json`
- Create: `.cursor/hooks/agent_handoff_stop.py`

- [x] **Step 1: Configure stop hook**

```json
{
  "version": 1,
  "hooks": {
    "stop": [
      {
        "command": ".cursor/hooks/agent_handoff_stop.py",
        "timeout": 5,
        "loop_limit": 2,
        "failClosed": false
      }
    ]
  }
}
```

- [x] **Step 2: Add hook script**

```python
#!/usr/bin/env python3
"""Cursor stop hook for required agent handoff checklist."""

from __future__ import annotations

import json
import re
import sys
from typing import Any


REQUIRED_PATTERNS = [
    re.compile(r"Handoff updated\s*:", re.IGNORECASE),
    re.compile(r"Architecture/repo map\s*:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"Log\s*:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"Agent TODO\s*:\s*(yes|no)", re.IGNORECASE),
]


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(collect_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(collect_strings(item))
        return strings
    return []


def main() -> int:
    raw_input = sys.stdin.read()
    try:
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw_input": raw_input}

    text = "\n".join(collect_strings(payload))
    has_checklist = all(pattern.search(text) for pattern in REQUIRED_PATTERNS)
    if has_checklist:
        print("{}")
        return 0

    print(
        json.dumps(
            {
                "followup_message": (
                    "Before finalizing covered work, add the required handoff "
                    "checklist with architecture/repo map status, dated log path "
                    "or reason, and Agent TODO status or reason."
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Task 3: Documentation Handoff Updates

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/logs/template.md`
- Create: `docs/logs/2026-07-01-log-cursor-agent-harness.md`
- Modify: `docs/AGENT_TODO.md`

- [x] **Step 1: Update docs**

Add a repo map to `docs/ARCHITECTURE.md`, expand the log template fields for features and experiments, create a dated feature log for this harness, and record the completed work in `docs/AGENT_TODO.md`.

### Task 4: Verification

**Files:**
- Test: `.cursor/hooks/agent_handoff_stop.py`

- [x] **Step 1: Make hook executable**

Run: `chmod +x .cursor/hooks/agent_handoff_stop.py`

- [x] **Step 2: Verify missing checklist prompts follow-up**

Run:

```bash
printf '{"response":"done"}' | .cursor/hooks/agent_handoff_stop.py
```

Expected: JSON containing `followup_message`.

- [x] **Step 3: Verify completed checklist allows stop**

Run:

```bash
printf '{"response":"Handoff updated:\n- Architecture/repo map: yes, updated\n- Log: yes, docs/logs/example.md\n- Agent TODO: yes, updated"}' | .cursor/hooks/agent_handoff_stop.py
```

Expected: `{}`.

- [x] **Step 4: Lint and format Python hook**

Run:

```bash
uv run ruff check .cursor/hooks/agent_handoff_stop.py --fix
uv run ruff format .cursor/hooks/agent_handoff_stop.py
```

Expected: both commands pass.
