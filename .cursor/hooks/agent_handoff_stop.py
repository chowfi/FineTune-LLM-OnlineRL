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
