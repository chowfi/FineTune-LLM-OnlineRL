"""Lightweight Pikafish evaluator construction.

Split out of ``game_session.py`` so callers that only need Pikafish (e.g.
``app.py``'s lifespan, before it knows whether to load the LLM or MuZero
engine) don't transitively pull in the heavy ``transformers``/``peft``/
``torch`` stack via ``game_session -> engine_player``.
"""

from __future__ import annotations

import os

from src.pikafish_eval import PikafishEvaluator


def build_pikafish() -> PikafishEvaluator:
    binary = os.environ.get("PIKAFISH_BIN", "").strip()
    if not binary:
        raise RuntimeError(
            "Set PIKAFISH_BIN to your Pikafish executable, e.g. "
            "export PIKAFISH_BIN=/home/fchow/bin/pikafish"
        )
    eng = PikafishEvaluator(binary, depth=8, movetime_ms=200, verbose=False)
    if not eng.enabled:
        raise RuntimeError(f"Pikafish failed to start: {binary}")
    return eng
