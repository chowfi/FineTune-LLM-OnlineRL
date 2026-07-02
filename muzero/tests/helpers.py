"""Shared test helpers: engine gating + fake evaluator."""

import os

import pytest

PIKAFISH_BIN = os.environ.get("PIKAFISH_BIN", "")

requires_engine = pytest.mark.skipif(
    not (PIKAFISH_BIN and os.path.exists(PIKAFISH_BIN)),
    reason="PIKAFISH_BIN not set or binary missing",
)


def make_evaluator():
    from src.pikafish_eval import PikafishEvaluator

    return PikafishEvaluator(
        binary_path=PIKAFISH_BIN,
        depth=8,
        timeout_sec=15.0,
        movetime_ms=100,
        verbose=False,
    )


class FakeEvaluator:
    """Scripted stand-in for PikafishEvaluator (legality + cp)."""

    enabled = True

    def __init__(self, cp_fn=lambda fen: 0.0, legal_fn=lambda fen: ["a0a1"]):
        self.cp_fn = cp_fn
        self.legal_fn = legal_fn

    def evaluate_cp(self, fen, moves=None):
        return self.cp_fn(fen)

    def list_legal_moves(self, fen):
        return self.legal_fn(fen)
