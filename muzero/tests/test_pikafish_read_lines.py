"""Regression test for the perft read-loop timeout.

``PikafishEvaluator._read_lines`` previously early-exited only on
``bestmove`` lines, but ``go perft 1`` output ends with ``Nodes searched: N``
instead — so every uncached ``list_legal_moves`` call blocked for the full
``timeout_sec`` (15 s in the MuZero config, ~8 h of dead time across a
2,000-ply warm start). No engine binary needed: a pipe stands in for the
process. See docs/logs/2026-07-03-log-warmstart-perft-timeout.md.
"""

import os
import time

from src.pikafish_eval import PikafishEvaluator


class _FakeProc:
    """Just enough of a Popen: a readable stdout that stays open (no EOF)."""

    def __init__(self, payload: bytes):
        read_fd, self._write_fd = os.pipe()
        os.write(self._write_fd, payload)
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)

    def poll(self):
        return None

    def close(self):
        os.close(self._write_fd)
        self.stdout.close()


def _make_evaluator_with_payload(payload: bytes):
    ev = PikafishEvaluator(binary_path="pikafish-binary-that-does-not-exist", depth=1)
    assert ev.proc is None  # binary not found -> never launched
    ev.proc = _FakeProc(payload)
    return ev


def test_read_lines_returns_promptly_after_perft_output():
    ev = _make_evaluator_with_payload(b"a0a1: 1\na0a2: 1\nNodes searched: 44\n")
    try:
        start = time.monotonic()
        lines = ev._read_lines(3.0)
        elapsed = time.monotonic() - start
    finally:
        ev.proc.close()
    assert lines[-1] == "Nodes searched: 44"
    assert elapsed < 1.0, f"perft read blocked {elapsed:.2f}s (ran to timeout)"


def test_read_lines_still_exits_on_bestmove():
    ev = _make_evaluator_with_payload(
        b"info depth 8 score cp 31 pv h2e2\nbestmove h2e2 ponder h9g7\n"
    )
    try:
        start = time.monotonic()
        lines = ev._read_lines(3.0)
        elapsed = time.monotonic() - start
    finally:
        ev.proc.close()
    assert lines[-1].startswith("bestmove")
    assert elapsed < 1.0
