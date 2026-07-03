# MuZero warm start ~15 s/ply: perft read-loop timeout in PikafishEvaluator

**Date:** 2026-07-03
**Agent/Author:** Claude Code (Fable 5)

## 1. Hypothesis / Goal
First real MuZero run sat in the warm-start phase for 6+ hours with no output
after `[warmstart] generating Pikafish games ...`. The smoke run had also taken
6m05s for only ~24 engine plies. Hypothesis: not a hang — a fixed ~15 s cost
per ply somewhere in the engine plumbing.

## 2. Configuration Changes
- `src/pikafish_eval.py` `_read_lines`: early-exit now also triggers on
  `Nodes searched` (perft terminator), not just `bestmove`.
- `muzero/warmstart.py`: per-game progress print (plies, result, running total)
  so warm start is observable.
- New regression test `muzero/tests/test_pikafish_read_lines.py` (fake-pipe
  process; no engine binary needed).

## 3. Run Command
```bash
uv run pytest muzero/tests -q   # 44 passed, 5 skipped
```

## 4. Quantitative Results
- Root cause reproduced locally: pre-fix, a perft-shaped payload blocked
  `_read_lines(3.0)` for 3.005 s (full timeout); post-fix, <0.05 s.
- `go perft 1` output ends with `Nodes searched: N` — the old loop only broke
  on `bestmove`/EOF, so **every uncached `list_legal_moves` call burned the
  full `timeout_sec`** (15 s in the MuZero evaluator config; 120 s in the LLM
  v2 pipeline's config).
- `muzero/env.py:step` calls `legal_moves()` once per ply → ~15 s/ply →
  2,000-ply warm start ≈ 8 h; observed 6 h ≈ ply ~1,400. Smoke: ~24 plies ×
  ~15 s ≈ 6 min. Both observations match the mechanism.

## 5. Qualitative Outcome
- The 6-hour "stall" was steady progress at 100× intended cost. Self-play
  would have crawled identically (one legality call per MCTS root), so the fix
  was mandatory, not cosmetic.
- Side benefit: the LLM v2 pipeline's Pikafish legality calls
  (`pikafish/timeout_sec: 120`) stop burning their timeout too.
- `evaluate_cp`/`bestmove_and_root_cp` were never affected (they exit on
  `bestmove`).

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md`: unchanged (no interface/component change).
- `docs/AGENT_TODO.md`: unchanged — "first run on the training machine"
  remains Active; restart required after pulling this fix.
- Related: `docs/logs/2026-07-02-log-muzero-implementation.md` (flagged the
  no-timeout `_wait` in warmstart's SimpleUciEngine, but missed this shared
  evaluator path).

## 6b. Addendum (same day): first fix was one line too strict
The v1 early-exit checked `lines[-1]`, but real Pikafish perft output ends
with a **trailing blank line** (`"\nNodes searched: N\n"` + endl — confirmed
by piping `go perft 1` into the engine on the training box), so `lines[-1]`
was `""` and the loop still ran to timeout. The restarted run showed the same
~15 s/ply signature (engines near-idle: ~6 CPU-seconds over ~7 min). v2 fix
scans back to the last **non-empty** line; regression-test payload updated to
the real output shape (leading + trailing blanks). Lesson recorded: validate
fake-engine fixtures against actual engine output, not idealized output.

## 7. Conclusion & Next Steps
- Kill the in-flight run (its buffer is in-memory only; nothing worth keeping),
  `git pull` on the training machine, rerun `uv run python -m muzero.train`.
- Expected warm start: ~10–15 min for 2,000 plies, with per-game progress
  lines. If per-ply cost is still seconds, the next suspect is engine-side
  (NNUE file location / Threads contention), not this read loop.
