# Training-health metrics: blunder rate, mate-win rate, search gain

**Date:** 2026-07-04
**Agent/Author:** Claude Code (Fable 5)

## 1. Hypothesis / Goal
Add three free training-health metrics (no extra engine calls) requested after
reviewing the first latest-mode run (iters 0–11): win rates are noisy 84-game
samples amplified by self-play oscillation; these measure learning more
directly.

## 2. Configuration Changes
- `muzero/config.py`: `blunder_cp_threshold: float = 200.0`.
- `muzero/env.py`: `step` info now includes `mover_cp_delta` (mover-perspective
  eval change across the mover's own move; shaping refactored to share it).
- `muzero/mcts.py`: `MCTS.run` returns `(visits, root_value, search_kl)` per
  game, where `search_kl = KL(visit distribution || raw pre-noise prior)`.
- `muzero/selfplay.py`: per-game `blunders`/`cp_moves`/`search_kls`
  bookkeeping; summaries carry them.
- `muzero/metrics.py`: `selfplay/blunder_rate`, `selfplay/mate_win_rate`
  (wins by checkmate, adjudicated wins excluded), `selfplay/mean_search_kl`.
- `muzero/train.py`: `run_gate` unpack updated for the 3-tuple.

## 3. Run Command
```bash
uv run pytest muzero/tests -q   # 57 passed, 5 skipped
```

## 4. Quantitative Results
Suite 54 → 57 passed. No training run in this session.

## 5. Qualitative Outcome
Expected reads: `blunder_rate` should decline steadily and monotonically if
learning is real (better early signal than win rates); `mate_win_rate` rising
from ~0 is the "learned to convert" milestone predicting repetition-draw
decline; `mean_search_kl` drifting down = policy internalizing search (the
MuZero improvement operator working). Forced opening moves carry no search KL
(excluded) but do count toward blunder stats (~1 forced move per game,
constant across iterations, trend-neutral).

Also this session (no code change): diagnosed the 2026-07-04 run "crash" as a
terminal-refresh SIGHUP — no traceback in wandb logs (an exception would have
printed one), timeline matches the SSH refresh. Remedy: run inside
`tmux new -s muzero`, detach with Ctrl-B D; resume with `--resume`.

## 6. Repo / Handoff Updates
- `docs/AGENT_TODO.md`: backlog item for §10 deferred metrics trimmed (blunder
  rate/search-gain now done; engine-best agreement still deferred).
- `docs/ARCHITECTURE.md`: §3f metric list already generic; no change needed.

## 7. Conclusion & Next Steps
Pull on the GPU box (inside tmux), resume the run, and watch the three new
`selfplay/` charts alongside `value_cp_correlation` (check at iter ~30).
