# Feature / Experiment Log Template

**Date:** 2026-07-13
**Agent/Author:** Claude (muzero)

## 1. Hypothesis / Goal
The gate ladder had a dead rung and a cliff. `gate/*_random` (uniform-random
legal mover) had been pinned at 1.0 for ~300 iterations (since roughly iter
30) and stopped conveying any signal. Meanwhile the top rung, raw Pikafish at
`gate_movetime_ms`, sits at the other extreme: ~99% losses this era, with
only 3 draws across 320 games. Between "trivial" and "crushing" there was no
gradient — no rung to show gradual strength gains as the ally improves.
Goal: replace the dead random rung with a **node-limited Pikafish** rung
that is graded and extensible, giving a middle instrument the ladder
currently lacks. Pikafish's `UCI_Elo`/`Skill Level` options are unavailable
in this build — verified via `echo "uci" | $PIKAFISH_BIN` (no strength
options listed) — so `go nodes N` (a hard search-node cap) is the dial
instead.

## 2. Configuration Changes
- `gate_pika_nodes: int = 128` (new config field, `muzero/config.py`) — the
  new mid-rung's Pikafish search-node cap. 128 is ~7 halvings below the top
  rung's effective node count (full Pikafish at `gate_movetime_ms=10ms`
  runs roughly ~10-20k nodes on this hardware), so it starts well inside
  the "beatable" end of the range.
- Bump protocol (comment in `config.py`, next to `gate_pika_nodes`): if
  `gate/win_rate_pika_nodes` > ~0.85 for 3+ consecutive gates, **double**
  `gate_pika_nodes`; if < ~0.15 for 3+ consecutive gates, **halve** it.
  Every change must be logged in a dated log — it redefines what the metric
  means, so it is never a silent edit.
- `grpo/beta`: not applicable (MuZero side, not GRPO).

## 3. Run Command
Not run this session — this is a measurement-only code/docs change on top
of an already-committed feature branch (`feat/pika-nodes-rung`). It takes
effect only once training is restarted with `--resume` on the training box
(see rollout checklist below); no run was executed here.
```bash
# On the training box, once ready to activate:
git pull && uv run pytest muzero/tests -q
uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt
```

## 4. Quantitative Results
- `uv run ruff check muzero --fix && uv run ruff format muzero`: all checks
  passed, 32 files left unchanged (nothing to reformat).
- `uv run pytest muzero/tests -q`: **89 passed, 7 skipped** (engine-gated
  tests skip without `PIKAFISH_BIN`), matching the expected baseline.
- No new gate readings yet — this rung has not been exercised against a
  live checkpoint. First readings are expected to land anywhere in [0, 1]
  since the starting `gate_pika_nodes=128` is an untuned guess; the target
  band is 30-70% win rate, and 2-3 gates should be enough to locate a
  reasonable level via the bump protocol.

## 5. Qualitative Outcome
- Code changes (Tasks 1-3, already committed on this branch before this
  session): `SimpleUciEngine` gained a `nodes=` mode (`muzero/warmstart.py`)
  that issues `go nodes N` instead of `go movetime N`; a subprocess-leak fix
  was found in review and applied alongside it — if the UCI handshake fails,
  the spawned engine process is now killed rather than left orphaned.
  `run_gate` (`muzero/train.py`) swaps the retired random rung for the new
  node-limited Pikafish rung, opens a second nested `SimpleUciEngine`
  (`weak_engine`) alongside the existing full-strength one, and closes both
  in a `finally` block. New metric keys: `gate/win_rate_pika_nodes`,
  `gate/draw_rate_pika_nodes`, `gate/loss_rate_pika_nodes`, and
  `gate/pika_nodes` (the current dial value, logged every gate so the
  bump-protocol history is reconstructable from wandb alone).
- This session (Task 4): lint pass (clean), `docs/ARCHITECTURE.md` gate
  ladder description updated, this log, and an `docs/AGENT_TODO.md` entry
  for the on-box rollout.
- The change is measurement-only and safe to land mid-experiment (does not
  touch training/gradient code); it only activates the next time training
  is restarted from a checkpoint.

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md`: updated the §3c gate-ladder sentence to describe
  the ladder as capture-greedy -> `gate/*_pika_nodes` (graded dial, bump
  protocol in `config.py`) -> full Pikafish at `gate_movetime_ms`, and
  noted the uniform-random rung's 2026-07-13 retirement (saturated at 1.0
  since ~iter 30; history remains in wandb).
- `docs/AGENT_TODO.md`: added an Active Tasks entry (first item) for the
  on-box restart-to-activate rollout, with the exact commands and the
  30-70% watch band.
- Related logs/docs: spec `docs/superpowers/specs/2026-07-13-node-limited-pika-rung-design.md`;
  plan `docs/superpowers/plans/2026-07-13-node-limited-pika-rung.md`.

## 7. Conclusion & Next Steps
- Not yet validated against a live run — the next agent (on the training
  box) should follow the rollout checklist: `git pull && uv run pytest
  muzero/tests -q`, stop training, restart with `--resume
  checkpoints/muzero_xiangqi/latest.pt`, then watch the first 2-3
  `gate/win_rate_pika_nodes` readings.
- Apply the double/halve bump protocol immediately if those readings sit
  outside the 30-70% band for 3+ consecutive gates, and log the change
  (new dated log) since it redefines the metric.
- This is measurement-only and does not disturb experiment #2's banked
  state (`truncation_consecutive` 6 -> 12, VERDICT: SUCCESS, see
  `docs/logs/2026-07-10-log-truncation-patience.md`).
