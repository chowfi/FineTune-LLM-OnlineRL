# Node-Limited Pikafish Gate Rung — Design

**Date:** 2026-07-13
**Status:** Approved (brainstorming session with user)
**Related:** `docs/superpowers/specs/2026-07-07-greedy-gate-rung-design.md` (rung
machinery this reuses), `docs/logs/2026-07-10-log-truncation-patience.md`
(experiment #2, whose plateau discussion motivated a graded upper rung)

## 1. Goal

The gate ladder has a dead rung and a cliff: `gate/*_random` has been pinned
at 1.00 for ~300 iterations (no information), and raw Pikafish at
`movetime 10` (~10–20k nodes/move) still wins ~99% of games — progress between
those extremes is invisible. Replace the random rung with a **node-limited
Pikafish rung** (`go nodes N`): a graded, machine-independent, extensible
strength dial sitting between greedy and full Pikafish.

**Why nodes, not UCI_Elo:** the current Pikafish build
(`dev-20260410-2ec20b9f`) exposes no `UCI_LimitStrength` / `UCI_Elo` /
`Skill Level` options (verified via `echo "uci" | $PIKAFISH_BIN`). Node
limiting needs no engine option — it is a `go`-command parameter — and each
halving of N costs the engine a roughly constant strength step. `nodes=1` is
raw evaluation with no lookahead.

**Non-goals:** human-Elo calibration (a node count is a dial, not a rating);
touching the arena; touching training rules (this is measurement-only);
a calibration sweep script (the running gate self-calibrates the dial —
decided during brainstorming).

## 2. Design

### 2a. Config (`muzero/config.py`)

- New: `gate_pika_nodes: int = 128` (~7 halvings below the full rung's
  effective node count — first guess at the model's current level).
- Comment documents the **bump protocol**: when `gate/win_rate_pika_nodes`
  stays above ~0.85 for 3+ consecutive gates, double the value and record the
  change in a dated log (each change redefines the metric; deliberate and
  documented, never silent). Symmetrically, halve it if the model scores
  below ~0.15 for 3+ gates.

### 2b. Engine wrapper (`muzero/warmstart.py`, `SimpleUciEngine`)

- New optional constructor arg `nodes: int | None = None` (default preserves
  current behavior exactly).
- `search()` sends `go nodes {self.nodes}` when `nodes` is set, else
  `go movetime {self.movetime_ms}` as today.
- Warmstart and the full-strength gate rung keep using movetime; no caller
  changes besides the new rung.

### 2c. Gate (`muzero/train.py`, `run_gate`)

- Rung order becomes: **greedy → pika-nodes → full Pikafish** (random rung
  removed; its `gate/*_random` metrics stop being emitted — history remains
  in wandb).
- The pika-nodes rung reuses the full Pikafish rung's machinery unchanged
  (same `_run_gate_rung` play path, same opening rotation, same
  illegal-move / engine-abort handling), with a second `SimpleUciEngine`
  instance constructed as
  `SimpleUciEngine(cfg.pikafish_bin, cfg.gate_movetime_ms, multipv=1,
  nodes=cfg.gate_pika_nodes)`.
- New metrics: `gate/win_rate_pika_nodes`, `gate/draw_rate_pika_nodes`,
  `gate/loss_rate_pika_nodes`, plus `gate/pika_nodes` = the dial value at
  gate time (keeps wandb charts interpretable across future bumps).
- Gate wall-clock stays ~unchanged (three rungs before and after; rung cost
  is dominated by the ally's own 800-sim search).

### 2d. Activation & expectations

- Takes effect on the next training restart. Measurement-only: no training
  rules change, so restarting mid-run does not contaminate experiment #2's
  banked state or the current plateau watch.
- First readings at nodes=128 are a coin flip — ~0.9 means the dial is too
  weak (apply the bump protocol, doubling once per confirming gate), ~0.1
  means too strong (halve likewise). Target band for an informative rung:
  30–70% win rate. Expect 2–3 gates to find the level.

## 3. Testing

- Update existing `run_gate` tests for the removed random rung and the new
  rung's metric keys (FakeEvaluator path, no engine needed).
- Unit test that `SimpleUciEngine(nodes=N)` sends `go nodes N` and that
  `nodes=None` still sends `go movetime` (fake stdin/stdout process, no
  engine).
- One engine-gated smoke test: `SimpleUciEngine(cfg.pikafish_bin,
  movetime_ms=10, multipv=1, nodes=8).search(startpos_fen)` returns a
  non-empty best move (skipped when `PIKAFISH_BIN` is absent, matching the
  existing engine-gated test convention).

## 4. Rollout checklist (box)

1. `git pull && uv run pytest muzero/tests -q` (engine-gated tests included).
2. Restart training at a natural point:
   `uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt`.
3. Watch the first 2–3 gates; apply the bump protocol if outside 30–70%.
4. Record the first readings and any dial change in a dated log.
