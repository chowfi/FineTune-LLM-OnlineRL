# Capture-Greedy Gate Rung — Design

**Date:** 2026-07-07
**Status:** Approved (brainstorming session with user)
**Related:** `docs/logs/2026-07-05-log-metrics-audit-and-buffer-fix.md` (introduced the
gate ladder), `muzero/train.py` (`run_gate`/`_run_gate_rung`)

## 1. Goal

The gate ladder's random rung has been saturated at 100% since iter 29 and the
Pikafish rung will read 0% for a long time — strength changes are currently
invisible. Add a middle rung: a **capture-greedy opponent** (always takes the
highest-value hanging piece, otherwise plays a random legal move). Its win
rate directly measures the model's tactical progress ("does it punish hanging
pieces / stop hanging its own"), which is exactly the weakness observed in
human play, and it becomes the before/after instrument for any upcoming
training-knob change (shaping weight, truncation adjudication).

**Non-goals:** checkpoint-vs-checkpoint Elo (separate follow-up); weakened-
Pikafish rungs; any change to training itself. Measurement only — the run can
be resumed with `--resume` and no learning behavior changes.

## 2. Design

### 2a. `muzero/gate_opponents.py` (new, ~25 lines)

```python
def greedy_capture_move(env, rng) -> str | None
```

- Enumerate `env.legal_moves()` (internal algebraic, from Pikafish legality —
  same source the gate already uses). If empty, return `None` (the rung's
  game loop treats `None` as "opponent aborted", ending the game, matching
  the existing engine-rung contract).
- For each move, decode the destination square via
  `algebraic_to_board_coords` and look up the captured piece's value on
  `env.board` using `PIECE_TYPE`/`PIECE_VALUE` from `muzero/encoding.py`
  (rook 9.0 > cannon 4.5 > horse 4.0 > elephant/advisor 2.0 > pawn 1.0;
  empty square = no capture).
- If any capture exists: return one of the maximum-value captures (ties
  broken by `rng.choice` over the tied moves). Otherwise: uniform
  `rng.choice` over all legal moves.
- Pure function; the caller supplies a seeded `np.random.Generator` so gate
  results are reproducible.

### 2b. `run_gate` in `muzero/train.py`

- Add a third rung between random and Pikafish, reusing `_run_gate_rung`
  unchanged: opponent callable closes over a seeded rng
  (`np.random.default_rng(cfg.seed)` — a separate generator from the random
  rung's so adding the rung does not perturb the random rung's move
  sequence).
- New metrics: `gate/win_rate_greedy`, `gate/draw_rate_greedy`,
  `gate/loss_rate_greedy`. Existing keys unchanged.
- Cost: +`gate_games` (20) games per gate. The dominant cost of every rung
  is the ALLY's own 800-sim search (~half the plies), so a third rung adds
  ~50% to total gate time (~5% of overall training throughput at
  `gate_every_loops=10`). Accepted trade — this rung is the primary
  strength instrument. `gate/seconds` makes the cost observable.

### 2c. Rollout

- Branch → tests → merge, per repo conventions. Takes effect on the next
  training restart (`--resume checkpoints/muzero_xiangqi/latest.pt`;
  weights/optimizer kept, buffer refills as usual). No from-scratch
  retraining — no network-shape change.
- Expected initial reading: ~60–85% wins vs greedy; the climb toward ~100%
  is the tactical progress bar. Collect 2–3 gate readings as a baseline
  before changing any training knob.

## 3. Testing

Unit tests (FakeEvaluator, no engine, in `muzero/tests/test_gate_opponents.py`):
1. Prefers the highest-value capture (rook over pawn) when both hang.
2. Ties between equal captures resolve within the tied set (seeded rng).
3. No captures available → returns some legal move (seeded, deterministic).
4. Empty legal list → `None`.
5. Values read from the ABSOLUTE board correctly for both colors (a black
   greedy opponent captures red pieces — sign handling via `abs()`).

The existing engine-gated `run_gate` path is exercised on the training box as
part of the restart checklist (three rungs print in gate metrics).
