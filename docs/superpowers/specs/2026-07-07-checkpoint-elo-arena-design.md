# Checkpoint Archive + Elo Arena — Design

**Date:** 2026-07-07
**Status:** Approved (brainstorming session with user)
**Related:** `docs/superpowers/specs/2026-07-07-greedy-gate-rung-design.md`
(built in the same round), `scripts/benchmark/elo_estimator.py` (reused fitter)

## 1. Goal

Give the training run a saturation-proof strength measurement: a relative Elo
curve over training time, computed by playing archived checkpoints against
each other. Answers "is the model actually stronger than last week, and by
how much" — the question blunder rate (symptom metric) and the gate rungs
(fixed outsiders that saturate) cannot answer.

**Facts constraining the design:** training currently saves only `latest.pt`
(overwritten every iteration). Historical checkpoints do not exist except the
manual `iter80-prebufferfix.pt` backup and the current latest. The curve
therefore starts with those two points and grows forward.

**Non-goals:** absolute Elo vs humans/engines; wiring the arena into the
training loop (offline tool, run on demand); GUI.

## 2. Design

### 2a. Checkpoint archiving (training loop, `muzero/train.py`)

- New config: `checkpoint_archive_every: int = 20` (`muzero/config.py`).
- At the end of each iteration where `(it + 1) % checkpoint_archive_every == 0`,
  additionally save `checkpoints/muzero_xiangqi/archive/iter_{it+1:04d}.pt`
  containing `{"ally": ally.state_dict(), "iteration": it + 1}` — network
  weights only (~90 MB; no optimizer/enemy state; loadable by
  `web/server/muzero_player.py` and the arena). Written atomically
  (tmp + `os.replace`) like `latest.pt`. Directory created on startup.
- Takes effect on the next training restart (same restart as the greedy rung).

### 2b. Arena (`muzero/arena.py`, run as `python -m muzero.arena`)

- **Checkpoint discovery:** all `archive/iter_*.pt` files, sorted by
  iteration, plus optional extra checkpoints via `--extra path[:label]`
  (repeatable) so `iter80-prebufferfix.pt` and `latest.pt` can join the pool.
  Player labels = `iter_0080`-style names.
- **Pairing:** adjacent checkpoints only (sorted order) — linear cost, and
  the Bradley–Terry chain composes ratings transitively.
- **Games per pair:** `--games-per-pair` (default 20) = the 10 book openings
  × both color assignments. Both players use the exact gate play path
  (`canonical_root` → noiseless argmax MCTS → `absolute_visits`), one
  `MuZeroPlayer`-style net each; the forced book opening supplies diversity
  (deterministic players would otherwise repeat one game). `--sims`
  (default 800) applies to BOTH players equally — lower it for cheaper runs;
  relative ratings stay comparable only within one arena run's sims setting.
- **Game rules:** `muzero/env.XiangqiEnv` with the standard training config
  (repetition draws, 300-ply cap, cp-adjudication ACTIVE as in training —
  arena games should end decisively like training games; requires
  `PIKAFISH_BIN` for legality/eval, same as the gate).
- **Output:** appends per-game rows to
  `data/arena/games.jsonl` (`{"white": label, "black": label, "result":
  "white"|"black"|"draw", "sims": N, "timestamp": ...}`) so repeated runs
  accumulate; skips pairs that already have `>= games-per-pair` recorded
  games at the current sims (idempotent re-runs as the archive grows).
- **Elo fit:** reuse `scripts/benchmark/elo_estimator.fit_ratings` (verify
  the result-string convention against `_result_signs` at implementation
  time), anchoring the OLDEST checkpoint at 0. Print a table
  (checkpoint, iteration, Elo, games played) and write
  `data/arena/ratings.json`.

### 2c. Interpretation caveats (documented in tool output)

- ~20 games/pair ⇒ roughly ±80 Elo per step; read the curve's shape across
  several checkpoints, not neighbor differences. `--games-per-pair 60`
  for tighter error bars.
- Ratings are relative (oldest = 0) and only comparable within a consistent
  `--sims` setting.

## 3. Testing

- Unit (no engine, FakeEvaluator + tiny nets): archive naming/discovery and
  sorting; pairing logic (adjacent only, respects already-played games in
  the jsonl); a 2-checkpoint mini-arena with tiny nets and 2 sims produces
  valid jsonl rows and a ratings table with the oldest anchored at 0;
  fit integration on synthetic scoresheets (a player winning 3:1 rates
  ~+190 ± tolerance).
- Archiving: `--smoke`-config test that a fake training loop tick writes
  `archive/iter_XXXX.pt` with the expected keys.
- Real engine-gated arena run happens on the box as part of the rollout
  checklist.
