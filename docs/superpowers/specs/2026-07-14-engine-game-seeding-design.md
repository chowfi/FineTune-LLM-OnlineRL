# Engine-Game Seeding — Design (experiment #3)

**Date:** 2026-07-14
**Status:** Approved (brainstorming session with user)
**Related:** `muzero/warmstart.py` (machinery this generalizes),
`docs/logs/2026-07-10-log-truncation-patience.md` (experiment #2, banked;
its arena plateau at iters 380–480 motivates this),
`docs/superpowers/specs/2026-07-13-node-limited-pika-rung-design.md`
(instrument that will help judge this).

## 1. Goal

The arena has been flat for ~100 iterations (iters 380–480: 769→617 within
noise) and the user's board observations identify two persistent weaknesses:
(a) the model continues a plan after the opponent has refuted it (its
self-play opponent shares its blind spots, so refutation patterns are absent
from training data), and (b) aimless quiet moves and unconverted winning
endgames against real resistance. Both are self-play echo-chamber problems.

**The change:** every training iteration, in addition to the ~84 self-play
games, generate a small number of Pikafish-vs-Pikafish games through the
identical environment/target pipeline and add them to the replay buffer —
a permanent trickle of expert demonstrations (unlike warmstart, whose games
washed out of the 1500-game buffer within ~18 iterations of startup).

**Non-goals:** endgame-start curriculum (separate candidate, possibly
experiment #4); changing self-play, rewards, or the referee; reanalyze;
raising the dose beyond the initial 5% (a follow-up experiment if banked).

## 2. Decisions made during brainstorming

- **Dose: `seed_games_per_loop = 4`** (~5% of an 84-game loop). Conservative
  first dose; raising it later is its own experiment.
- **Teacher: warmstart-strength engine** (`warmstart_movetime_ms = 50`,
  `warmstart_multipv = 4`) — no new strength dials.
- **Variety: temperature-style sampling.** For plies ≤ `temperature_moves`
  (30, the same constant self-play uses), the PLAYED move is sampled from
  the engine's top-4 multipv moves weighted by the existing softmax over
  centipawn scores (the same `probs` array already computed for the policy
  target); after ply 30 the engine's best move is always played. Without
  this, the deterministic engine would replay ~10 identical games (one per
  opening-book line) forever. Warmstart adopts the same sampling (variety
  is harmless-to-helpful there; no separate code path).

## 3. Design

### 3a. Config (`muzero/config.py`)

- New: `seed_games_per_loop: int = 4` in the self-play block. `0` disables
  seeding entirely (and is the revert setting). Comment documents the
  experiment (date, rationale, dose), the sampling scheme, and the revert
  criteria from §5.

### 3b. Generation (`muzero/warmstart.py`)

- Generalize the existing warmstart game loop into a shared helper (e.g.
  `play_engine_game(cfg, engine, evaluator, rng) -> GameHistory`) used by
  both `generate_warmstart_games` (loops until `warmstart_plies`) and a new
  `generate_seed_games(cfg, buffer, evaluator, n_games, rng) -> dict`
  (loops exactly `n_games`, returns `{"games": n, "plies": total}`).
- The helper contains the one new behavior: at plies ≤ `temperature_moves`,
  sample the played move from the multipv softmax distribution using the
  provided seeded rng; afterwards play `lines[0]`. The forced opening-book
  first ply is unchanged.
- Training records are byte-compatible with today's warmstart records:
  same policy targets (multipv softmax), same root values (tanh(cp/600)),
  same env (referee, rewards, truncation, repetition rules), same
  `buffer.add(history)`. Nothing downstream changes.
- `generate_seed_games` creates and closes its own `SimpleUciEngine` per
  call (a few games; construction cost is negligible against 50ms/move
  search, and it avoids holding a subprocess open across the whole loop).

### 3c. Training loop (`muzero/train.py`)

- Each iteration, immediately after self-play generation and before
  training, when `cfg.seed_games_per_loop > 0`: call
  `generate_seed_games(cfg, buffer, evaluator, cfg.seed_games_per_loop, rng)`
  with a seeded rng that varies per iteration (e.g. seeded from
  `cfg.seed + iteration` — reproducible but not repeating).
- Engine outage must not kill training: wrap the call in try/except
  (Exception), print an operator-visible warning, and continue the
  iteration as self-play-only.
- New metric logged every iteration: `buffer/seeded_games` (games actually
  seeded this iteration — 0 on outage or when disabled).
- Metric purity falls out automatically: seeded games go through
  `buffer.add` directly and never produce self-play summaries, so all
  `selfplay/*` metrics keep measuring ONLY the model's own play.

### 3d. Cost

~4 games × ~120 plies × 50 ms ≈ 25 s engine time per iteration (~2% of a
~20–30 min iteration). No GPU cost.

## 4. Interpretation & instruments

- **Success:** greedy-gate and pika-nodes-gate bands trending up over 5+
  post-change gates; arena slope positive again across iters ~500–560;
  subjectively fewer "continues the refuted plan" incidents at the board.
- **Expected mechanical shifts (not verdicts):** `loss/policy` may DROP
  (engine policy targets are sharper than search-visit targets);
  `buffer/mean_game_length` may shift slightly; `buffer/seeded_games` ≈ 4.
- **Watch/revert:** set `seed_games_per_loop: 0` if `loss/value` climbs or
  `value_cp_correlation` degrades sustained (~15+ iters beyond normal
  churn — the value-target-mismatch failure mode: outcomes between two
  Pikafishes mislabel positions the model itself couldn't hold), or if
  either engine-gate band falls clearly below its pre-change band for 3+
  gates. Weights keep everything learned; only the data mix reverts.
- **Caveat for reading gates:** this restart may coincide with the
  node-limited rung's dial calibration — interpret `gate/*_pika_nodes`
  levels against `gate/pika_nodes` (the logged dial), and judge trends
  within a fixed dial setting.

## 5. Testing

- Unit (FakeEngine / no engine): sampling rule — ply ≤ temperature_moves
  samples from the multipv distribution (seeded rng ⇒ reproducible, and a
  distribution check that non-best moves are actually played sometimes);
  ply > temperature_moves always plays lines[0].
- Unit: `generate_seed_games` plays exactly n_games, adds each to the
  buffer, returns the stats dict; n_games=0 ⇒ no engine constructed.
- Unit: training-loop wiring — seeder exceptions are swallowed with a
  warning and `buffer/seeded_games` logs 0; success path logs 4.
- Regression: warmstart still fills the buffer to `warmstart_plies`
  (existing engine-gated test keeps passing).
- One engine-gated smoke: `generate_seed_games` with n_games=1 on the real
  binary produces a valid GameHistory (skipped without `PIKAFISH_BIN`).

## 6. Rollout (box)

1. `git pull && uv run pytest muzero/tests -q`.
2. Stop training; `uv run python -m muzero.train --resume
   checkpoints/muzero_xiangqi/latest.pt` (weights/buffer carry over — this
   is a process restart, not training from scratch).
3. Confirm `buffer/seeded_games=4` in the next iteration's log line.
4. Judge per §4 after 5+ gates / next two arena top-ups; record readings
   in the experiment log; revert = config 0 + restart.
