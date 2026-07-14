# Engine-Game Seeding — experiment #3

**Date:** 2026-07-14
**Agent/Author:** Claude (feat/engine-game-seeding)

## 1. Hypothesis / Goal

Arena Elo has been flat across iters 380–480 (successive top-up readings
769/708/688/771/750/617 — net ~0 movement over 100 iterations). Independently,
the user has observed two recurring weaknesses playing the checkpoint at the
board: (a) the agent continues executing a plan after it has already been
refuted by the opponent's reply, and (b) aimless quiet moves and unconverted
endgames when the opponent puts up resistance rather than folding immediately.

Root-cause hypothesis: a **self-play echo chamber**. Because both sides of
every training game are played by the same (or a frozen copy of the same)
network, the opponent shares the ally's blind spots — it doesn't reliably
punish refuted plans or defend endgames the way an external, stronger source
of moves would, so the network never learns to correct for those failure
modes. See spec §1 (`docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md`)
for the full writeup.

**Goal:** inject a small, steady stream of external (engine) game data into
the replay buffer every iteration, cheaply enough not to dominate training
time, to break the echo chamber without changing the self-play pipeline
itself.

## 2. Configuration Changes

- `seed_games_per_loop`: 0 (implicit, no seeding existed) → **4** (~5% of an
  84-game self-play loop) — `muzero/config.py`.
- `muzero/warmstart.py`: extracted the inline warmstart game loop into a
  shared `play_engine_game(cfg, engine, evaluator, rng)`, with move selection
  now going through a new pure `_pick_move_index(lines, ply, temperature_moves,
  rng)` helper. Teacher engine is warmstart-strength Pikafish (50 ms/move,
  MultiPV 4). For the first `temperature_moves` (=30) plies, the played move
  is sampled from the softmax over the MultiPV centipawn scores
  (`_multipv_probs`) — for variety, so seeded games don't all repeat the same
  opening line. From ply 30 onward the engine's best line is always played.
  The policy *target* stored in the buffer is unaffected by which move is
  played — it is always the full MultiPV softmax distribution (`_play_move`
  unchanged) — so only the played-move variety changes, not the target
  quality.
  - **Deliberate side effect:** warmstart itself (`generate_warmstart_games`,
    cold-start only) now also samples early moves via the same shared helper,
    instead of always playing the engine's best line. This was accepted in
    the spec — variety is harmless-to-helpful for warmstart data too, and one
    code path is simpler than maintaining two.
- New `generate_seed_games(cfg, buffer, evaluator, n_games, rng)` in
  `muzero/warmstart.py`: plays `n_games` engine-vs-engine games via
  `play_engine_game` and adds each to the buffer through the identical
  `ReplayBuffer.add` path self-play games use — same env, referee, rewards,
  and target computation. `n_games <= 0` is a no-op (engine never
  constructed).
- New `seed_engine_games(cfg, buffer, evaluator, iteration)` in
  `muzero/train.py`, wired into the main training loop directly after
  `aggregate_game_summaries`. Never raises: an engine outage during seeding
  is caught, logged, and degrades that iteration to self-play-only
  (`buffer/seeded_games=0.0`) rather than crashing the run. Per-iteration RNG
  is `cfg.seed + iteration` — reproducible without replaying the same sampled
  engine games every loop.
- `metrics["buffer/seeded_games"]` now logged every iteration.

## 3. Run Command

Local (dev box, this session): `uv run ruff check muzero --fix && uv run
ruff format muzero`, `uv run pytest muzero/tests -q`.

Not run for real training in this session — this is the docs/handoff task
(Task 5) after Tasks 1–4 implemented the feature. Rollout to the training box
is a separate, pending step (see §6/§7 and the new AGENT_TODO.md bullet).

## 4. Quantitative Results

Pre-change baseline (for comparison once seeding is live):
- `gate/*_greedy` band: ~0.45–0.80 (gates 13–16).
- `gate/*_pika_nodes`: rung baseline TBD from its first post-2026-07-13
  readings (the node-limited-pika dial may still be calibrating — see the
  gate-reading caveat below).
- Arena: ~700–770, flat across iters 380–480.
- Mate-win rate: ~0.20.

Post-change results: **fill in as data arrives.**

## 5. Qualitative Outcome

Not yet observed in play — pending restart + several gates/arena top-ups on
the training box. Expected mechanical shifts once seeding is active (these
are mechanical predictions, not success/failure verdicts in themselves):
- `loss/policy` may drop somewhat — engine MultiPV softmax targets are
  sharper than self-play's own policy targets on the same fraction of
  positions.
- `buffer/seeded_games` ≈ 4 every iteration once running.
- Wall-clock cost: roughly +25 s/iteration for the 4 engine games at
  warmstart-strength settings (50 ms/move, MultiPV 4).

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md`: updated the MuZero Xiangqi (§3f) component
  description — added a passage on `warmstart.py`'s shared `play_engine_game`,
  `generate_seed_games`, `seed_games_per_loop=4`, the `buffer/seeded_games`
  metric, and the note that `selfplay/*` metrics stay model-play-only because
  seeded games bypass the self-play summary path.
- `docs/AGENT_TODO.md`: appended a "DECIDED 2026-07-14" resolution note to
  the 2026-07-12 "decide experiment #3" bullet, and added a new first bullet
  under Active Tasks: "MuZero: engine-game seeding (experiment #3) — restart
  to activate," with the rollout checklist.
- Related logs/docs: spec `docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md`,
  plan `docs/superpowers/plans/2026-07-14-engine-game-seeding.md`.

## 7. Conclusion & Next Steps

Implementation (Tasks 1–4) is committed: config knob, shared
`play_engine_game` + `_pick_move_index`, `generate_seed_games`, and
`seed_engine_games` wired into the training loop. This log (Task 5) closes
out lint/docs/handoff accounting for the change. The feature is **not yet
active on the training box** — it takes effect on the next process restart
(`--resume`), not from scratch.

**Pre-registered SUCCESS criteria** (judge after 5+ gates and the next two
arena top-ups):
- Both engine-gate bands (`gate/*_greedy` and `gate/*_pika_nodes`) trending
  up over 5+ gates.
- Arena slope positive across iters ~500–560 (vs the flat 380–480 baseline).
- Subjectively fewer refuted-plan incidents at the board.

**Pre-registered REVERT criteria:** set `seed_games_per_loop: 0` and restart
if either:
- `loss/value` or `value_cp_correlation` degrade in a sustained way (~15+
  iterations beyond ordinary churn), or
- either engine gate (`gate/*_greedy` or `gate/*_pika_nodes`) falls below its
  pre-change band for 3+ consecutive gates.

**Gate-reading caveat:** interpret `gate/*_pika_nodes` readings against the
logged `gate/pika_nodes` dial value at the time — the node-limited rung may
still be mid-calibration (double/halve protocol, 2026-07-13), so judge trends
within a fixed dial setting, not across dial changes.

**Rollout (on the training box):**
```bash
git pull && uv run pytest muzero/tests -q
# stop the running training process
uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt
```
Confirm `buffer/seeded_games=4` appears in the next iteration's log line.
This is a **process restart, not a fresh run from scratch** — the checkpoint
and buffer state carry over via `--resume`.

**Two known-minor follow-ups from review** (non-blocking):
- No local (non-engine-gated) test exercises `_pick_move_index`'s wiring
  through the full `play_engine_game` loop with more than one candidate
  line — the sampling-distribution behavior of `_pick_move_index` itself is
  unit-tested directly, and the multi-candidate path through the game loop is
  only covered by the engine-gated tests (which run on the box, not in this
  sandbox).
- `_multipv_probs` is computed twice per sampled ply (once for the played-move
  sample, once for the policy target inside `_play_move`) — deterministic and
  harmless, just a small redundant computation, not correctness-affecting.
