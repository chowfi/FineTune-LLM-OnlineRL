# Greedy Gate Rung + Checkpoint Elo Arena

**Date:** 2026-07-07
**Agent/Author:** Claude Code (subagent-driven development)

## 1. Hypothesis / Goal

The training run's play-strength metrics plateaued (~iter 48–140: blunder rate
flat at ~0.29, mate rate flat) while world-model losses kept improving, and
the gate's random rung saturated at 100% — strength changes were invisible.
Build two instruments before touching any training knob: (1) a capture-greedy
gate opponent measuring exactly the user-observed weakness ("doesn't punish
hanging pieces"), (2) a checkpoint Elo curve that never saturates. Specs:
`docs/superpowers/specs/2026-07-07-greedy-gate-rung-design.md`,
`docs/superpowers/specs/2026-07-07-checkpoint-elo-arena-design.md`.

## 2. Configuration Changes

Branch `muzero-gate-arena` (8 implementation commits 1a10da7…b1a23d8 + 1
docs commit, 9 total):

- `muzero/gate_opponents.py` (new) — `greedy_capture_move(env, rng)`:
  highest-value capture (king override 100), random tie-break/fallback;
  8 unit tests incl. type + determinism pins.
- `muzero/train.py` `run_gate` — three rungs (random / greedy via
  `cfg.seed + 1` rng / Pikafish); new metrics `gate/win|draw|loss_rate_greedy`
  and `gate/seconds`. Honest cost note: each rung's cost is dominated by the
  ally's own 800-sim search, so 3 rungs ≈ +50% gate time (~5% throughput at
  gate_every_loops=10) — accepted, observable via gate/seconds.
- `muzero/train.py` `maybe_archive_checkpoint` + `checkpoint_archive_every=20`
  (config) — ally-weights snapshots to `checkpoints/muzero_xiangqi/archive/`;
  never overwrites (resume-from-rollback protection), non-fatal on OSError,
  operator log line.
- `muzero/arena.py` (new) — `python -m muzero.arena`: numeric-sorted
  checkpoint discovery (extras auto-relabeled `iter_NNNN` + same-iteration
  dedupe), adjacent-pair matches on the exact gate play path (openings ×
  colors; `start` offset continues rotation on top-ups), append-only
  `data/arena/games.jsonl` (gitignored), BT fit via
  `scripts/benchmark/elo_estimator.fit_ratings` (oldest anchored 0, anchor
  identifiability guard), table + `data/arena/ratings.json`.
- Tests: 96 passed, 6 skipped (was 80/6 before the branch).

## 3. Run Command

```bash
uv run pytest muzero/tests web/tests -q   # 96 passed, 6 skipped
uv run python -m muzero.arena --help
```

## 4. Quantitative Results

- Synthetic Elo fit check: 75% score → +190.8 (theory ≈ +191). ✓
- Review catches (subagent two-stage review): three arena data-poisoning bugs
  (dead-Pikafish games minted permanent fake draws; `--extra latest.pt`
  label identity drifts across runs corrupting the fit; deterministic
  top-ups replayed identical games inflating confidence), archive clobber on
  resume-from-rollback, unguarded archive write that could kill training,
  and a misleading gate-cost estimate in the spec (corrected).

## 5. Qualitative Outcome

- Takes effect on the next training restart (`--resume`, no from-scratch).
- Operational notes: if an arena games.jsonl predating commit b1a23d8 ever
  exists, prune rows labeled by non-canonical stems (e.g. "latest") once;
  archive grows ~2.5 GiB/week with no pruning (TODO backlog).

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md` §3f — three-rung ladder, archiving, arena.
- `docs/AGENT_TODO.md` — new Active restart-and-baseline task (2–3 greedy
  gate readings BEFORE changing one knob: shaping_weight 0.3→0.1 or
  truncation_consecutive 6→12); arena/archive follow-ups in Backlog;
  Completed entry.

## 7. Conclusion & Next Steps

Restart on the box (`--resume`), watch `gate/win_rate_greedy` (expect
~0.6–0.85 initially; the climb is the tactics progress bar), let 2–3 gates
accumulate as the baseline, run the first arena once ≥2 rating points exist
(archives + `--extra latest.pt`), then change exactly one training knob and
attribute the effect with both instruments. NOTE: `iter80-prebufferfix.pt`
is from the PRE-canonicalization run (115-plane) — it cannot load into the
arena and does not belong on this run's curve; the curve starts with the
canonical run's own archives.
