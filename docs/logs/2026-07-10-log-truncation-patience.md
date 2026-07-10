# Truncation Patience (6 → 12) — experiment #2

**Date:** 2026-07-10
**Agent/Author:** Claude Code

## 1. Hypothesis / Goal

Experiment #1 (shaping 0.3→0.1) is banked: greedy gate moved 0.56 → ~0.74
(11 post-change readings, iters 219–319, incl. a first-ever 20/20 at 309) and
the arena showed a slope-break (+100 Elo across 220→260 after a flat
180→220); gains then consolidated — 11 readings with no further trend =
the agreed flattening signal. Remaining top weakness (user-observed at the
board + mate rate stuck ~0.10): aimless endgames. Cause: ~70% of decisive
self-play games are adjudicated (loser ≤ −800cp for 6 consecutive own-moves)
— the model rarely plays conversions. Doubling the referee's patience to 12
forces winners to finish more games: direct endgame practice, more genuine
mates in the training data. Single-knob discipline: nothing else changes.

## 2. Configuration Changes

- `muzero/config.py`: `truncation_consecutive` 6 → 12. No other change.
- Side effects to expect (mechanical, not verdicts): `truncation_rate` down,
  `mean_plies` up (~+10–15), iteration wall-clock slightly up, the
  replay-buffer truncated-tail down-weight window doubles (derived as
  2×consecutive), and `mate_win_rate` becomes both more honest and the
  primary success metric.
- **Instrument caveat:** gate and arena games use the same env rules, so
  post-change gate/arena rows are played under the stingier referee. A few
  formerly-adjudicated wins may become draws/cap-outs, which could shave the
  measured greedy win rate slightly WITHOUT a strength change — judge this
  experiment primarily by mate rate and the Elo curve's slope, and compare
  greedy readings against a fresh post-change baseline rather than the old
  band.

## 3. Run Command

```bash
# box: git pull && uv run pytest muzero/tests -q
# stop training, then:
uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt
```

## 4. Quantitative Results (fill in as data arrives)

- Pre-change baseline: mate rate ~0.10 (spikes 0.179–0.202); greedy gate
  band 0.50–1.00 (mean 0.74); truncation_rate ~0.72; mean_plies ~100;
  Elo +180 at iter 260 (oldest-anchored); blunder ~0.25.
- Post-change (iters 322–337, first 16 iters):
  - truncation_rate ~0.55 (from ~0.72) and mean_plies ~119 (from ~100) —
    both predicted mechanical shifts landed.
  - Gate 1 of 3 (iter 329): greedy 0.90/0.00/0.10 (top of old band despite
    the stingier referee — new baseline anchor), random 1.00, Pikafish 0/20;
    gate/seconds 2044 (longer games, as predicted).
  - mate_win_rate mean ~0.13 (baseline ~0.10), two 0.202 readings, noisy
    (0.048 at 337) — below the >0.15-sustained success bar; too early.
  - repetition_draw_rate ~0.31 band — clear of the 0.45 revert line but
    drifted up slightly; primary watch metric.
  - blunder_rate steady ~0.25; loss/value up to ~5.3 and value_cp_corr
    0.68→~0.57 = expected recalibration to longer games + buffer refill
    after restart (mean_sampled_age climbing toward ~750 equilibrium).
- Arena 2026-07-10 (run under old rules via config pin, pairs through 320):
  ratings DISCONTINUOUS with the 2026-07-09 fit on the same stored pairs
  (160→180 gap +392 vs +102 from what should be identical games) and the
  +1701 total is contradicted by both gates — magnitudes untrusted pending
  a games.jsonl audit (sims values + per-pair W/D/L). Curve SHAPE (monotonic
  climb, flat 300→320) remains consistent with gate history.

## 5. Qualitative Outcome

- Success = mate rate climbing past ~0.15 sustained; arena Elo slope
  positive; endgames subjectively purposeful in human play.
- Revert = repetition-draw rate ballooning past ~0.45 sustained (games
  degenerating into shuffling instead of converting) or greedy band clearly
  degrading for 3+ gates beyond the instrument caveat above → set back to 6.

## 6. Repo / Handoff Updates

- `docs/AGENT_TODO.md` — active task updated to experiment #2 watch protocol.
- Related: `docs/logs/2026-07-08-log-shaping-weight-cut.md` (experiment #1,
  banked), `docs/logs/2026-07-07-log-greedy-rung-and-elo-arena.md`
  (instruments).

## 7. Conclusion & Next Steps

Restart to activate; continue periodic arena top-ups; judge after 3+ gates
AND enough iterations for endgame data to accumulate (~30–40 iters). Queued
next if endgames improve but hanging pieces persist: engine-game seeding
(5–10% Pikafish games per iteration).
