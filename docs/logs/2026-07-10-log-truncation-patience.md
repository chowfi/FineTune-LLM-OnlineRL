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
- Post-change update 2 (iters 386–444, gates 7–12): mate_win_rate
  **confirmed** — ~0.18–0.19 sustained for ~60 iters, record 0.310 at
  iter 422. Greedy gate cooled off the honeymoon: 1.00 (389), then 0.65,
  0.65, 0.70, 0.45, 0.80 — mean ~0.65, back inside the old 0.50–1.00 band,
  with greedy DRAWS (0.05–0.20/gate) now appearing per the instrument
  caveat (patient referee converts some old adjudicated wins to draws).
  NOT a revert trigger (band not clearly below old band; blunder ~0.25
  steady). Watch rule: two more gates ≤0.5 = real pattern, investigate.
  Losses now slowly improving off the new plateau (total 6.59→6.42,
  policy 5.33→5.21, consistency −0.968 best, material 0.084 best).
  Arena top-up over archives 340–440 pending = tiebreaker for the gate dip.
- Post-change update (iters 322–385, six gates): greedy gate **0.90, 0.95,
  0.70, 1.00, 1.00, 1.00** (mean ~0.93 vs old-band 0.74 — three consecutive
  perfect 20/20s, despite the stingier-referee caveat predicting the
  opposite bias). mate_win_rate last ~30 iters ~0.17–0.18 sustained
  (record 0.298 at iter 359; repeated 0.20+ readings) — clears the >0.15
  success bar. Pikafish gate DRAW 0.05 at iter 379 (third-ever non-loss).
  repetition_draw_rate ~0.27 avg, single 0.43 blip, trending down — revert
  tripwire never approached. blunder ~0.25 unchanged; loss/value plateaued
  ~6.0 and value_cp_corr recovering ~0.6 (recalibration done; buffer at
  equilibrium, mean_sampled_age ~740). New-era archives: 340/360/380.
  **Verdict: success on gate + mate-rate instruments; awaiting arena
  top-up (340/360/380 + latest, new rules, no config pin) to confirm and
  bank.**
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
  the raw table (180→+392 … 320→+1701) was a FITTER BUG, not real strength —
  `fit_ratings` initialised free players at Elo 1500 vs the 0-anchor and
  L-BFGS-B (optimising in raw Elo units, ~0.006 nll per point) stalled far
  from the optimum once the chain grew to 9 free players. games.jsonl audit
  showed healthy close pairs (e.g. 160-180: 8W 9D 3L). Fixed in
  `scripts/benchmark/elo_estimator.py` (optimise in nat units, init at
  anchor mean, tighter ftol/gtol; regression test in
  `muzero/tests/test_arena.py`). Converged refit of the same 200 games:
  160→0, 180→100, 200→78, 220→78, 240→120, 260→176, **280→276, 300→373,
  320→353** — reproduces the 2026-07-09 table for old points and extends
  the experiment-#1 climb through 300, flat 300→320 (matches gate
  consolidation). End-of-era anchor for experiment #2: ~+350 at iter 320.

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

**VERDICT (2026-07-12): SUCCESS — BANKED.** All pre-registered criteria met:
mate_win_rate ~0.18–0.19 sustained (bar 0.15; record 0.310 at iter 422);
arena 2026-07-12 (fixed fitter, new-era pairs under new rules): 320→356,
**340→464, 360→641, 380→725** — +369 Elo across iters 320–380 (~+6.2/iter),
the steepest era of the run (vs ~+3.7/iter in experiment #1, ~+1.3 early).
The gates-7–12 greedy dip (0.45–0.80) is overruled: it coincided with the
fastest arena climb → churn + draw-caveat, not regression. The iter-386
extra point (+813, 20 games, 6 iters after 380) is noise — ignore.
Truncation stays at 12. Next knob candidate if hanging pieces persist at
the board (user to judge via web UI): engine-game seeding (5–10% Pikafish
games per iteration).
