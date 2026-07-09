# Shaping Weight Cut (0.3 → 0.1) — the one-knob experiment

**Date:** 2026-07-08
**Agent/Author:** Claude Code

## 1. Hypothesis / Goal

Play-strength metrics plateaued from ~iter 48 through 210 (blunder rate flat
at ~0.25–0.30, mate rate ~0.02–0.14) while every world-model loss kept
improving. The first greedy-gate baseline (iters 159–209, 20 games/gate):
**0.40, 0.35, 0.95, 0.45, 0.60, 0.60** — the model is roughly even against a
zero-strategy capture bot. Hypothesis: the dense material-shaping reward
(`shaping_weight=0.3`, per-move ±0.3·tanh(Δcp/200)) taught blunder avoidance
from scratch but is now saturated and drowns the terminal ±1 win signal
inside the 10-step value targets (several shaping terms per window vs one
distant terminal). Cutting it to 0.1 makes winning ~3× more important
relative to material safety. Single-knob discipline: NOTHING else changes.

## 2. Configuration Changes

- `muzero/config.py`: `shaping_weight` 0.3 → 0.1 (comment documents baseline
  + revert criterion). No other change.
- Takes effect on training restart (`--resume`); value-head targets shrink in
  magnitude, so expect `loss/value` and `value_cp_correlation` to wobble for
  ~10–20 iterations while the head recalibrates — adjustment, not damage.
  The blunder-rate METRIC is unaffected (it reads engine cp deltas, not
  rewards) and stays comparable across the change.

## 3. Run Command

```bash
# box:
git pull && uv run pytest muzero/tests -q
# stop training, then:
uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt
```

## 4. Quantitative Results (fill in as gates arrive)

- Pre-change baseline (shaping 0.3): greedy gate 0.40/0.35/0.95/0.45/0.60/0.60
  (mean ≈ 0.56, ±80 Elo-scale noise at 20 games); blunder ~0.25;
  mate ~0.06 (spikes to 0.14); value_cp_corr ~0.46; Elo anchor points
  archived at iters 160/180/200.
- Post-change greedy gates (iters 219–269): **0.75, 0.95, 0.85, 0.85, 0.50,
  0.55** — mean ≈ 0.74 vs pre-change 0.56 (pooled 89/120 vs 67/120 wins,
  z ≈ 3.0 before correcting for within-gate clumping; the protocol's first
  three readings all cleared the baseline mean → **verdict: knob worked**).
  Later 0.50/0.55 readings show churn persists — a band shift up, not
  variance eliminated.
- Mate rate: record highs post-change — 0.179 (iters 223, 261), **0.202
  (iter 272, all-time high)**; running average ~0.11 vs ~0.08 before.
- **First-ever non-losses vs full Pikafish:** gate draws 0.10 at iter 249
  and 0.05 at iter 269 (all prior gates were 100% losses).
- Blunder rate: ~0.25 band throughout, 60+ iterations post-change — the
  feared regression never appeared. Consistency loss reached new bests
  (−0.956). Arena Elo step still pending (archives 160–260 now straddle the
  change).

## 5. Qualitative Outcome

- Success = greedy win rate trending above the 0.35–0.95 baseline band and/or
  arena Elo slope steepening and/or mate rate climbing; blunder rate must not
  regress past ~0.35 sustained.
- Failure/revert = greedy win rate below the baseline band for 3+ consecutive
  gates, or blunder rate re-inflating toward 0.4+ → revert to 0.3 (weights
  keep everything learned; only the reward rules roll back).

## 6. Repo / Handoff Updates

- `docs/AGENT_TODO.md` — active task updated to the post-knob watch protocol.
- Related: `docs/logs/2026-07-07-log-greedy-rung-and-elo-arena.md`
  (instruments), `docs/superpowers/specs/2026-07-07-*.md`.

## 7. Conclusion & Next Steps

Restart to activate; run the arena concurrently (CPU) for the Elo baseline
over iters 160/180/200 + latest; judge after 3+ post-change gates with BOTH
instruments; keep `truncation_consecutive` 6→12 as the NEXT single knob if
mate rate stays low after this one settles.
