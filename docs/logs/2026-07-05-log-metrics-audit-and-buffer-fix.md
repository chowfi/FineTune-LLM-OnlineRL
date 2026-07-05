# MuZero Metrics Audit + Replay-Buffer Staleness Fix

**Date:** 2026-07-05
**Agent/Author:** Claude Code

## 1. Hypothesis / Goal

Audit of the latest-mode run (iters 12–80, 2026-07-04/05) surfaced four concerns:
rising `loss/value` (6.0 → 7.6) with `value_cp_correlation ≈ 0`, ~80%
`truncation_rate`, persistently negative `mean_ally_cp_auc` (−390 → −650)
despite symmetric self-play, and sustained black > red win-rate stretches.
Root-cause each; fix what is actually broken; drop metrics that carry no
information.

**Root causes found:**

1. **Rising value loss — real pathology + benign component.** `buffer_games=5000`
   at 84 games/iter is a ~60-iteration window; `buffer_age` plateaued at ~2350
   games (≈ half of capacity) exactly when the buffer filled (~iter 60–70).
   Value targets bootstrap off `root_values` recorded at play time
   (`replay_buffer.n_step_value`) — on average from a ~28-iteration-stale
   policy — while frozen PER priorities (computed once at `add()`, value
   TD-error, no IS correction) steer sampling toward exactly those stale
   positions. Benign component: draws fell 52% → ~20% over the same span, so
   value targets moved from near-0 to hard-to-predict ±1 outcomes. (Note the
   logged `loss/value` is ~2× per-position CE — initial inference `.mean()`
   plus K unrolled CEs each /K — so per-position CE was ~3.0 → 3.8 vs a
   uniform-prediction 6.4; far from broken, but trending wrong.)
2. **Truncation rate was misread — mostly good news.** `truncated=True` is the
   cp-adjudication *win* (loser ≤ −800cp for 6 consecutive own-moves), not the
   ply-cap draw. 0.48 → 0.80 truncation = games went from half-drawn to ~80%
   decisive. Reward bookkeeping verified correct: only the loser gets −1 at
   adjudication and the alternating-sign n-step return converts it to +1 for
   the winner's states.
3. **Negative `mean_ally_cp_auc` — metric artifact.** Sign chain verified
   correct end-to-end. The bias was *when* it sampled: only immediately after
   the tracked side's own moves — in a ~50%-blunder regime that is the bottom
   of the eval sawtooth every time (your blunder is priced in, the reply
   blunder is not), for both colors. Plus lost-game tails (≤ −800 for 12+
   plies pre-adjudication) drag per-game means down.
4. **Black > red oscillation — structural.** `encoding.py` never canonicalizes
   color: planes are absolute (red 0–6, black 7–13, rank 0 top) + an stm
   plane, while policy/value are mover-perspective. The net learns "play red"
   and "play black" as two loosely-coupled tasks whose relative progress
   oscillates (black 0.55–0.73 iters 25–41; red back ahead ~67–80). Red's
   forced 1-move opening book adds a small structural black edge. Left as-is
   (self-correcting; the canonicalization refactor invalidates checkpoints) —
   candidate for a fresh-restart follow-up.
5. **Gate uninformative.** `run_gate` played only full-strength Pikafish at
   10 ms — 0/20 with zero draws at every gate is the expected result for a
   young net and carries no signal.

## 2. Configuration Changes

- `muzero/config.py` — `buffer_games`: **5000 → 1500** (~18 iterations of
  self-play; mean sampled age drops from ~28 to ~9 iterations).
- `muzero/selfplay.py` — `_Game.ally_cps` now records the tracked color's cp
  after **every** ply (both movers), fixing the sawtooth-trough sampling bias
  in `selfplay/mean_ally_cp_auc`. Dropped the now-unused `final_red_cp` from
  game summaries (saves one engine eval per game).
- `muzero/metrics.py` — **removed** `selfplay/mean_final_ally_cp` (noisy
  duplicate of win/loss rate) and `selfplay/games_per_promotion` (constant 84
  in latest mode; `selfplay/promotions` retained for frozen-enemy runs).
- `muzero/train.py` — gate is now a two-rung ladder (`_run_gate_rung`):
  `gate/win|draw|loss_rate_random` vs a uniform-random legal mover (the rung
  where early progress resolves; should sit near 1.0 win) and the existing
  `gate/win|draw|loss_rate` vs raw Pikafish at `gate_movetime_ms` (keys
  unchanged for wandb continuity). `buffer_age` is now logged as
  `buffer/mean_sampled_age` instead of `loss/buffer_age` (it is a sampling
  diagnostic, not a loss term).
- Tests updated: `test_config.py` (buffer size), `test_metrics.py` (removed
  metrics), `test_selfplay.py` (every-ply `ally_cps`, `mean_ally_cp == 0.0`
  for the symmetric ±50 fake-evaluator game).

## 3. Run Command

```bash
uv run pytest muzero/tests -q
uv run ruff check muzero --fix && uv run ruff format muzero
```

Training not relaunched from here (runs on the 5090 box). The buffer change
takes effect on restart — the buffer is not persisted, so `--resume
checkpoints/muzero_xiangqi/latest.pt` keeps the net/optimizer and refills a
1500-game buffer from live self-play.

## 4. Quantitative Results

- **Tests:** 56 passed, 5 skipped (engine-gated), 0 failed.
- **Ruff:** clean (1 file reformatted).
- Run evidence for the diagnosis (iters 12→80): `loss/value` 6.0→7.6 while
  `buffer_age` 34→~2350 (plateau ≈ buffer capacity/2); `blunder_rate`
  0.68→~0.45; draws 0.52→~0.2; `mate_win_rate` 0→0.02–0.08;
  `value_cp_correlation` mean ≈ +0.08.

## 5. Qualitative Outcome

- Training itself is healthy on the policy/search side (blunders down, real
  mates appearing, draws collapsing); the one real defect was value-target
  staleness from the oversized replay window, now fixed in config.
- Two metrics were lying (`mean_ally_cp_auc` sampling bias, gate-vs-superhuman)
  and two were dead weight (`mean_final_ally_cp`, `games_per_promotion`);
  all addressed. Expect `mean_ally_cp_auc` to jump toward ~0 on restart purely
  from the metric fix — do not read that step as a strength gain.

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md`: §3f updated (buffer window rationale, every-ply cp
  tracking, removed metrics, gate ladder, `buffer/mean_sampled_age`).
- `docs/AGENT_TODO.md`: restart-task note updated to reference the gate ladder
  and buffer change; follow-ups added (color canonicalization on fresh
  restart, per-color blunder-rate diagnostic, `per_alpha` as a second lever,
  raise `truncation_consecutive` if mate rate stays flat).
- Related logs/docs: `docs/logs/2026-07-04-log-training-health-metrics.md`
  (the metrics this audits), `docs/logs/2026-07-03-log-latest-selfplay-mode.md`.

## 7. Conclusion & Next Steps

- Restart (or resume) the training run to pick up `buffer_games=1500` and the
  gate ladder. Primary signals going forward: `gate/win_rate_random` (expect
  → 1.0), `selfplay/blunder_rate` (down), `selfplay/mate_win_rate` (up),
  `loss/value` (should stop climbing within ~10 iterations of the smaller
  buffer filling), `selfplay/value_cp_correlation` (should finally trend
  positive).
- If `value_cp_correlation` is still ~0 after ~30 iterations: drop
  `per_alpha` 0.6 → 0.3 (frozen-priority bias), then consider
  reanalyze-style fresh bootstrap values.
- If `mate_win_rate` stays < 0.1 after ~50 iterations: raise
  `truncation_consecutive` 6 → 10–12 to force conversion practice.
- On any fresh restart: consider color canonicalization in `encoding.py`
  (flip board + swap plane groups for black) — halves the representation task
  and removes the red/black oscillation.
