# Feature / Experiment Log

**Date:** 2026-07-02
**Agent/Author:** Claude (subagent)

## 1. Hypothesis / Goal
Close a review gap on `muzero-xiangqi`: several spec §10 wandb metrics were designed but never wired into the self-play/metrics/replay-buffer/train pipeline. Add only the CHEAP ones — those computable from data already produced by self-play and training, with no extra Pikafish engine calls or GPU introspection.

## 2. Configuration Changes
- `muzero/selfplay.py`: `_Game.__init__` gains `ally_entropies` and `ally_value_cp_pairs` lists. `_record_and_step` computes root-policy entropy from MCTS `visits` right before the ally moves, and pairs `(root_value, ally_cp)` right after the ally moves (using the `red_cp` already returned in `env.step`'s `info`, sign-adjusted for `ally_side`). `_finish`'s summary dict gains `mean_root_entropy`, `value_cp_pairs`, `mean_ally_cp`, `games_this_era`.
- `muzero/metrics.py`: `aggregate_game_summaries` gains `selfplay/mean_root_entropy`, `selfplay/mean_ally_cp_auc` (per-game cp-AUC proxy), `selfplay/value_cp_correlation` (Pearson correlation between pooled root values and ally cps, NaN/variance-guarded), and `selfplay/games_per_promotion`. All new lookups use `.get(...)` with defaults so summaries lacking the new keys (the old-style test fixture) still work unmodified.
- `muzero/replay_buffer.py`: `GameHistory.__init__` gains `buffer_index = 0`. `ReplayBuffer.add()` stamps `game.buffer_index = self.total_games_added` before incrementing the counter, inside the existing lock. `sample_batch()` now also returns `mean_buffer_age` (mean of `total_games_added - buffer_index` over the sampled picks) as a 0-d `np.float32`.
- `muzero/train.py`: `MuZeroTrainer.train_batch` pops `mean_buffer_age` from the incoming batch dict *before* `_to_tensors` (which would otherwise choke tensorizing a bare scalar), and reports it back as `result["buffer_age"]`. The existing main-loop `loss_sums` averaging picks this up automatically as `loss/buffer_age` — no main-loop change needed.
- `docs/AGENT_TODO.md`: appended a backlog line under the existing MuZero follow-ups item listing the deferred (non-cheap) §10 metrics: fraction of ally moves matching Pikafish best, GPU inference batch utilization, per-era win-rate breakdown.
- `docs/ARCHITECTURE.md`: expanded §3f's MuZero description to mention the new diagnostics and where they're wired.

## 3. Run Command
```bash
uv run pytest muzero/tests -v
uv run ruff check muzero --fix && uv run ruff format muzero
```

## 4. Quantitative Results
- Test suite: **42 passed, 5 skipped** (up from 34 passed / 5 skipped — 8 new tests added, 0 removed, the original `test_aggregate_game_summaries` and `test_sample_batch_shapes`/`test_train_batch_runs_and_updates_params` fixtures are byte-for-byte unmodified and still pass).
- `ruff check muzero --fix`: all checks passed (no fixes needed beyond formatting).
- `ruff format muzero`: reformatted 2 files (`muzero/selfplay.py`, `muzero/tests/test_selfplay.py`) for line-wrap only; no behavior change.
- The 5 skips are the pre-existing engine-gated tests (`PIKAFISH_BIN` not set locally); unaffected by this change.

## 5. Qualitative Outcome
New tests exercise `SelfPlayWorker._record_and_step`/`_finish` directly against a real `XiangqiEnv` + `FakeEvaluator` (no engine binary needed), confirming: entropy is only recorded when the ally is about to move (and is ~0 for a one-hot/forced-opening visit distribution); cp pairing only fires right after the ally's own move (not the enemy's); and `_finish`'s summary carries the accumulated values through. `metrics.py` tests cover the `.get`-guarded fallback path (old-style summaries with no new keys → 0.0 everywhere), a perfectly-correlated value/cp pair set (correlation ≈ 1.0), a `None`-skipping cp-AUC mean, and both `games_per_promotion` branches (with/without a promotion). `replay_buffer`/`train` tests confirm `mean_buffer_age` is present, finite, non-negative, monotonic with `buffer_index` assignment order, and that `train_batch` pops it out of the tensorized batch and returns a finite `buffer_age`.

One noteworthy nuance surfaced while writing the selfplay test: `env.step`'s `red_cp` is captured *after* the side-to-move flip, so a flat/constant fake cp function produces a *negated* sign relative to the naive "ally just got +50cp" intuition — worth remembering if a future engine-gated smoke test asserts on `value_cp_pairs` sign.

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md`: updated (§3f expanded, see above).
- `docs/AGENT_TODO.md`: updated (deferred §10 metrics backlog line added).
- Related logs/docs: this log; no other logs touched.

## 7. Conclusion & Next Steps
Cheap §10 diagnostics are fully wired end-to-end (self-play → aggregation → wandb via the existing `metrics.update({"loss/...": ...})` / `MetricsLogger.log` path) with no additional engine calls. Remaining deferred §10 items (Pikafish-best-move agreement rate, GPU batch utilization, per-era win-rate breakout) need either an extra engine call per ally move or GPU introspection plumbing and are intentionally left for a follow-up — tracked in `docs/AGENT_TODO.md`. Next agent picking up MuZero work should still start with the existing active task: first real-machine run with `PIKAFISH_BIN` set.
