# Latest-Weights Self-Play Mode (stock MuZero) — Implementation

**Date:** 2026-07-03
**Agent/Author:** Claude Code (Fable 5), subagent-driven development

## 1. Hypothesis / Goal
Switch default self-play to stock MuZero (latest network plays both sides) per
`docs/superpowers/specs/2026-07-03-muzero-latest-selfplay-design.md`, keeping
the original frozen-enemy promotion scheme as an ablation behind
`self_play_mode="frozen_enemy"`. Motivation: half the replay data in the
frozen scheme comes from the stale enemy net, and 3-consecutive-wins promotion
is statistically weak under 60–75% draw rates.

## 2. Configuration Changes
- `muzero/config.py`: `self_play_mode: str = "latest"` (validated) +
  `truncation_symmetric` derived in `__post_init__`.
- `muzero/env.py`: per-color saturation streaks; symmetric truncation in
  latest mode (either side's 6-turn hopeless streak ends the game, that side
  loses); frozen mode byte-identical (proven equivalent in review).
- `muzero/selfplay.py`: `_round_groups` spec helper — latest mode runs ONE
  batched MCTS group per lockstep round with Dirichlet noise on every root;
  coordinator `promotion_enabled` gate; diagnostics (root entropy, mover-persp
  value-cp pairs) cover every move in latest mode; new `ally_cps` keeps
  `mean_ally_cp` tracked-color semantics in both modes.
- `muzero/metrics.py`: `selfplay/red_win_rate`, `selfplay/black_win_rate`.
- `muzero/train.py`: latest mode skips the enemy deepcopy (`enemy = ally`),
  shares the ally `NetRunner`; new `load_checkpoint` helper; checkpoint omits
  the `"enemy"` entry in latest mode and resume falls back
  `ckpt.get("enemy", ckpt["ally"])` — old/new checkpoints load in either mode.

## 3. Run Command
```bash
uv run pytest muzero/tests -v   # 54 passed, 5 skipped (engine-gated)
uv run python -m muzero.train --help
```

## 4. Quantitative Results
Suite 45 → 54 passed (9 new tests) across 7 commits on branch
`muzero-latest-selfplay` (`88482ad`..`38a3620` + docs). No training run yet in
the new mode.

## 5. Qualitative Outcome
- Review loop caught a fixture bug in the plan itself: a constant
  side-to-move-perspective cp makes BOTH movers look hopeless; the asymmetric
  test needed a side-differentiated `cp_fn`.
- Reviewer-driven addition: `test_record_and_step_latest_mode_records_every_move`
  so a silent revert to ally-only diagnostics fails the suite.
- Frozen-mode behavior verified byte-identical by trace and by the pinned
  legacy tests (three tests now explicitly opt in to
  `self_play_mode="frozen_enemy"`).

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md` §3f: self-play modes description replaced.
- `docs/AGENT_TODO.md`: new Active task "restart training in latest-selfplay
  mode" with resume/compare guidance.
- Spec + plan under `docs/superpowers/` (plan self-review removed two
  superseded code drafts before execution).

## 7. Conclusion & Next Steps
Merge `muzero-latest-selfplay` → `main`, pull on the GPU box, then either
`--resume` the current checkpoint into latest mode or restart fresh (cleaner
wandb separation for the ablation). The frozen-enemy baseline stays one config
edit away for the writeup comparison.
