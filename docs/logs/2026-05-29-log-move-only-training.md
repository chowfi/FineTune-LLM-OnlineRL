# Experiment Log: Move-only GRPO (no thinking)

**Date:** 2026-05-29
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal
Thinking tokens may dilute the RL signal. Test whether training only on `Move: <uci>` with pure Pikafish engine rewards speeds learning vs think+move format.

## 2. Configuration Changes
- `prompt/move_only: True` — system/user prompts and sampler responses are one line: `Move: h2e2`
- `grpo/train_move_tokens_only: True` — GRPO PG/KL/clip use `move_mask` only (not think tokens)
- `reward/engine_only: True` — reward = Pikafish cp-shaped engine signal; no format/reasoning bonus
- `generate/max_new_tokens: 32` (was 384)
- `sampler/generate_grounded_reasoning` auto-disabled when move_only

## 3. Run Command
Same v2 training launch as prior run (restart required).

## 4. Quantitative Results
*(Pending.)* Compare vs prior run: `game_chosen_is_engine_best_overall_rate`, offline eval `good@1`, wall time per ally turn.

## 5. Conclusion & Next Steps
- Web play UI still uses think+move prompts unless updated separately.
- To revert: set the three flags above to `False` and restore `generate/max_new_tokens: 384`.
