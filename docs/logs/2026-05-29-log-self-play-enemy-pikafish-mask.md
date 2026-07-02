# Self-play enemy Pikafish legality mask

**Date:** 2026-05-29
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Self-play enemy scored moves from gym `enemy_actions` only, allowing ~7–12% Pikafish-illegal plies in ally-win games (audit on ep 33–44). Align training enemy harness with ally + web UI: Pikafish-legal list → score → argmax.

## 2. Configuration Changes

- `LLM_RL_agent_FSDP_v2.py`: added `apply_pikafish_enemy_legal_mask()` (Black, `side_to_move=b`, UCI→algebraic mapping).
- `generate_self_play_enemy_move()` applies mask before `get_flipped_enemy_legal_actions()`.
- New episode metric: `game/enemy_pikafish_prune_rate` (% of gym enemy moves pruned per masked turn); CSV column `game_enemy_pikafish_prune_rate`.

## 3. Run Command

Unchanged training entrypoint; takes effect on next run/resume.

## 4. Quantitative Results

Pre-change audit (latest `output.log`, ep 33–44): 29/397 enemy plies (7.3%) Pikafish-illegal but gym-played; 11.6% in ally-win games, 0% in truncated.

## 5. Conclusion & Next Steps

- Expect harder, more realistic self-play; ally win rate may drop slightly.
- Watch `game/enemy_pikafish_prune_rate` in W&B / CSV; non-zero means gym was wider than Pikafish.
- `[self-play opponent] Pikafish mask gym=… pf=… pruned=…` logs when moves are dropped.
