# MuZero for Xiangqi — Design Spec Session

**Date:** 2026-07-02
**Agent/Author:** Claude Code (Fable 5)

## 1. Hypothesis / Goal
Design a MuZero-inspired (EfficientZero-style) self-play agent for Xiangqi on a
single RTX 5090, as a new subsystem independent of the LLM/GRPO pipeline.
Brainstorming session (superpowers:brainstorming): explore requirements,
resolve ambiguities, produce an approved design spec before any implementation.

## 2. Configuration Changes
- New: `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md` (approved design spec)
- No code changes; no dependency changes yet (implementation will add a `muzero/` package)

## 3. Run Command
Not run — design/documentation session only.

## 4. Quantitative Results
Not applicable (no training run). Key spec numbers for reference: 800 MCTS
sims, K=8 unroll, 3×28 parallel games, 5,000-game buffer, ~512 games' worth of
positions per train loop, ~2,000-ply Pikafish warm start, 10-move opening book,
enemy sync after 3 consecutive ally wins.

## 5. Qualitative Outcome
- User's detailed request disambiguated; three assumptions confirmed by user:
  hybrid Pikafish reward role, "MCTS of 8" = K=8 training unroll, "512 games
  per train loop" = positions covering ~512 games.
- Approach chosen: custom EfficientZero-style implementation in-repo
  (vs adapting muzero-general / starting with Gumbel MuZero — both rejected,
  Gumbel kept as a drop-in fallback behind the isolated search interface).
- Legality is Pikafish-only via existing `src/pikafish_eval.py`
  `list_legal_moves` (perft 1, cached); gym_xiangqi provides board mechanics only.

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md`: unchanged (no components added yet; will be updated
  when `muzero/` lands)
- `docs/AGENT_TODO.md`: added active task for MuZero implementation planning
- Related logs/docs: `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`

## 7. Conclusion & Next Steps
- Design approved by user in-session; spec committed.
- Next: user reviews the written spec file; on approval, run
  superpowers:writing-plans to produce the implementation plan under
  `docs/superpowers/plans/`, then implement `muzero/` per the plan.
