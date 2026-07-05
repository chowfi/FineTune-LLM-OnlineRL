# MuZero Color Canonicalization + Mirror Augmentation

**Date:** 2026-07-05
**Agent/Author:** Claude Code (subagent-driven development; coordinator + per-task implementer/spec-reviewer/quality-reviewer subagents)

## 1. Hypothesis / Goal

The 2026-07-05 metrics audit identified the absolute-orientation encoding as
the root cause of the red/black win-rate oscillation: with mover-perspective
heads but fixed-frame inputs, the net learns "play red" and "play black" as
two loosely-coupled skills. Canonicalize so the network always sees "me at the
bottom, moving up, my pieces in planes 0–6" (one skill, color-balanced — which
also serves the user's goal of playing against the net on either side), and
add left-right mirror augmentation (~2× effective data). Spec:
`docs/superpowers/specs/2026-07-05-muzero-color-canonicalization-design.md`;
plan: `docs/superpowers/plans/2026-07-05-muzero-color-canonicalization.md`.

## 2. Configuration Changes

Branch `muzero-canonical`, 11 implementation commits (f24c80c…1777ff4) + docs:

- `muzero/encoding.py` — new primitives `flip_board`, `mirror_board`,
  `flip_action`, `mirror_action` (involutions; int or int-ndarray; TypeError
  on non-integer input, ValueError out of range) and `absolute_visits`.
  `encode_observation` flips the whole history stack when black is to move
  and drops the side-to-move plane: **input_planes 115 → 114**
  (`config.py` derivation now `14*history_length + 2`).
- `muzero/replay_buffer.py` — `make_target(game, t, mirror=None)`: flips
  black-ply actions/policy targets into the mover frame (`_target_action`,
  `_dense_policy`), emits mover-perspective material (with a documented
  frozen-terminal-frame convention for absorbing states), and applies
  per-sample LR-mirror augmentation drawn from the buffer's seeded RNG.
  `GameHistory` storage stays absolute.
- `muzero/selfplay.py` / `muzero/train.py` — shared `canonical_root(env)`
  helper flips root legal moves in; `absolute_visits` unflips MCTS results
  before storage/stepping, in both self-play and the gate.
- Tests: 68 passing (+6 engine-gated skips locally), including a keystone
  doorway-consistency test (black's a3a4 must decode as red's a6a5 in the
  canonical frame), mirror obs/target consistency pins, a fault-injection-
  verified root-adapter test, and an engine-gated legal-set bijection test
  (runs on the training box).

## 3. Run Command

```bash
uv run pytest muzero/tests -q          # 68 passed, 6 skipped locally
uv run ruff check muzero --fix && uv run ruff format muzero
```

Training launch is on the 5090 box (fresh run; see §7).

## 4. Quantitative Results

- Full suite: 68 passed, 6 skipped (engine-gated). Ruff clean on `muzero/`.
- Every task passed two-stage review (spec compliance, then code quality);
  notable review catches: a sign bug and a wrong expected-literal in the
  plan's own test code (engine-UCI vs internal-algebraic conversion), the
  single-legal-move involution-cancellation limit of the root-adapter test
  (documented in its docstring), and a duplicated adapter that was
  deduplicated into `canonical_root`.
- Fault injection (performed twice, independently): removing either
  direction of the root adapter makes
  `test_generate_stores_absolute_actions_for_black` fail (2466 vs 4905).

## 5. Qualitative Outcome

- Known, accepted limitation: TOTAL absence of the root adapter is not unit-
  testable (flip is an involution; a single-legal-move root cancels). Covered
  by the engine-gated bijection test (flip primitive vs real legality) and
  the live watch criteria below.
- Old checkpoints (115-plane first conv) are incompatible — fresh runs only.
- Pre-existing lint debt found in `scripts/claude_plays.py` (untouched;
  logged in AGENT_TODO).

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md` §3f — canonical-frame description + frame-convention
  paragraph (what's absolute at rest, where the flips live).
- `docs/AGENT_TODO.md` — canonicalization moved to Completed; Active task
  rewritten as the fresh-launch checklist; new backlog item: human-vs-MuZero
  web play adapter; scripts lint debt noted.
- Related: `docs/logs/2026-07-05-log-metrics-audit-and-buffer-fix.md` (the
  audit that motivated this).

## 7. Conclusion & Next Steps

On the 5090 box:

```bash
git pull
PIKAFISH_BIN=<path> uv run pytest muzero/tests -q   # engine tests must pass, esp. test_flip_maps_legal_move_sets
uv run python -m muzero.train --smoke --no-wandb --iterations 1 --device cpu
uv run python -m muzero.train                       # FRESH run — no --resume
```

Watch criteria (first ~30 iterations): `red_win_rate` ≈ `black_win_rate`
(no sustained 0.3+ gaps); `blunder_rate` falling at least as fast as the old
run's 0.68 → 0.45; `gate/win_rate_random` → ~1.0; `loss/value` flat-to-down
(1500-game buffer). If blunder rate falls *slower* than the old run, stop —
prime suspect is a mis-wired flip; check `search_kl` at black roots first.
