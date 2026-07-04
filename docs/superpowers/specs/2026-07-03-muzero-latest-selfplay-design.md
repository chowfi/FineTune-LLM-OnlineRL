# MuZero-Style Self-Play (Latest Weights Both Sides) — Design Spec

**Date:** 2026-07-03
**Status:** Approved by user (design conversation, 2026-07-03; approach A)
**Amends:** `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md` §7 (self-play opponent)

## 1. Goal

Replace the default self-play scheme — learner vs frozen enemy with promotion
after 3 consecutive wins — with stock MuZero self-play: the latest network
plays **both** sides of every game. The frozen-enemy scheme remains available
behind a config flag as a research ablation. Motivation: in the current
scheme, half the replay data comes from the stale frozen network, and the
3-consecutive-wins promotion test is statistically weak (draws reset the
streak; observed 60–75% draw rates make promotion partly luck).

## 2. Decisions (confirmed with user)

| Decision | Choice |
|---|---|
| Keep frozen-enemy machinery? | Yes, behind `self_play_mode` config flag; **default `"latest"`** (MuZero style), `"frozen_enemy"` preserves current behavior exactly |
| Truncation in latest mode | **Symmetric**: truncate when either side has been hopeless (≤ truncation_cp) for truncation_consecutive of its own turns; that side is scored the loser; tail down-weighting unchanged |
| Approach | A — runner-level switch: in latest mode no enemy net exists; both sides share the ally `NetRunner` |

## 3. Config (`muzero/config.py`)

- New field `self_play_mode: str = "latest"`; allowed values `"latest"`,
  `"frozen_enemy"`; validated in `__post_init__` (raise `ValueError` otherwise).
- `__post_init__` derives `truncation_symmetric: bool = (self_play_mode ==
  "latest")` — not independently settable, so mode and truncation semantics
  cannot disagree.
- `promote_after_consecutive_wins` is unchanged and only meaningful in
  frozen mode.

## 4. Environment (`muzero/env.py`)

- Replace the single `_sat_streak` with per-color streaks
  (`{"w": 0, "b": 0}`), each incremented/reset on that color's own post-move
  mover-perspective cp.
- `truncation_symmetric=True`: either color's streak reaching
  `truncation_consecutive` ends the game; the saturated color loses (result
  `red_win`/`black_win` accordingly, `truncated=True`, mover reward −1 when
  the mover is the loser — the trigger always fires on the saturated side's
  own move, so the mover is always the loser, same as today).
- `truncation_symmetric=False`: behavior byte-identical to current
  (only the `ally_side` streak counts).
- **Scope note (decided at final review):** warm start and the fixed-Pikafish
  gate build `XiangqiEnv` from the same config, so in latest mode they also
  adjudicate symmetrically. This is intentional: warm start's old red-only
  asymmetry was an artifact of its hardcoded `ally_side="w"` (symmetric
  adjudication of engine-vs-engine games is strictly better), and gate games
  ending when either side is hopeless for 6 turns is standard computer-chess
  adjudication — it shortens gating and counts decisive positions as wins.

## 5. Self-play (`muzero/selfplay.py`)

- `SelfPlayWorker` reads `cfg.self_play_mode`:
  - **latest**: each lockstep round makes ONE `MCTS.run` call over all active
    games using the (shared) ally runner — larger inference batches;
    `add_noise=True` for every root (stock MuZero); root-entropy and
    value-cp diagnostic pairs recorded for **all** moves (every move is the
    learner's).
  - **frozen_enemy**: current two-group behavior unchanged (noise on ally
    roots only, diagnostics ally-only).
- `ally_side` continues to alternate per game in both modes; in latest mode
  it is purely a metrics label ("tracked color") preserving wandb continuity.
- `SelfPlayCoordinator` gains `promotion_enabled = (cfg.self_play_mode ==
  "frozen_enemy")`; when disabled, `report_result` counts games/streaks but
  never copies weights and `era` stays 0.

## 6. Training loop (`muzero/train.py`)

- **latest**: skip the enemy `deepcopy` (no duplicate 22M-param net);
  `enemy_runner = ally_runner`; coordinator constructed with the ally net in
  both slots and the shared lock (promotion disabled anyway).
- **frozen_enemy**: wiring unchanged.
- Checkpoints: save `"enemy"` weights only in frozen mode; on resume use
  `ckpt.get("enemy", ckpt["ally"])`. Existing checkpoints (which always have
  the key) load in either mode; a frozen-mode checkpoint can be resumed
  directly into latest mode (the enemy entry is simply ignored).

## 7. Metrics (`muzero/metrics.py`)

- Add `selfplay/red_win_rate` and `selfplay/black_win_rate` (always logged;
  in latest mode the color split is the informative view, e.g. first-mover
  advantage).
- All existing keys keep logging in both modes (promotion keys are flat in
  latest mode). The fixed-Pikafish gate is unchanged and becomes the primary
  progress signal in latest mode, as in the MuZero paper.

## 8. Testing

- Config: mode validation raises on bad values; `truncation_symmetric`
  derivation for both modes.
- Env: symmetric-truncation test mirroring the existing ally-side test but
  with the **non-ally** color hopeless (e.g. black saturated → `red_win`,
  `truncated=True`); existing asymmetric test still passes with
  `self_play_mode="frozen_enemy"`.
- Selfplay: latest-mode worker with a single shared runner generates games
  end-to-end (engine-gated smoke variant); coordinator with promotion
  disabled never syncs weights and keeps `era == 0`.
- Train: resume from a checkpoint lacking the `"enemy"` key.
- Existing frozen-mode tests updated only where they must opt in to
  `self_play_mode="frozen_enemy"`.

## 9. Rollout / compatibility

- Default flips to latest mode: the next `uv run python -m muzero.train`
  after pulling runs MuZero-style. The in-flight BN-fixed run's checkpoint
  remains loadable (`--resume`) in either mode; network format is untouched.
- The frozen-enemy ablation is one config edit away
  (`self_play_mode="frozen_enemy"`), keeping the A/B comparison for the
  research writeup.

## 10. Alternatives considered

- **Promote-every-game** (sync enemy←ally after each game): approximates
  latest-weights without rewiring, but constant weight copies, still-stale
  in-round data, and a muddier ablation. Rejected.
- **Full removal** of the frozen-enemy path: simplest code, but loses the
  cheap A/B ablation; rejected by user.
