# MuZero Color Canonicalization + Mirror Augmentation — Design

**Date:** 2026-07-05
**Status:** Approved (brainstorming session with user)
**Predecessors:** `2026-07-02-muzero-xiangqi-design.md`, `2026-07-03-muzero-latest-selfplay-design.md`, `docs/logs/2026-07-05-log-metrics-audit-and-buffer-fix.md`

## 1. Goal

The network currently sees the board in absolute orientation (red planes 0–6,
black planes 7–13, fixed row order) with a side-to-move plane, while its
policy/value heads are mover-perspective. It therefore learns "play red" and
"play black" as two loosely-coupled skills, which the 2026-07-05 metrics audit
identified as the root cause of the sustained red/black win-rate oscillation
(black 0.55–0.73 for iters 25–41, red ahead by ~67–80) and an ~2× waste of
training signal.

This change canonicalizes the observation so the network always sees "me at
the bottom of my mental picture, moving up, my pieces in planes 0–6" —
AlphaZero-style — and adds left-right mirror data augmentation (Xiangqi is
left-right symmetric), roughly doubling effective training data.

**Non-goals:** any change to the env rules, engine boundary, replay-buffer
storage format, metrics semantics, or self-play scheme. No new
hyperparameters. Human-vs-MuZero play UI is out of scope (separate follow-up).

## 2. Core Rule (Approach A — boundary flip)

Everything **at rest** stays absolute: `XiangqiEnv`, `GameHistory` (boards,
actions, policy indices), the engine/FEN boundary, warmstart games, and all
metrics. Canonicalization is applied only where data crosses the network
boundary — two doorways plus one adapter:

```
play time:    env(absolute) ──[doorway 1: encode_observation flips]──> net
              net visits ──[MCTS-root adapter unflips]──> env/history(absolute)
train time:   GameHistory(absolute) ──[doorway 2: make_target flips]──> net
```

## 3. Components

### 3a. Transform primitives (`muzero/encoding.py`)

- `flip_board(board)` → `board[::-1] * -1` — reverse rows and negate signed
  piece ids (positive=red, negative=black), mapping "black to move" positions
  into the frame red enjoys. Works regardless of which end of the array red
  occupies, by construction.
- `flip_action(idx)` — decompose the flat index into (from-row, from-col,
  to-row, to-col), map `r → 9 − r`, columns unchanged, recompose.
- `mirror_board(board)` → `board[:, ::-1]`; `mirror_action(idx)` — `c → 8 − c`,
  rows unchanged.
- All four are involutions (self-inverse). Vertical flip + color swap is
  sufficient (no 180° rotation needed): Xiangqi piece rules are left-right
  symmetric and the palace/river are centered.

### 3b. Doorway 1 — `encode_observation` (play + train encode path)

When `side_to_move == "b"`, apply `flip_board` to **every** board in the
history stack (the whole stack is oriented to the current mover, matching
AlphaZero). The side-to-move plane is **dropped** — it is constant under
canonicalization. `input_planes` becomes `14 * history_length + 2 = 114`
(repetition and no-progress planes remain). All callers (selfplay, gate,
`make_target`, `diagnose_consistency`) inherit the flip automatically.

### 3c. MCTS-root adapter (selfplay + gate)

Policy logits are now in the mover's canonical frame, so root legal-move
indices must be flipped on the way in and visit-count keys / chosen actions
flipped back to absolute on the way out. One shared helper (e.g.
`canonical_root(env)` / unflip of the visits dict) used by
`SelfPlayWorker` and `_run_gate_rung`. `GameHistory` continues to store
**absolute** actions and policy indices (homogeneous, debuggable at rest).

Interior MCTS nodes need no change: the tree already uses negamax
mover-perspective values, and interior actions are chosen from (and fed back
into) the network's own canonical frame — self-consistent as long as training
(doorway 2) feeds actions the same way.

`warmstart.py` needs **no change**: it stores absolute moves/targets and
doorway 2 converts at sample time.

### 3d. Doorway 2 — `ReplayBuffer.make_target`

For each unroll offset `s = t + k`:
- Observation and consistency observation already canonicalize via doorway 1
  (they call `encode_observation` with `to_play_history[s]`).
- **Policy target indices** and **unroll actions** at plies where black moved
  get `flip_action` applied, so they index the same physical moves the flipped
  boards show.
- **Material target** is negated on black-to-move states ("my material minus
  theirs" instead of red-minus-black).
- Value / reward / moves-left targets are already mover-perspective —
  unchanged.

**Mirror augmentation** also lives here: per *sample*, the buffer's existing
seeded RNG flips a coin; if heads, `mirror_board` is applied to all boards
feeding both observations and `mirror_action` to all policy indices and
unroll actions of that sample — always the full sample together, never
partially. Always on; no config knob (an ablation is a one-line revert).
Material and moves-left are mirror-invariant.

### 3e. Config / network

- `MuZeroConfig.__post_init__`: `input_planes = 14 * history_length + 2`.
- No network-code change (first conv reads `input_planes`). **Old checkpoints
  become incompatible** (first-layer shape) — fresh run required; the user
  has accepted this and stopped the in-flight run.

## 4. Error handling

- `encode_observation` keeps its `side_to_move in ("w","b")` assertion.
- `flip_action`/`mirror_action` validate index range (reuse the existing
  0–8100 checks) so corrupted indices raise instead of wrapping silently.
- Mirror coin uses `ReplayBuffer.rng` (seeded) — runs stay reproducible.

## 5. Testing

1. **Involution tests (exhaustive, pure math):** `flip_action∘flip_action =
   id` over all 8100 indices; same for `mirror_action` and compositions;
   `flip_board`/`mirror_board` involutions on random boards.
2. **Semantic tests:** for real positions with black to move (sampled from a
   scripted game covering pawns, palace pieces, captures): flipping the board
   and flipping a legal black move yields a move that is legal on the flipped
   board and lands on the flipped destination square.
3. **Keystone doorway-consistency test:** record a short scripted game the way
   self-play does, run `make_target` over it, and assert that for every black
   ply the policy-target index decoded in the flipped frame names the same
   physical move that was played. Variant with the mirror flag forced on.
   This makes the worst failure mode (boards flipped, targets not — silent
   data poisoning) unable to pass CI.
4. **End-to-end smokes:** existing selfplay/gate/`--smoke` tests re-run under
   the new encoding; tests asserting 115 planes updated to 114.

## 6. Rollout

- Branch `muzero-canonical`; full local suite; on the 5090 box: engine-gated
  tests + `--smoke`, then launch **fresh** (no `--resume`; warmstart runs as
  usual).
- **Success criteria (first ~30 iterations):** `selfplay/red_win_rate` and
  `black_win_rate` track each other closely (no sustained 0.3+ gaps);
  `selfplay/blunder_rate` falls at least as fast as the previous run's
  0.68 → 0.45 trajectory. If it falls slower, stop and investigate — a
  mis-wired flip is the prime suspect.
- Baseline for the writeup: the stopped run (iters 0–80 + resumed segment)
  with checkpoint backup `iter80-prebufferfix.pt`.
