# MuZero Color Canonicalization + Mirror Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonicalize MuZero observations so the network always sees "me at the bottom, moving up, my pieces in planes 0–6" (AlphaZero-style), and add left-right mirror data augmentation.

**Architecture:** Approach A from `docs/superpowers/specs/2026-07-05-muzero-color-canonicalization-design.md` — everything at rest (env, GameHistory, engine, metrics) stays absolute; flips happen only at two doorways (`encode_observation` for boards, `make_target` for actions/policies/material) plus one MCTS-root adapter in selfplay/gate. Old checkpoints become incompatible (115 → 114 input planes); a fresh training run follows.

**Tech Stack:** Python 3.12, numpy, pytest, `uv` for everything (`uv run pytest`, `uv run ruff`). Repo conventions: `ruff check . --fix && ruff format .` before finishing; work on branch `muzero-canonical`.

**Key domain facts an implementer must know:**
- Boards are `(10, 9)` `np.int8`; **row 0 = black's back rank (top), row 9 = red's back rank (bottom)**; red pieces are positive ids 1–16, black negative. Red pawns start row 6, black pawns row 3; red advances toward row 0, black toward row 9.
- Action index formula (`muzero/encoding.py`): `idx = (fr*9 + fc)*90 + (tr*9 + tc)`, 8100 total. Internal algebraic move strings map file letter a–i → column 0–8 and rank digit → row directly (`"a0a1"` = (0,0)→(1,0), index 9).
- `GameHistory` (`muzero/replay_buffer.py`) stores `boards`/`to_play_history`/`rep_history`/`no_progress_history` with L+1 entries and `actions`/`rewards`/`policy_indices`/`policy_probs`/`root_values` with L entries. All absolute — this plan does NOT change what is stored.
- Value/reward targets are already mover-perspective (negamax sign alternation in `n_step_value`); they need **no** change. Only policy indices, unroll actions, and the material target change frames in training.
- The dynamics network embeds actions as from/to one-hot planes (`network.py:action_planes`) — it has no notion of color, so canonical action indices are learnable as long as play time and train time agree.

---

### Task 0: Branch setup

- [ ] **Step 0.1: Create the branch**

```bash
cd "/Users/fionachow/Documents/NYU/CDS/Spring 2024/DS-GA 3001.005 - Reinforcement Learning/Projects"
git checkout main && git pull && git checkout -b muzero-canonical
```

Expected: `Switched to a new branch 'muzero-canonical'`.

---

### Task 1: Transform primitives

**Files:**
- Modify: `muzero/encoding.py` (add four functions after `index_to_move`)
- Test: `muzero/tests/test_encoding.py` (append tests)

- [ ] **Step 1.1: Write the failing tests** — append to `muzero/tests/test_encoding.py`:

```python
def test_flip_action_involutions_exhaustive():
    from muzero.encoding import flip_action, mirror_action

    idx = np.arange(8100, dtype=np.int64)
    assert np.array_equal(flip_action(flip_action(idx)), idx)
    assert np.array_equal(mirror_action(mirror_action(idx)), idx)
    # the two mirrors act on different axes, so they commute
    assert np.array_equal(
        flip_action(mirror_action(idx)), mirror_action(flip_action(idx))
    )


def test_flip_action_scalar_semantics():
    from muzero.encoding import flip_action, mirror_action

    # black pawn push (3,0)->(4,0) flips to red pawn push (6,0)->(5,0)
    assert index_to_move(flip_action(move_to_index("a3a4"))) == "a6a5"
    # left-right mirror: file a -> file i
    assert index_to_move(mirror_action(move_to_index("a3a4"))) == "i3i4"
    assert isinstance(flip_action(move_to_index("a3a4")), int)


def test_flip_action_rejects_bad_indices():
    from muzero.encoding import flip_action, mirror_action

    for bad in (-1, 8100):
        with pytest.raises(ValueError):
            flip_action(bad)
        with pytest.raises(ValueError):
            mirror_action(bad)


def test_flip_board_involution_and_color_swap():
    from muzero.encoding import flip_board, mirror_board

    board = _start_board()
    # the start position is vertically AND horizontally symmetric
    np.testing.assert_array_equal(flip_board(board), board)
    np.testing.assert_array_equal(mirror_board(board), board)
    # asymmetric position: remove a red pawn at (6,0)
    board[6, 0] = 0
    fb = flip_board(board)
    np.testing.assert_array_equal(flip_board(fb), board)  # involution
    assert fb[3, 0] == 0  # the gap lands at the flipped square...
    assert fb[6, 0] < 0  # ...and black's pawn (now negative) sits at (6,0)
    mb = mirror_board(board)
    np.testing.assert_array_equal(mirror_board(mb), board)
    assert mb[6, 8] == 0 and mb[6, 0] > 0
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_encoding.py -v -k "flip or mirror"`
Expected: FAIL / ERROR with `ImportError: cannot import name 'flip_action'`.

- [ ] **Step 1.3: Implement the primitives** — in `muzero/encoding.py`, insert after `index_to_move` (line 45):

```python
def flip_board(board: np.ndarray) -> np.ndarray:
    """Vertical flip + color swap: the same position seen by the other side.

    Rows reverse (r -> 9-r) and signed piece ids negate, so a black-to-move
    position maps into the frame red enjoys (own army nearest row 9)."""
    return np.ascontiguousarray(board[::-1] * -1)


def mirror_board(board: np.ndarray) -> np.ndarray:
    """Left-right mirror (Xiangqi rules are left-right symmetric)."""
    return np.ascontiguousarray(board[:, ::-1])


def _transform_action(idx, row_map, col_map):
    a = np.asarray(idx, dtype=np.int64)
    if np.any(a < 0) or np.any(a >= 8100):
        raise ValueError(f"action index out of range: {idx!r}")
    frm, to = a // 90, a % 90
    fr, fc = frm // 9, frm % 9
    tr, tc = to // 9, to % 9
    out = (row_map(fr) * 9 + col_map(fc)) * 90 + (row_map(tr) * 9 + col_map(tc))
    if isinstance(idx, (int, np.integer)):
        return int(out)
    return out


def flip_action(idx):
    """Top-bottom mirror of a flat action index (rows r -> 9-r).

    Matches flip_board; accepts a python int or an int array (vectorized)."""
    return _transform_action(idx, lambda r: 9 - r, lambda c: c)


def mirror_action(idx):
    """Left-right mirror of a flat action index (cols c -> 8-c).

    Matches mirror_board; accepts a python int or an int array."""
    return _transform_action(idx, lambda r: r, lambda c: 8 - c)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_encoding.py -v`
Expected: all PASS (new flip/mirror tests plus the pre-existing encoding tests, which are untouched by this task).

- [ ] **Step 1.5: Commit**

```bash
git add muzero/encoding.py muzero/tests/test_encoding.py
git commit -m "feat(muzero): board/action flip and mirror transform primitives"
```

---

### Task 2: Doorway 1 — canonical `encode_observation`, 115 → 114 planes

**Files:**
- Modify: `muzero/encoding.py:61-84` (`encode_observation`)
- Modify: `muzero/config.py` (`__post_init__`, `input_planes` comment)
- Modify: `muzero/mcts.py:99` (docstring plane count only)
- Test: `muzero/tests/test_encoding.py`, `muzero/tests/test_config.py`, `muzero/tests/test_env.py:33`, `muzero/tests/test_mcts.py:27-28,72`, `muzero/tests/test_network.py:21`, `muzero/tests/test_replay_buffer.py:53,61`

- [ ] **Step 2.1: Write the failing test** — append to `muzero/tests/test_encoding.py`:

```python
def test_encode_observation_canonicalizes_black_to_move():
    from muzero.encoding import flip_board

    board = _start_board()
    board[6, 0] = 0  # remove a red pawn so the position is asymmetric
    obs_w = encode_observation([board], "w", 1, 0, history_length=8)
    obs_b = encode_observation([board], "b", 1, 0, history_length=8)
    assert obs_w.shape == obs_b.shape == (114, 10, 9)
    # red to move: newest slot is the absolute board
    np.testing.assert_array_equal(obs_w[98:112], board_planes(board))
    # black to move: newest slot is the flipped board (mover's canonical frame)
    np.testing.assert_array_equal(obs_b[98:112], board_planes(flip_board(board)))
    # rep / no-progress broadcast planes moved down to 112 / 113
    assert np.allclose(obs_b[112], 1.0 / 3.0)
    assert np.allclose(obs_b[113], 0.0)
```

- [ ] **Step 2.2: Run it to verify it fails**

Run: `uv run pytest muzero/tests/test_encoding.py::test_encode_observation_canonicalizes_black_to_move -v`
Expected: FAIL — shape is `(115, 10, 9)` and no flip is applied.

- [ ] **Step 2.3: Implement** — replace `encode_observation` in `muzero/encoding.py` with:

```python
def encode_observation(
    boards: list,
    side_to_move: str,
    repetition_count: int,
    no_progress: int,
    history_length: int = 8,
) -> np.ndarray:
    """Stack of the last `history_length` boards in the MOVER's canonical
    frame (oldest first, zero-padded): when black is to move every board is
    flipped so the mover always attacks "up" with its pieces in planes 0-6.
    No side-to-move plane — it is constant under canonicalization. Plus
    repetition / no-progress broadcast planes."""
    assert side_to_move in ("w", "b"), side_to_move
    hist = list(boards)[-history_length:]
    if side_to_move == "b":
        hist = [flip_board(b) for b in hist]
    stacks = [
        np.zeros((14, 10, 9), dtype=np.float32)
        for _ in range(history_length - len(hist))
    ]
    stacks = stacks + [board_planes(b) for b in hist]
    rep = np.full(
        (1, 10, 9), min(max(int(repetition_count), 0), 3) / 3.0, dtype=np.float32
    )
    nop = np.full(
        (1, 10, 9), min(max(int(no_progress), 0), 100) / 100.0, dtype=np.float32
    )
    return np.concatenate(stacks + [rep, nop], axis=0)
```

In `muzero/config.py`, change the derivation and its comment:

```python
    input_planes: int = 0  # derived: 14 * history_length + 2 (set in __post_init__)
```

and in `__post_init__`:

```python
        self.input_planes = 14 * self.history_length + 2
```

In `muzero/mcts.py:99`, update the docstring `(obs (115,10,9) float32, ...)` → `(obs (input_planes,10,9) float32, ...)`.

- [ ] **Step 2.4: Update the plane-count assertions in existing tests**

`muzero/tests/test_encoding.py` — in `test_encode_observation_shape_and_padding` change `(115, 10, 9)` → `(114, 10, 9)` and delete the line `assert obs[112].max() == 1.0  # side-to-move plane (red)`. Replace `test_encode_observation_broadcast_planes` with:

```python
def test_encode_observation_broadcast_planes():
    board = _start_board()
    obs = encode_observation([board], "w", 1, 50, history_length=8)
    assert np.allclose(obs[112], 1.0 / 3.0)  # repetition_count=1
    assert np.allclose(obs[113], 0.5)  # no_progress=50
    obs_b = encode_observation([board], "b", 3, 200, history_length=8)
    assert np.allclose(obs_b[112], 1.0)  # clamped at 3
    assert np.allclose(obs_b[113], 1.0)  # clamped at 100
```

`muzero/tests/test_config.py:15` → `assert cfg.input_planes == 14 * cfg.history_length + 2 == 114`; line 21 (`history_length=4`) → `assert cfg.input_planes == 58`.
`muzero/tests/test_env.py:33` → `assert obs.shape == (114, 10, 9)`.
`muzero/tests/test_mcts.py:27,28,72` → `(114, 10, 9)` in all three `standard_normal` calls.
`muzero/tests/test_network.py:21` → `obs = torch.randn(4, 114, 10, 9)`.
`muzero/tests/test_replay_buffer.py:53,61` → `(8, 114, 10, 9)` for `obs` and `consistency_obs`.

Also in `muzero/replay_buffer.py:150` (the zero consistency-obs fallback) nothing changes — it already uses `cfg.input_planes`.

- [ ] **Step 2.5: Run the full suite**

Run: `uv run pytest muzero/tests -q`
Expected: all pass, 5 engine-gated skips locally.

- [ ] **Step 2.6: Commit**

```bash
git add muzero/encoding.py muzero/config.py muzero/mcts.py muzero/tests
git commit -m "feat(muzero): canonical mover-frame encoding, drop stm plane (115->114)"
```

---

### Task 3: Doorway 2 — `make_target` flips black plies (keystone test)

**Files:**
- Modify: `muzero/replay_buffer.py` (`_dense_policy`, `make_target`; add `_target_action`)
- Test: `muzero/tests/test_replay_buffer.py`

- [ ] **Step 3.1: Write the failing tests** — append to `muzero/tests/test_replay_buffer.py`:

```python
from muzero.encoding import flip_action, move_to_index


def make_alternating_game(length=6):
    """Deterministic game: red plays a6a5, black plays a3a4, one-hot policies.

    Boards hold kings plus one pawn each, with an extra red rook so the
    material balance is nonzero ((9+1-1)/10 = +0.9 red-perspective)."""
    g = GameHistory()
    board = np.zeros((10, 9), dtype=np.int8)
    board[9, 4], board[0, 4] = 1, -1  # kings
    board[6, 0], board[3, 0] = 12, -12  # one pawn each
    board[9, 0] = 8  # extra red rook -> red is +9.0 material
    for t in range(length + 1):
        g.boards.append(board.copy())
        g.to_play_history.append("w" if t % 2 == 0 else "b")
        g.rep_history.append(1)
        g.no_progress_history.append(t)
    for t in range(length):
        a = move_to_index("a6a5") if t % 2 == 0 else move_to_index("a3a4")
        g.actions.append(a)
        g.rewards.append(0.0)
        g.policy_indices.append(np.array([a], dtype=np.int64))
        g.policy_probs.append(np.array([1.0], dtype=np.float32))
        g.root_values.append(0.0)
    g.result = "draw_max_plies"
    return g


def test_make_target_flips_black_plies_keystone():
    """THE doorway-consistency test: for every black ply the action and
    policy target, decoded in the flipped frame, must name the same physical
    move that was played. A boards-flipped-but-targets-not bug cannot pass."""
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    tgt = buf.make_target(g, 0, mirror=False)
    for k in range(cfg.unroll_steps):
        stored = g.actions[k]
        expected = stored if g.to_play_history[k] == "w" else flip_action(stored)
        assert tgt["actions"][k] == expected
        assert int(np.argmax(tgt["target_policy"][k])) == expected
    # black's a3a4 must literally become red's a6a5 in the canonical frame
    assert tgt["actions"][1] == move_to_index("a6a5")


def test_make_target_material_is_mover_perspective():
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    tgt = buf.make_target(g, 0, mirror=False)
    for k in range(cfg.unroll_steps + 1):
        expected = 0.9 if g.to_play_history[k] == "w" else -0.9
        assert tgt["target_material"][k] == np.float32(expected)
```

- [ ] **Step 3.2: Run to verify they fail**

Run: `uv run pytest muzero/tests/test_replay_buffer.py -v -k "keystone or mover_perspective"`
Expected: FAIL — `make_target() got an unexpected keyword argument 'mirror'`.

- [ ] **Step 3.3: Implement** — in `muzero/replay_buffer.py`:

Add imports at the top (replacing the current encoding import line):

```python
from muzero.encoding import (
    encode_observation,
    flip_action,
    material_balance,
    mirror_action,
    mirror_board,
)
```

Replace `_dense_policy` and add `_target_action`:

```python
    def _target_action(self, game: GameHistory, s: int, mirror: bool) -> int:
        """Stored absolute action -> the frame the network sees at state s."""
        a = game.actions[s]
        if mirror:
            a = mirror_action(a)
        if game.to_play_history[s] == "b":
            a = flip_action(a)
        return a

    def _dense_policy(self, game: GameHistory, t: int, mirror: bool) -> np.ndarray:
        idx = game.policy_indices[t]
        if mirror:
            idx = mirror_action(idx)
        if game.to_play_history[t] == "b":
            idx = flip_action(idx)
        dense = np.zeros(self.config.action_space, dtype=np.float32)
        dense[idx] = game.policy_probs[t]
        s = dense.sum()
        return dense / s if s > 0 else np.full_like(dense, 1.0 / dense.shape[0])
```

Replace `make_target` (signature + body; the `mirror` parameter defaults to a coin flip — Task 4's augmentation — but this task always passes/receives it explicitly in tests, so implement the full signature now):

```python
    def make_target(self, game: GameHistory, t: int, mirror: bool | None = None) -> dict:
        cfg = self.config
        K, L = cfg.unroll_steps, len(game)
        if mirror is None:
            mirror = bool(self.rng.integers(0, 2))  # LR-mirror augmentation
        boards = [mirror_board(b) for b in game.boards] if mirror else game.boards
        uniform = np.full(cfg.action_space, 1.0 / cfg.action_space, dtype=np.float32)

        obs = encode_observation(
            boards[: t + 1],
            game.to_play_history[t],
            game.rep_history[t],
            game.no_progress_history[t],
            cfg.history_length,
        )
        actions = np.array(
            [
                (
                    self._target_action(game, t + k, mirror)
                    if t + k < L
                    else int(self.rng.integers(0, cfg.action_space))
                )
                for k in range(K)
            ],
            dtype=np.int64,
        )
        policies, pmask, values, moves_left, material = [], [], [], [], []
        for k in range(K + 1):
            s = t + k
            if s < L:
                policies.append(self._dense_policy(game, s, mirror))
                pmask.append(1.0)
                values.append(self.n_step_value(game, s))
            else:
                policies.append(uniform)
                pmask.append(0.0)
                values.append(0.0)  # absorbing states train value to 0
            moves_left.append(min(max(L - s, 0), cfg.moves_left_max))
            sb = min(s, L)
            mat = material_balance(game.boards[sb]) / 10.0  # mirror-invariant
            if game.to_play_history[sb] == "b":
                mat = -mat  # mover-perspective, matching the canonical frame
            material.append(mat)
        rewards = np.array(
            [game.rewards[t + k] if t + k < L else 0.0 for k in range(K)],
            dtype=np.float32,
        )
        # SimSiam target: one random real future observation within the unroll.
        max_kc = min(K, L - t)
        k_c = int(self.rng.integers(1, max_kc + 1)) if max_kc >= 1 else 0
        c_obs = (
            encode_observation(
                boards[: t + k_c + 1],
                game.to_play_history[t + k_c],
                game.rep_history[t + k_c],
                game.no_progress_history[t + k_c],
                cfg.history_length,
            )
            if k_c > 0
            else np.zeros((cfg.input_planes, 10, 9), dtype=np.float32)
        )
        return {
            "obs": obs,
            "actions": actions,
            "target_policy": np.stack(policies),
            "policy_mask": np.array(pmask, dtype=np.float32),
            "target_value": np.array(values, dtype=np.float32),
            "target_reward": rewards,
            "target_moves_left": np.array(moves_left, dtype=np.int64),
            "target_material": np.array(material, dtype=np.float32),
            "consistency_obs": c_obs,
            "consistency_k": k_c,
        }
```

Note: `add()`'s priority computation uses only `root_values`/`n_step_value` (frame-free) and `sample_batch` calls `make_target(game, t)` positionally — both untouched.

- [ ] **Step 3.4: Run the full replay-buffer file, then the whole suite**

Run: `uv run pytest muzero/tests/test_replay_buffer.py -v && uv run pytest muzero/tests -q`
Expected: all pass. (`test_sample_batch_shapes` still passes: flipped/mirrored indices are bijections, so policy rows still sum to 1.)

- [ ] **Step 3.5: Commit**

```bash
git add muzero/replay_buffer.py muzero/tests/test_replay_buffer.py
git commit -m "feat(muzero): make_target emits mover-canonical actions/policies/material"
```

---

### Task 4: Mirror augmentation test (behavior already implemented in Task 3)

Task 3 implemented the `mirror` parameter and per-sample coin flip; this task pins the augmentation's correctness so a partial mirror (boards without targets, or vice versa) cannot pass.

**Files:**
- Test: `muzero/tests/test_replay_buffer.py`

- [ ] **Step 4.1: Write the failing-or-passing test (it must pass if Task 3 is correct; write it and verify)**

```python
def test_make_target_mirror_consistency():
    from muzero.encoding import mirror_action

    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    buf = ReplayBuffer(cfg)
    g = make_alternating_game(length=6)
    buf.rng = np.random.default_rng(7)
    plain = buf.make_target(g, 0, mirror=False)
    buf.rng = np.random.default_rng(7)  # same k_c draw for both calls
    mirrored = buf.make_target(g, 0, mirror=True)
    # observations mirror along the column axis, plane-for-plane
    np.testing.assert_array_equal(mirrored["obs"], plain["obs"][:, :, ::-1])
    np.testing.assert_array_equal(
        mirrored["consistency_obs"], plain["consistency_obs"][:, :, ::-1]
    )
    # every action / policy target mirrors with the boards, never without
    for k in range(cfg.unroll_steps):
        assert mirrored["actions"][k] == mirror_action(int(plain["actions"][k]))
        assert int(np.argmax(mirrored["target_policy"][k])) == mirror_action(
            int(np.argmax(plain["target_policy"][k]))
        )
    # frame-independent targets are identical
    np.testing.assert_array_equal(mirrored["target_value"], plain["target_value"])
    np.testing.assert_array_equal(
        mirrored["target_material"], plain["target_material"]
    )


def test_make_target_default_draws_mirror_from_seeded_rng():
    cfg = replace(MuZeroConfig(), unroll_steps=4, td_steps=2)
    g = make_alternating_game(length=6)
    buf_a, buf_b = ReplayBuffer(cfg), ReplayBuffer(cfg)
    a = buf_a.make_target(g, 0)
    b = buf_b.make_target(g, 0)
    np.testing.assert_array_equal(a["obs"], b["obs"])  # same seed -> same draw
```

- [ ] **Step 4.2: Run and verify both pass**

Run: `uv run pytest muzero/tests/test_replay_buffer.py -v -k mirror`
Expected: PASS. If `test_make_target_mirror_consistency` fails, the mirror wiring from Task 3 is partial — fix `make_target`/`_dense_policy`/`_target_action` before proceeding; do not weaken the test.

- [ ] **Step 4.3: Commit**

```bash
git add muzero/tests/test_replay_buffer.py
git commit -m "test(muzero): pin mirror augmentation obs/target consistency"
```

---

### Task 5: MCTS-root adapter — selfplay + gate flip legal in / visits out

**Files:**
- Modify: `muzero/encoding.py` (add `absolute_visits`)
- Modify: `muzero/selfplay.py` (`generate`)
- Modify: `muzero/train.py` (`_run_gate_rung`)
- Test: `muzero/tests/test_encoding.py`, `muzero/tests/test_selfplay.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `muzero/tests/test_encoding.py`:

```python
def test_absolute_visits():
    from muzero.encoding import absolute_visits, flip_action

    visits = {move_to_index("a6a5"): 7, move_to_index("e6e5"): 3}
    assert absolute_visits(visits, "w") is visits  # red: identity, no copy
    unflipped = absolute_visits(visits, "b")
    assert unflipped == {
        flip_action(move_to_index("a6a5")): 7,
        flip_action(move_to_index("e6e5")): 3,
    }
```

Append to `muzero/tests/test_selfplay.py` (uses the tiny-net pattern from `test_selfplay_smoke_generates_games` but with `FakeEvaluator`, so it runs without an engine; the scripted game is: forced red opening, then black's only legal move `a3a4`, then `max_game_plies=2` ends it):

```python
def test_generate_stores_absolute_actions_for_black():
    """If the root adapter forgets to unflip black's chosen move, the stored
    action is flip_action(a3a4)=a6a5, which this test rejects."""
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        num_workers=1,
        games_per_worker=1,
        max_game_plies=2,
        device="cpu",
    )
    torch.manual_seed(0)
    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=lambda fen: ["a3a4"])
    ally = MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    coord = SelfPlayCoordinator(cfg, ally, ally)
    runner = NetRunner(ally, "cpu")
    worker = SelfPlayWorker(cfg, runner, runner, buf, coord, evaluator, worker_id=0)
    summaries = worker.generate(1)
    assert len(summaries) == 1
    game = buf.games[0]
    # ply 0 is the forced red opening (absolute by construction);
    # ply 1 is black's MCTS move and must be stored ABSOLUTE:
    assert game.actions[1] == move_to_index("a3a4")
    assert list(game.policy_indices[1]) == [move_to_index("a3a4")]
```

(Match the existing imports at the top of `test_selfplay.py` — it already imports `replace`, `torch`, `MuZeroConfig`, `MuZeroNet`, `ReplayBuffer`, `SelfPlayCoordinator`, `SelfPlayWorker`, `NetRunner`, `FakeEvaluator`, and `move_to_index`; add any of these that are missing.)

- [ ] **Step 5.2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_encoding.py::test_absolute_visits muzero/tests/test_selfplay.py::test_generate_stores_absolute_actions_for_black -v`
Expected: `test_absolute_visits` FAILS with ImportError. The selfplay test FAILS because black's stored action is the *flipped* index (the net now emits canonical-frame moves and nothing unflips them — this is exactly the bug class the adapter fixes). If it passes at this point, STOP: doorways 1–2 are inconsistent somewhere; investigate before continuing.

- [ ] **Step 5.3: Implement**

In `muzero/encoding.py`, after `mirror_action`:

```python
def absolute_visits(visits: dict, side_to_move: str) -> dict:
    """MCTS root visits keyed in the mover's canonical frame -> absolute."""
    if side_to_move == "w":
        return visits
    return {flip_action(a): v for a, v in visits.items()}
```

In `muzero/selfplay.py`: extend the encoding import to include `absolute_visits, flip_action`, and in `generate()` replace the root-building and result-consuming block with:

```python
                roots = []
                for g in group:
                    legal = np.array(
                        [move_to_index(m) for m in g.env.legal_moves()], dtype=np.int64
                    )
                    if g.env.side_to_move == "b":
                        legal = flip_action(legal)  # canonical frame in ...
                    roots.append((g.env.observation().astype(np.float32), legal))
                results = self.mcts.run(runner, roots, add_noise=add_noise)
                for g, (visits, root_value, search_kl) in zip(group, results):
                    # ... absolute frame out (before storing or stepping)
                    visits = absolute_visits(visits, g.env.side_to_move)
                    action = select_action(
                        visits, g.env.plies, self.cfg.temperature_moves, self.rng
                    )
```

(the rest of the loop body is unchanged — `_record_and_step` continues to receive absolute actions/visits, so `GameHistory` stays absolute at rest.)

In `muzero/train.py` `_run_gate_rung`, extend the local encoding import to `from muzero.encoding import absolute_visits, flip_action, index_to_move, move_to_index` and replace the ally branch with:

```python
            if env.side_to_move == ally_side:
                legal = np.array(
                    [move_to_index(m) for m in env.legal_moves()], dtype=np.int64
                )
                if env.side_to_move == "b":
                    legal = flip_action(legal)
                ((visits, _, _),) = mcts.run(
                    runner,
                    [(env.observation().astype(np.float32), legal)],
                    add_noise=False,
                )
                visits = absolute_visits(visits, env.side_to_move)
                move = index_to_move(max(visits, key=visits.get))
```

- [ ] **Step 5.4: Run the full suite**

Run: `uv run pytest muzero/tests -q`
Expected: all pass, 5 engine-gated skips locally.

- [ ] **Step 5.5: Commit**

```bash
git add muzero/encoding.py muzero/selfplay.py muzero/train.py muzero/tests
git commit -m "feat(muzero): MCTS-root canonical adapter for selfplay and gate"
```

---

### Task 6: Engine-gated semantic test — flip maps legal-move sets exactly

**Files:**
- Test: `muzero/tests/test_env.py` (append; uses existing `requires_engine`/`make_evaluator` from `muzero/tests/helpers.py`)

- [ ] **Step 6.1: Write the test**

```python
@requires_engine
def test_flip_maps_legal_move_sets():
    """Flipping the board + flipping each legal move must yield exactly the
    legal-move set of the flipped position (colors swapped, red to move)."""
    from muzero.encoding import flip_action, flip_board, move_to_index
    from src.xiangqi_board import board_to_uci_fen, engine_uci_to_algebraic

    evaluator = make_evaluator()
    cfg = MuZeroConfig()
    env = XiangqiEnv(cfg, evaluator)
    env.reset()
    env.step(engine_uci_to_algebraic("h2e2"))  # red central cannon; black to move
    black_legal = {move_to_index(m) for m in env.legal_moves()}
    flipped_fen = board_to_uci_fen(flip_board(env.board), "w")
    flipped_legal = {
        move_to_index(engine_uci_to_algebraic(u))
        for u in evaluator.list_legal_moves(flipped_fen)
    }
    assert {flip_action(a) for a in black_legal} == flipped_legal
```

(Match `test_env.py`'s existing imports — it already has `MuZeroConfig`, `XiangqiEnv`, and the helpers; add the `requires_engine`/`make_evaluator` import if not present.)

- [ ] **Step 6.2: Run locally (skips) and note for the training box**

Run: `uv run pytest muzero/tests/test_env.py -v -k flip_maps`
Expected locally: SKIPPED (no engine). This test runs for real on the 5090 box in Task 7.

- [ ] **Step 6.3: Commit**

```bash
git add muzero/tests/test_env.py
git commit -m "test(muzero): engine-gated legal-set flip bijection test"
```

---

### Task 7: Lint, docs, merge, and fresh-run launch checklist

**Files:**
- Modify: `docs/ARCHITECTURE.md` (§3f), `docs/AGENT_TODO.md`
- Create: `docs/logs/2026-07-05-log-color-canonicalization.md` (use `docs/logs/template.md`)

- [ ] **Step 7.1: Lint + full suite**

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pytest muzero/tests -q
```

Expected: ruff clean; all tests pass (5 engine-gated skips locally).

- [ ] **Step 7.2: Update docs**

- `docs/ARCHITECTURE.md` §3f: change "115×10×9 board tensors" to "114×10×9 mover-canonical board tensors (board flipped + colors swapped when black is to move; no side-to-move plane; LR-mirror augmentation at target time)" and note that `GameHistory` stays absolute with flips at `encode_observation`/`make_target`/MCTS-root only.
- `docs/AGENT_TODO.md`: close the "color-canonicalize `encoding.py`" backlog bullet (move under Completed with a pointer to the spec + log); add a follow-up bullet: "MuZero: human-vs-MuZero play adapter for the web UI (user goal: play against the net)".
- Write `docs/logs/2026-07-05-log-color-canonicalization.md` per the template: hypothesis (kill red/black skill split, ~2× signal), config changes (input_planes 114, canonical encode, make_target flips + mirror aug, root adapter), run command (tests only locally; fresh training launch on the box), results (test counts), next steps (launch + watch criteria below).

- [ ] **Step 7.3: Commit docs, merge to main, push**

```bash
git add docs
git commit -m "docs(muzero): canonicalization log + architecture/TODO updates"
git checkout main && git merge --no-ff muzero-canonical -m "merge: muzero color canonicalization + mirror augmentation" && git push
```

- [ ] **Step 7.4: Launch checklist for the 5090 box (user runs these)**

```bash
cd ~/Documents/FineTune-LLM-OnlineRL && git pull
PIKAFISH_BIN=<path> uv run pytest muzero/tests -q        # engine tests incl. legal-set flip
uv run python -m muzero.train --smoke --no-wandb --iterations 1 --device cpu
uv run python -m muzero.train                            # FRESH run — no --resume (old ckpts are 115-plane)
```

Watch criteria (first ~30 iterations, from the spec): `selfplay/red_win_rate` ≈ `black_win_rate` (no sustained 0.3+ gaps); `selfplay/blunder_rate` falls at least as fast as the previous run's 0.68 → 0.45; `gate/win_rate_random` → ~1.0. If blunder rate falls *slower* than the old run, stop and investigate a mis-wired flip first.

---

## Self-Review (completed)

- **Spec coverage:** primitives → Task 1; doorway 1 + 114 planes → Task 2; doorway 2 (actions/policies/material) → Task 3; mirror aug → Tasks 3–4; MCTS-root adapter (selfplay + gate) → Task 5; warmstart needs no change (stores absolute; doorway 2 converts) — verified against `warmstart.py:119-139`; testing layers 1/2/3/4 → Tasks 1/6/3–4/7; rollout + success criteria → Task 7.
- **Known limitation (accepted in spec discussion):** the Task 5 selfplay test catches a missing *unflip* (out-direction); a missing flip-*in* of root legal moves is not mechanically catchable without a trained net — mitigated by keeping both directions adjacent in one code block and by the Task 7 watch criteria.
- **Type consistency:** `flip_action`/`mirror_action` accept int or ndarray and return matching type (Task 1 definition; used both ways in Tasks 3 and 5); `make_target(game, t, mirror=None)` signature consistent across Tasks 3–4; `absolute_visits(visits, side_to_move)` consistent between Tasks 5's encoding, selfplay, and train call sites.
