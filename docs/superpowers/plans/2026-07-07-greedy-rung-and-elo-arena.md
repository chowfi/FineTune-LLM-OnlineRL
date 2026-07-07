# Greedy Gate Rung + Checkpoint Elo Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A capture-greedy gate opponent (`gate/*_greedy` metrics) plus checkpoint archiving every 20 iterations and an offline arena tool that fits a relative Elo curve over archived checkpoints.

**Architecture:** Per `docs/superpowers/specs/2026-07-07-greedy-gate-rung-design.md` and `docs/superpowers/specs/2026-07-07-checkpoint-elo-arena-design.md`. The greedy opponent is a pure function slotted into the existing `_run_gate_rung`; archiving is a small helper called from the training loop; the arena is a standalone `python -m muzero.arena` that replays the gate's exact play path for both seats and reuses `scripts/benchmark/elo_estimator.fit_ratings` (verified importable; result convention is `"win"|"loss"|"draw"` from WHITE's perspective per `_result_signs`).

**Tech Stack:** Python 3.12, numpy, torch, pytest, `uv`. Conventions: `uv run ruff check muzero --fix && uv run ruff format muzero` before each commit; commit messages end with a blank line then `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Key domain facts:**
- `muzero/train.py` `run_gate` currently builds `random_move`/`engine_move` closures and calls `_run_gate_rung(cfg, runner, evaluator, opponent_move)` → `(wins, draws)`; `_run_gate_rung` plays `cfg.gate_games` games with alternating ally colors and treats an opponent returning `None` as game-abort.
- The play path both gate and arena must use: `canonical_root(env)` (muzero/selfplay.py) → `MCTS.run(runner, [(obs, legal)], add_noise=False)` → `absolute_visits(visits, env.side_to_move)` → argmax → `index_to_move`.
- `muzero/encoding.py` exports `PIECE_TYPE` (abs piece id → type index; king=0) and `PIECE_VALUE` (type index → value; king 0.0, rook 9.0, cannon 4.5, horse 4.0, elephant/advisor 2.0, pawn 1.0).
- `src/xiangqi_board.py` exports `algebraic_to_board_coords(move) -> ((fr,fc),(tr,tc))` and `engine_uci_to_algebraic`. `env.board` is signed int8, positive=red. `cfg.opening_book` is a 10-tuple of ENGINE-UCI red first moves.
- Checkpoint block in the training loop (muzero/train.py:388-400) saves `latest.pt` atomically each iteration; `ckpt_data` includes `"iteration": it + 1`.
- `muzero/tests/helpers.py` has `FakeEvaluator(cp_fn=..., legal_fn=...)`; fake legal moves are ENGINE-UCI strings (env converts rank r → 9−r: engine `"a3a4"` ⇒ internal `"a6a5"` red pawn push; engine `"i6i5"` ⇒ internal `"i3i4"` black pawn push). `env.fen()`'s second token is the side to move.
- Existing test suite baseline: `uv run pytest muzero/tests web/tests -q` → 80 passed, 6 skipped.

---

### Task 0: Branch

- [ ] **Step 0.1:**

```bash
cd "/Users/fionachow/Documents/NYU/CDS/Spring 2024/DS-GA 3001.005 - Reinforcement Learning/Projects"
git checkout main && git pull --ff-only && git checkout -b muzero-gate-arena
```

---

### Task 1: `greedy_capture_move` (pure function + tests)

**Files:**
- Create: `muzero/gate_opponents.py`
- Test: `muzero/tests/test_gate_opponents.py`

- [ ] **Step 1.1: Write the failing tests** — create `muzero/tests/test_gate_opponents.py`:

```python
from dataclasses import replace

import numpy as np

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from muzero.gate_opponents import greedy_capture_move
from muzero.tests.helpers import FakeEvaluator


def make_env(legal_fn, cp_fn=lambda fen: 0.0):
    cfg = replace(MuZeroConfig(), device="cpu")
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=cp_fn, legal_fn=legal_fn))
    env.reset()
    return env


def test_prefers_highest_value_capture():
    # Engine-UCI legals convert to internal: "b7e7" -> "b2e2"?? NO — use
    # moves whose INTERNAL destination we control. Internal red cannon at
    # (7,1) is board value 10; internal (2,1) holds black cannon -10 and
    # (0,0) holds black rook -8 on the start board. We offer red two
    # captures: cannon takes cannon vs cannon takes rook... The start
    # board's own piece values are what greedy reads, so craft legals as
    # ENGINE-UCI whose internal targets are: (3,0) black pawn (-12, value
    # 1.0) and (0,1) black horse (-6, value 4.0).
    # internal (3,0) == engine rank 6 file a; internal (0,1) == engine rank 9 file b.
    env = make_env(lambda fen: ["a6a6", "b9b9"])  # placeholder — see step note
    ...
```

STOP — the above sketch shows the trap: crafting engine-UCI strings whose
internal DESTINATIONS land on specific start-board pieces is error-prone in
prose. Instead write the tests directly against a HAND-BUILT board, bypassing
env: `greedy_capture_move` only needs `env.legal_moves()` and `env.board`, so
use a minimal stub object. Final test file:

```python
import numpy as np

from muzero.gate_opponents import greedy_capture_move


class StubEnv:
    """greedy_capture_move only touches .legal_moves() and .board."""

    def __init__(self, board, moves):
        self.board = board
        self._moves = list(moves)

    def legal_moves(self):
        return list(self._moves)


def empty_board():
    return np.zeros((10, 9), dtype=np.int8)


def test_prefers_highest_value_capture():
    board = empty_board()
    board[4, 4] = 10  # red cannon (the mover's piece; value irrelevant)
    board[4, 0] = -12  # black pawn  (value 1.0) at internal a5... row 4 col 0
    board[4, 8] = -8  # black rook  (value 9.0) at row 4 col 8
    # internal algebraic: col a-i = 0-8, rank digit = row. (4,0)="a4", (4,8)="i4".
    moves = ["e4a4", "e4i4"]  # capture pawn vs capture rook
    rng = np.random.default_rng(0)
    for _ in range(5):  # deterministic regardless of rng draws
        assert greedy_capture_move(StubEnv(board, moves), rng) == "e4i4"


def test_tie_between_equal_captures_stays_in_tied_set():
    board = empty_board()
    board[4, 4] = 10
    board[4, 0] = -12  # pawn
    board[4, 8] = -13  # pawn (same value)
    moves = ["e4a4", "e4i4", "e4e5"]  # two pawn captures + one quiet move
    rng = np.random.default_rng(0)
    picks = {greedy_capture_move(StubEnv(board, moves), rng) for _ in range(20)}
    assert picks <= {"e4a4", "e4i4"}  # never the quiet move
    assert len(picks) == 2  # both ties reachable


def test_no_captures_falls_back_to_random_legal():
    board = empty_board()
    board[4, 4] = 10
    moves = ["e4e5", "e4e3", "e4d4"]
    rng = np.random.default_rng(0)
    picks = {greedy_capture_move(StubEnv(board, moves), rng) for _ in range(30)}
    assert picks <= set(moves)
    assert len(picks) > 1  # actually random, not pinned


def test_empty_legal_list_returns_none():
    assert greedy_capture_move(StubEnv(empty_board(), []), np.random.default_rng(0)) is None


def test_black_greedy_captures_red_pieces():
    board = empty_board()
    board[5, 4] = -10  # black cannon (mover)
    board[5, 0] = 12  # red pawn (value 1.0)
    board[5, 8] = 8  # red rook (value 9.0)
    moves = ["e5a5", "e5i5"]
    rng = np.random.default_rng(0)
    assert greedy_capture_move(StubEnv(board, moves), rng) == "e5i5"


def test_king_capture_outranks_everything():
    board = empty_board()
    board[4, 4] = 10
    board[4, 0] = -8  # rook (9.0)
    board[4, 8] = -1  # king
    moves = ["e4a4", "e4i4"]
    rng = np.random.default_rng(0)
    assert greedy_capture_move(StubEnv(board, moves), rng) == "e4i4"
```

- [ ] **Step 1.2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_gate_opponents.py -v`
Expected: `ModuleNotFoundError: No module named 'muzero.gate_opponents'`.

- [ ] **Step 1.3: Implement** — create `muzero/gate_opponents.py`:

```python
"""Engine-free gate opponents (spec 2026-07-07-greedy-gate-rung-design)."""

from __future__ import annotations

import numpy as np

from muzero.encoding import PIECE_TYPE, PIECE_VALUE
from src.xiangqi_board import algebraic_to_board_coords

# PIECE_VALUE rates kings 0.0 (material-balance convention); for a greedy
# opponent, taking the king ends the game and must outrank any material.
_KING_CAPTURE_VALUE = 100.0


def greedy_capture_move(env, rng: np.random.Generator) -> str | None:
    """Highest-value capture if any (ties broken randomly), else a uniform
    random legal move. Returns None when there are no legal moves.

    Only reads `env.legal_moves()` and `env.board` (signed int8, +red)."""
    moves = env.legal_moves()
    if not moves:
        return None
    best_value = 0.0
    best_moves: list[str] = []
    for move in moves:
        (_, _), (tr, tc) = algebraic_to_board_coords(move)
        target = int(env.board[tr, tc])
        if target == 0:
            continue
        if abs(target) == 1:
            value = _KING_CAPTURE_VALUE
        else:
            value = PIECE_VALUE[PIECE_TYPE[abs(target)]]
        if value > best_value:
            best_value, best_moves = value, [move]
        elif value == best_value:
            best_moves.append(move)
    pool = best_moves if best_moves else moves
    return str(rng.choice(pool))
```

- [ ] **Step 1.4: Run tests**

Run: `uv run pytest muzero/tests/test_gate_opponents.py -v` → 6 passed. Then `uv run ruff check muzero --fix && uv run ruff format muzero`.

- [ ] **Step 1.5: Commit**

```bash
git add muzero/gate_opponents.py muzero/tests/test_gate_opponents.py
git commit -m "feat(muzero): capture-greedy gate opponent"
```

---

### Task 2: Wire the greedy rung into `run_gate`

**Files:**
- Modify: `muzero/train.py` (`run_gate`, ~lines 186-223)
- Test: `muzero/tests/test_gate_opponents.py` (append one integration test)

- [ ] **Step 2.1: Write the failing test** — append to `muzero/tests/test_gate_opponents.py`:

```python
def test_gate_rung_plays_greedy_without_engine(tmp_path):
    """_run_gate_rung + greedy opponent runs on FakeEvaluator (no Pikafish)."""
    import torch
    from dataclasses import replace

    from muzero.config import MuZeroConfig
    from muzero.mcts import NetRunner
    from muzero.network import MuZeroNet
    from muzero.tests.helpers import FakeEvaluator
    from muzero.train import _run_gate_rung

    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        gate_games=2,
        max_game_plies=2,
        device="cpu",
    )
    torch.manual_seed(0)
    runner = NetRunner(MuZeroNet(cfg), "cpu")

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    rng = np.random.default_rng(0)

    def greedy(env):
        return greedy_capture_move(env, rng)

    wins, draws = _run_gate_rung(cfg, runner, evaluator, greedy)
    assert 0 <= wins + draws <= cfg.gate_games  # both 2-ply games complete
    assert draws == 2  # max_game_plies=2 -> both games drawn at the cap
```

- [ ] **Step 2.2: Run to verify it passes or fails meaningfully**

Run: `uv run pytest muzero/tests/test_gate_opponents.py::test_gate_rung_plays_greedy_without_engine -v`
This test exercises EXISTING `_run_gate_rung` machinery, so it should PASS already — it is a regression net for the wiring change, not a red-first test. Confirm it passes; if it fails, STOP and report.

- [ ] **Step 2.3: Modify `run_gate` in `muzero/train.py`.** Update the docstring's ladder description, add the greedy rung between random and Pikafish, and add the three metrics. Replace the body from `rng = np.random.default_rng(cfg.seed)` through the `return {...}` with:

```python
    rng = np.random.default_rng(cfg.seed)
    # Separate generator so adding the greedy rung does not perturb the
    # random rung's historical move sequence.
    greedy_rng = np.random.default_rng(cfg.seed + 1)

    def random_move(env):
        moves = env.legal_moves()
        return str(rng.choice(moves)) if moves else None

    def greedy_move(env):
        return greedy_capture_move(env, greedy_rng)

    engine = SimpleUciEngine(cfg.pikafish_bin, cfg.gate_movetime_ms, multipv=1)

    def engine_move(env):
        lines = engine.search(env.fen())
        if not lines:
            return None
        return engine_uci_to_algebraic(lines[0][0])

    n = cfg.gate_games
    try:
        rand_wins, rand_draws = _run_gate_rung(cfg, runner, evaluator, random_move)
        greedy_wins, greedy_draws = _run_gate_rung(cfg, runner, evaluator, greedy_move)
        pika_wins, pika_draws = _run_gate_rung(cfg, runner, evaluator, engine_move)
    finally:
        engine.close()
    return {
        "gate/win_rate_random": rand_wins / n,
        "gate/draw_rate_random": rand_draws / n,
        "gate/loss_rate_random": (n - rand_wins - rand_draws) / n,
        "gate/win_rate_greedy": greedy_wins / n,
        "gate/draw_rate_greedy": greedy_draws / n,
        "gate/loss_rate_greedy": (n - greedy_wins - greedy_draws) / n,
        "gate/win_rate": pika_wins / n,
        "gate/draw_rate": pika_draws / n,
        "gate/loss_rate": (n - pika_wins - pika_draws) / n,
    }
```

Add `from muzero.gate_opponents import greedy_capture_move` to `run_gate`'s function-local imports (next to the `SimpleUciEngine` import). Update the docstring to mention three rungs.

- [ ] **Step 2.4: Full suite**

Run: `uv run pytest muzero/tests web/tests -q` → 87 passed (80 + 7 new), 6 skipped. Then ruff check/format.

- [ ] **Step 2.5: Commit**

```bash
git add muzero/train.py muzero/tests/test_gate_opponents.py
git commit -m "feat(muzero): greedy rung in the gate ladder (gate/*_greedy)"
```

---

### Task 3: Checkpoint archiving every 20 iterations

**Files:**
- Modify: `muzero/config.py` (add `checkpoint_archive_every`), `muzero/train.py` (helper + call in main loop, checkpoint block ~lines 388-400)
- Test: `muzero/tests/test_train.py` (append)

- [ ] **Step 3.1: Write the failing test** — append to `muzero/tests/test_train.py` (check its existing imports; it already uses `MuZeroConfig`/tiny configs — add imports as needed following file style):

```python
def test_maybe_archive_checkpoint(tmp_path):
    from dataclasses import replace

    import torch

    from muzero.config import MuZeroConfig
    from muzero.network import MuZeroNet
    from muzero.train import maybe_archive_checkpoint

    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        checkpoint_archive_every=20,
    )
    net = MuZeroNet(cfg)
    assert maybe_archive_checkpoint(cfg, net, iteration=19) is None
    path = maybe_archive_checkpoint(cfg, net, iteration=20)
    assert path is not None and path.endswith("archive/iter_0020.pt")
    ckpt = torch.load(path, map_location="cpu")
    assert set(ckpt) == {"ally", "iteration"}
    assert ckpt["iteration"] == 20
    # disabled when the interval is 0
    cfg0 = replace(cfg, checkpoint_archive_every=0)
    assert maybe_archive_checkpoint(cfg0, net, iteration=20) is None
```

- [ ] **Step 3.2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_train.py::test_maybe_archive_checkpoint -v`
Expected: ImportError (`maybe_archive_checkpoint` doesn't exist) or TypeError on the unknown config field.

- [ ] **Step 3.3: Implement.** In `muzero/config.py`, under the `# Misc` section next to `checkpoint_dir`:

```python
    # Every N iterations, also save a permanent ally-weights snapshot to
    # checkpoint_dir/archive/iter_NNNN.pt for the Elo arena. 0 disables.
    checkpoint_archive_every: int = 20
```

In `muzero/train.py`, add module-level (after `load_checkpoint`):

```python
def maybe_archive_checkpoint(cfg: MuZeroConfig, ally, iteration: int):
    """Ally-weights-only snapshot for the Elo arena (spec 2026-07-07)."""
    every = cfg.checkpoint_archive_every
    if every <= 0 or iteration % every != 0:
        return None
    archive_dir = os.path.join(cfg.checkpoint_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, f"iter_{iteration:04d}.pt")
    tmp = path + ".tmp"
    torch.save({"ally": ally.state_dict(), "iteration": iteration}, tmp)
    os.replace(tmp, path)
    return path
```

(`os` and `torch` are already imported in train.py's main/module scope — verify `import os` is module-level; if it is only inside `main()`, add it module-level.) In the main loop, right after the existing `os.replace(tmp_path, ckpt_path)` line for `latest.pt`, add:

```python
        maybe_archive_checkpoint(cfg, ally, it + 1)
```

- [ ] **Step 3.4: Run** `uv run pytest muzero/tests/test_train.py -v` then the full suite (88 passed, 6 skipped) + ruff.

- [ ] **Step 3.5: Commit**

```bash
git add muzero/config.py muzero/train.py muzero/tests/test_train.py
git commit -m "feat(muzero): archive ally checkpoints every 20 iters for the arena"
```

---

### Task 4: The arena (`muzero/arena.py`)

**Files:**
- Create: `muzero/arena.py`
- Test: `muzero/tests/test_arena.py`

- [ ] **Step 4.1: Write the failing tests** — create `muzero/tests/test_arena.py`:

```python
import json
from dataclasses import replace

import numpy as np
import torch

from muzero.arena import (
    discover_checkpoints,
    fit_arena_elo,
    games_needed,
    play_pair,
)
from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.tests.helpers import FakeEvaluator


def tiny_cfg(**over):
    defaults = dict(
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=2,
        interior_topk=2,
        max_game_plies=2,
        device="cpu",
    )
    defaults.update(over)
    return replace(MuZeroConfig(), **defaults)


def save_tiny(tmp_path, name, iteration, seed):
    torch.manual_seed(seed)
    net = MuZeroNet(tiny_cfg())
    path = tmp_path / name
    torch.save({"ally": net.state_dict(), "iteration": iteration}, path)
    return path


def fake_evaluator():
    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    return FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)


def test_discover_checkpoints_sorts_by_iteration(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    save_tiny(archive, "iter_0040.pt", 40, seed=1)
    save_tiny(archive, "iter_0020.pt", 20, seed=2)
    extra = save_tiny(tmp_path, "iter80-prebufferfix.pt", 81, seed=3)
    found = discover_checkpoints(str(archive), extras=[str(extra)])
    assert [c.label for c in found] == ["iter_0020", "iter_0040", "iter80-prebufferfix"]
    assert [c.iteration for c in found] == [20, 40, 81]


def test_games_needed_respects_existing_rows(tmp_path):
    rows = [
        {"white": "a", "black": "b", "result": "draw", "sims": 2},
        {"white": "b", "black": "a", "result": "win", "sims": 2},
        {"white": "a", "black": "b", "result": "win", "sims": 800},  # other sims
    ]
    assert games_needed(rows, "a", "b", sims=2, games_per_pair=4) == 2
    assert games_needed(rows, "a", "b", sims=800, games_per_pair=4) == 3


def test_play_pair_writes_valid_rows(tmp_path):
    a = save_tiny(tmp_path, "iter_0020.pt", 20, seed=1)
    b = save_tiny(tmp_path, "iter_0040.pt", 40, seed=2)
    cfg = tiny_cfg()
    rows = play_pair(
        cfg,
        fake_evaluator(),
        ("iter_0020", str(a)),
        ("iter_0040", str(b)),
        n_games=2,
    )
    assert len(rows) == 2
    whites = {r["white"] for r in rows}
    assert whites == {"iter_0020", "iter_0040"}  # colors alternate
    for r in rows:
        assert r["result"] in ("win", "loss", "draw")
        assert r["sims"] == cfg.num_simulations
    # max_game_plies=2 forces draws in this stub world
    assert all(r["result"] == "draw" for r in rows)


def test_fit_arena_elo_anchors_oldest_and_rates_winner_higher():
    # synthetic: "new" beats "old" 15-5 as an even color split
    rows = []
    for i in range(20):
        white, black = ("new", "old") if i % 2 == 0 else ("old", "new")
        new_is_white = white == "new"
        new_wins = i < 15
        result = "win" if (new_wins == new_is_white) else "loss"
        rows.append({"white": white, "black": black, "result": result, "sims": 2})
    ratings = fit_arena_elo(rows, order=["old", "new"])
    assert ratings["old"] == 0.0
    assert 100.0 < ratings["new"] < 400.0  # ~+190 for 75%
```

- [ ] **Step 4.2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_arena.py -v`
Expected: `ModuleNotFoundError: No module named 'muzero.arena'`.

- [ ] **Step 4.3: Implement** — create `muzero/arena.py`:

```python
"""Checkpoint arena: adjacent-pair matches + relative Elo.

Spec: docs/superpowers/specs/2026-07-07-checkpoint-elo-arena-design.md.
Run offline: `uv run python -m muzero.arena` (needs PIKAFISH_BIN, like the
gate). Ratings are relative (oldest checkpoint anchored at 0) and only
comparable within a single --sims setting."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, replace

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.encoding import absolute_visits, index_to_move
from muzero.env import XiangqiEnv
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet
from muzero.selfplay import canonical_root
from scripts.benchmark.elo_estimator import fit_ratings
from src.xiangqi_board import engine_uci_to_algebraic

_ITER_RE = re.compile(r"iter_?(\d+)")


@dataclass
class Checkpoint:
    label: str
    path: str
    iteration: int


def discover_checkpoints(archive_dir: str, extras: list[str] | None = None) -> list[Checkpoint]:
    """archive/iter_*.pt plus extras, sorted by training iteration.

    Iteration comes from the checkpoint's own "iteration" key when present,
    else from an iter<NNNN> pattern in the filename."""
    paths = []
    if os.path.isdir(archive_dir):
        paths = [
            os.path.join(archive_dir, f)
            for f in sorted(os.listdir(archive_dir))
            if f.startswith("iter_") and f.endswith(".pt")
        ]
    paths += list(extras or [])
    out = []
    for p in paths:
        label = os.path.splitext(os.path.basename(p))[0]
        ckpt = torch.load(p, map_location="cpu")
        iteration = ckpt.get("iteration")
        if iteration is None:
            m = _ITER_RE.search(label)
            if not m:
                raise ValueError(f"cannot determine iteration for {p}")
            iteration = int(m.group(1))
        out.append(Checkpoint(label=label, path=p, iteration=int(iteration)))
    out.sort(key=lambda c: c.iteration)
    return out


def games_needed(rows: list[dict], a: str, b: str, *, sims: int, games_per_pair: int) -> int:
    played = sum(
        1
        for r in rows
        if {r["white"], r["black"]} == {a, b} and r.get("sims") == sims
    )
    return max(0, games_per_pair - played)


def _load_player(cfg: MuZeroConfig, path: str):
    net = MuZeroNet(cfg).to(cfg.device)
    ckpt = torch.load(path, map_location=cfg.device)
    net.load_state_dict(ckpt["ally"])
    net.eval()
    return NetRunner(net, cfg.device), MCTS(cfg)


def _play_game(cfg: MuZeroConfig, evaluator, players: dict, opening_uci: str) -> str:
    """players: side ("w"/"b") -> (runner, mcts). Returns win/loss/draw
    from WHITE's (red's) perspective — the elo_estimator convention."""
    env = XiangqiEnv(cfg, evaluator)
    env.reset()
    opening = engine_uci_to_algebraic(opening_uci)
    if opening is not None and opening in env.legal_moves():
        env.step(opening)
    while env.result is None:
        runner, mcts = players[env.side_to_move]
        obs, legal = canonical_root(env)
        if len(legal) == 0:
            break
        ((visits, _, _),) = mcts.run(runner, [(obs, legal)], add_noise=False)
        visits = absolute_visits(visits, env.side_to_move)
        env.step(index_to_move(max(visits, key=visits.get)))
    if env.result == "red_win":
        return "win"
    if env.result == "black_win":
        return "loss"
    return "draw"


def play_pair(cfg: MuZeroConfig, evaluator, a: tuple, b: tuple, n_games: int) -> list[dict]:
    """a/b: (label, path). Plays n_games alternating colors, cycling the
    opening book every two games. Returns jsonl-ready row dicts."""
    (label_a, path_a), (label_b, path_b) = a, b
    player_a = _load_player(cfg, path_a)
    player_b = _load_player(cfg, path_b)
    rows = []
    for g in range(n_games):
        opening = cfg.opening_book[(g // 2) % len(cfg.opening_book)]
        a_is_white = g % 2 == 0
        players = (
            {"w": player_a, "b": player_b}
            if a_is_white
            else {"w": player_b, "b": player_a}
        )
        result = _play_game(cfg, evaluator, players, opening)
        rows.append(
            {
                "white": label_a if a_is_white else label_b,
                "black": label_b if a_is_white else label_a,
                "result": result,
                "sims": cfg.num_simulations,
                "opening": opening,
            }
        )
    return rows


def fit_arena_elo(rows: list[dict], order: list[str]) -> dict:
    """Relative Elo with the oldest player (order[0]) anchored at 0."""
    games = [
        {"white": r["white"], "black": r["black"], "result": r["result"]}
        for r in rows
    ]
    ratings, _theta, _nll = fit_ratings(games, fixed_ratings={order[0]: 0.0})
    return {label: float(ratings.get(label, 0.0)) for label in order}


def main() -> None:
    from src.pikafish_eval import PikafishEvaluator

    ap = argparse.ArgumentParser(description="Checkpoint arena + relative Elo")
    ap.add_argument("--archive-dir", default="checkpoints/muzero_xiangqi/archive")
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional checkpoint .pt files (repeatable), e.g. latest.pt",
    )
    ap.add_argument("--games-per-pair", type=int, default=20)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="data/arena")
    args = ap.parse_args()

    cfg = replace(
        MuZeroConfig(), num_simulations=args.sims, device=args.device
    )
    checkpoints = discover_checkpoints(args.archive_dir, extras=args.extra)
    if len(checkpoints) < 2:
        raise SystemExit(
            f"Need >= 2 checkpoints, found {len(checkpoints)} "
            f"(archive: {args.archive_dir}; use --extra to add more)"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    games_path = os.path.join(args.out_dir, "games.jsonl")
    rows: list[dict] = []
    if os.path.exists(games_path):
        with open(games_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    evaluator = PikafishEvaluator(
        binary_path=cfg.pikafish_bin,
        depth=cfg.pikafish_depth,
        timeout_sec=cfg.pikafish_timeout_sec,
        movetime_ms=cfg.pikafish_movetime_ms,
        verbose=False,
    )
    for prev, curr in zip(checkpoints, checkpoints[1:]):
        need = games_needed(
            rows, prev.label, curr.label, sims=args.sims, games_per_pair=args.games_per_pair
        )
        if need == 0:
            print(f"[arena] {prev.label} vs {curr.label}: complete, skipping")
            continue
        print(f"[arena] {prev.label} vs {curr.label}: playing {need} games ...")
        new_rows = play_pair(
            cfg, evaluator, (prev.label, prev.path), (curr.label, curr.path), need
        )
        with open(games_path, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
        rows.extend(new_rows)

    order = [c.label for c in checkpoints]
    fit_rows = [r for r in rows if r.get("sims") == args.sims]
    skipped = len(rows) - len(fit_rows)
    if skipped:
        print(f"[arena] ignoring {skipped} rows from other --sims settings")
    ratings = fit_arena_elo(fit_rows, order)
    print(f"\n{'checkpoint':<28}{'iter':>6}{'Elo':>8}{'games':>7}")
    for c in checkpoints:
        n = sum(1 for r in fit_rows if c.label in (r["white"], r["black"]))
        print(f"{c.label:<28}{c.iteration:>6}{ratings[c.label]:>8.0f}{n:>7}")
    print(
        "\nNote: ~20 games/pair => roughly +-80 Elo per step; read the curve's"
        " shape across several checkpoints, not neighbor differences."
    )
    with open(os.path.join(args.out_dir, "ratings.json"), "w") as f:
        json.dump(
            {
                c.label: {"iteration": c.iteration, "elo": ratings[c.label]}
                for c in checkpoints
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
```

Note on the elo_estimator import: `from scripts.benchmark.elo_estimator import fit_ratings` is verified to work from the repo root (namespace package). If pytest's import mode complains, add the repo-root sys.path shim used elsewhere — but verify first; do not add it preemptively.

- [ ] **Step 4.4: Run tests**

Run: `uv run pytest muzero/tests/test_arena.py -v` → 4 passed. Full suite: `uv run pytest muzero/tests web/tests -q` → 92 passed, 6 skipped. Ruff check/format.

- [ ] **Step 4.5: Commit**

```bash
git add muzero/arena.py muzero/tests/test_arena.py
git commit -m "feat(muzero): checkpoint arena with relative Elo (python -m muzero.arena)"
```

---

### Task 5: Docs, merge, restart checklist

**Files:**
- Modify: `docs/ARCHITECTURE.md` (§3f), `docs/AGENT_TODO.md`
- Create: `docs/logs/2026-07-07-log-greedy-rung-and-elo-arena.md` (per template)

- [ ] **Step 5.1:** ARCHITECTURE §3f: gate ladder is now three rungs (`gate/*_random`, `gate/*_greedy`, `gate/*` = Pikafish); note `checkpoint_archive_every=20` snapshots to `checkpoints/muzero_xiangqi/archive/` and `python -m muzero.arena` (adjacent-pair matches, Bradley–Terry via `scripts/benchmark/elo_estimator`, output `data/arena/{games.jsonl,ratings.json}`). AGENT_TODO: update the Active training task — the next restart activates the greedy rung + archiving; add arena first-run to the checklist (`uv run python -m muzero.arena --extra checkpoints/muzero_xiangqi/iter80-prebufferfix.pt --extra checkpoints/muzero_xiangqi/latest.pt`); note baseline plan (2–3 greedy-rung gate readings before changing shaping/truncation knobs). Log file per `docs/logs/template.md`.

- [ ] **Step 5.2: Verify + merge**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
uv run pytest muzero/tests web/tests -q     # 92 passed, 6 skipped
git add docs && git commit -m "docs(muzero): greedy rung + arena log and architecture updates"
git checkout main && git merge --no-ff muzero-gate-arena -m "merge: greedy gate rung + checkpoint elo arena" && git push
```

- [ ] **Step 5.3: Box restart checklist (user runs):**

```bash
# on the 5090 box
git pull
uv run pytest muzero/tests -q                 # incl. engine-gated
# stop training (Ctrl-C), then resume — picks up greedy rung + archiving:
uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt
# after 2+ archives exist (~40 iters), first arena run:
PIKAFISH_BIN=... uv run python -m muzero.arena \
  --extra checkpoints/muzero_xiangqi/iter80-prebufferfix.pt \
  --extra checkpoints/muzero_xiangqi/latest.pt
```

---

## Self-Review (completed)

- **Spec coverage:** greedy spec §2a → Task 1 (incl. king-capture guard, both-colors test), §2b → Task 2 (separate rng, key names), §3 tests → Tasks 1-2. Arena spec §2a archiving → Task 3, §2b discovery/pairing/games/jsonl/fit/CLI → Task 4 (result convention pinned to elo_estimator's white-perspective win/loss/draw; idempotent re-runs via `games_needed`; `--sims` filter with skipped-row notice; oldest anchored at 0), §2c caveats → printed in arena output, §3 tests → Tasks 3-4. Rollout → Task 5.
- **Placeholder scan:** Task 1 deliberately shows a discarded test sketch followed by the final version — the STOP paragraph explains why (engine-UCI destination crafting is error-prone; stub the env instead). No TBDs.
- **Type consistency:** `greedy_capture_move(env, rng)` used identically in Tasks 1-2; `Checkpoint(label, path, iteration)` / `games_needed(rows, a, b, *, sims, games_per_pair)` / `play_pair(cfg, evaluator, (label, path), (label, path), n_games)` / `fit_arena_elo(rows, order)` consistent between Task 4 code and tests; archive filename `iter_%04d.pt` matches `discover_checkpoints`'s `iter_` prefix filter and `test_maybe_archive_checkpoint`.
- **Known risk flagged for implementers:** `test_play_pair_writes_valid_rows` and the gate-rung test depend on the FakeEvaluator two-ply draw pattern; if `env.step` rejects the stub moves, STOP and report rather than patching assertions.
