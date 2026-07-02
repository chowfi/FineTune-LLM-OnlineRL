# MuZero for Xiangqi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EfficientZero-style MuZero agent for Xiangqi specified in `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`, trainable on one RTX 5090.

**Architecture:** New `muzero/` package. Pikafish (`src/pikafish_eval.py`) is the only legality/eval source; `gym_xiangqi` applies moves. Board→tensor encoding feeds representation/dynamics/prediction nets searched by batched pUCT MCTS; self-play (ally vs frozen enemy) fills a 5000-game PER buffer; training unrolls K=8 with policy/value/reward/moves-left/material/SimSiam losses.

**Tech Stack:** PyTorch 2.10 (bf16, single CUDA device), numpy, gym==0.26.2 + gym-xiangqi, Pikafish via existing `PikafishEvaluator`, wandb, pytest, uv.

**Conventions used throughout (read first):**
- *Internal algebraic* moves: `a0a1` strings where row 0 = top (Black), row 9 = bottom (Red) — matches `src/xiangqi_board.py`. Engine (Pikafish) UCI has rank 0 at the bottom; convert with `engine_uci_to_algebraic` / `algebraic_to_engine_move`. Everything inside `muzero/` uses internal coords; conversions happen only at the engine boundary (`env.py`, `warmstart.py`, opening book).
- Board state: 10×9 numpy int array, positive ids 1–16 = Red, negative = Black (gym_xiangqi convention).
- Perspective rule: rewards and values are always from the perspective of the player who is about to move at that state ("mover perspective"). `PikafishEvaluator.evaluate_cp(fen)` returns cp from the side-to-move perspective; Red-perspective cp = `cp if stm == "w" else -cp`.
- Engine-dependent tests are skipped unless env var `PIKAFISH_BIN` points to a Pikafish binary.
- After each task: `ruff check . --fix && ruff format .` before committing (CLAUDE.md rule). Never edit `unsloth_compiled_cache/`.

**Run all tests with:** `uv run pytest muzero/tests -v`

---

### Task 1: Package scaffold + config

**Files:**
- Create: `muzero/__init__.py` (empty), `muzero/tests/__init__.py` (empty)
- Create: `muzero/config.py`
- Create: `muzero/tests/helpers.py`
- Test: `muzero/tests/test_config.py`

- [ ] **Step 1: Ensure pytest is available**

Run: `uv run python -c "import pytest; print(pytest.__version__)"`
If it fails: `uv add --dev pytest`

- [ ] **Step 2: Write the failing test**

`muzero/tests/test_config.py`:
```python
from muzero.config import MuZeroConfig


def test_defaults_match_spec():
    cfg = MuZeroConfig()
    assert cfg.num_simulations == 800
    assert cfg.unroll_steps == 8
    assert cfg.num_workers == 3
    assert cfg.games_per_worker == 28
    assert cfg.buffer_games == 5000
    assert cfg.games_per_train_loop == 512
    assert len(cfg.opening_book) == 10
    assert cfg.input_planes == 14 * cfg.history_length + 3 == 115
    assert len(cfg.loss_weights) == 6
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'muzero'`

- [ ] **Step 4: Implement**

`muzero/__init__.py` and `muzero/tests/__init__.py`: empty files.

`muzero/tests/helpers.py`:
```python
"""Shared test helpers: engine gating + fake evaluator."""

import os

import pytest

PIKAFISH_BIN = os.environ.get("PIKAFISH_BIN", "")

requires_engine = pytest.mark.skipif(
    not (PIKAFISH_BIN and os.path.exists(PIKAFISH_BIN)),
    reason="PIKAFISH_BIN not set or binary missing",
)


def make_evaluator():
    from src.pikafish_eval import PikafishEvaluator

    return PikafishEvaluator(
        binary_path=PIKAFISH_BIN,
        depth=8,
        timeout_sec=15.0,
        movetime_ms=100,
        verbose=False,
    )


class FakeEvaluator:
    """Scripted stand-in for PikafishEvaluator (legality + cp)."""

    enabled = True

    def __init__(self, cp_fn=lambda fen: 0.0, legal_fn=lambda fen: ["a0a1"]):
        self.cp_fn = cp_fn
        self.legal_fn = legal_fn

    def evaluate_cp(self, fen, moves=None):
        return self.cp_fn(fen)

    def list_legal_moves(self, fen):
        return self.legal_fn(fen)
```

`muzero/config.py`:
```python
"""All MuZero-Xiangqi hyperparameters. The only place numbers live."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MuZeroConfig:
    # Engine
    pikafish_bin: str = os.environ.get("PIKAFISH_BIN", "pikafish")
    pikafish_depth: int = 8
    pikafish_movetime_ms: int = 100
    pikafish_timeout_sec: float = 15.0

    # Encoding
    history_length: int = 8
    input_planes: int = 115  # 14 * history_length + 3
    action_space: int = 8100  # 90 from-squares x 90 to-squares

    # Network
    channels: int = 192
    repr_blocks: int = 12
    dyn_blocks: int = 8
    value_bins: int = 601
    value_max: float = 3.0  # h-transformed units; n-step returns live in ~[-2,2]
    reward_bins: int = 21
    reward_max: float = 2.0
    moves_left_max: int = 200

    # MCTS
    num_simulations: int = 800
    interior_topk: int = 64  # non-root nodes expand top-k prior actions
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    temperature_moves: int = 30  # sample proportional to visits for first N plies

    # Self-play
    num_workers: int = 3
    games_per_worker: int = 28
    promote_after_consecutive_wins: int = 3
    max_game_plies: int = 300
    # Red first moves in ENGINE UCI (rank 0 = bottom): central/edge cannon,
    # horses, elephants, pawn advances.
    opening_book: tuple = (
        "h2e2", "b2e2", "h0g2", "b0c2", "c0e2",
        "g0e2", "c3c4", "g3g4", "a3a4", "i3i4",
    )

    # Env rewards / adjudication
    shaping_weight: float = 0.3
    shaping_cp_scale: float = 200.0
    repetition_penalty: float = -0.3
    repetition_cp_ok: float = -100.0  # penalize repeater whose cp >= this
    repetition_swing_cp: float = 50.0  # "no threat" = cp swing below this
    truncation_cp: float = -800.0
    truncation_consecutive: int = 6

    # Replay / training
    buffer_games: int = 5000
    games_per_train_loop: int = 512
    batch_size: int = 512
    unroll_steps: int = 8
    td_steps: int = 10
    discount: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    # (policy, value, reward, moves_left, material, consistency)
    loss_weights: tuple = (1.0, 0.25, 1.0, 0.2, 0.1, 2.0)
    per_alpha: float = 0.6
    truncated_tail_weight: float = 0.3

    # Warm start
    warmstart_plies: int = 2000
    warmstart_movetime_ms: int = 50
    warmstart_multipv: int = 4
    warmstart_train_batches: int = 200

    # Fixed-opponent gate
    gate_every_loops: int = 10
    gate_games: int = 20
    gate_movetime_ms: int = 10

    # Misc
    device: str = "cuda"
    seed: int = 0
    checkpoint_dir: str = "checkpoints/muzero_xiangqi"
    wandb_project: str = "muzero-xiangqi"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero docs/superpowers/plans/2026-07-02-muzero-xiangqi.md
git commit -m "feat(muzero): package scaffold + config dataclass"
```

---

### Task 2: Move ↔ action-index encoding

**Files:**
- Create: `muzero/encoding.py` (part 1)
- Test: `muzero/tests/test_encoding.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_encoding.py`:
```python
import numpy as np

from muzero.encoding import index_to_move, move_to_index


def test_move_index_round_trip():
    for move in ["a0a1", "h9g7", "e6e5", "i9i0"]:
        idx = move_to_index(move)
        assert 0 <= idx < 8100
        assert index_to_move(idx) == move


def test_all_indices_decode_and_reencode():
    for idx in range(0, 8100, 173):
        assert move_to_index(index_to_move(idx)) == idx


def test_index_formula():
    # a0 -> square 0, a1 -> square 9: index = 0 * 90 + 9
    assert move_to_index("a0a1") == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_encoding.py -v`
Expected: FAIL with `No module named 'muzero.encoding'`

- [ ] **Step 3: Implement**

`muzero/encoding.py`:
```python
"""Board -> tensor planes and internal-algebraic move <-> flat action index."""

from __future__ import annotations

import numpy as np

from src.xiangqi_board import algebraic_to_board_coords, board_coords_to_algebraic

# gym_xiangqi piece ids (abs value) -> type index
# 0 king, 1 advisor, 2 elephant, 3 horse, 4 rook, 5 cannon, 6 pawn
PIECE_TYPE = {
    1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3,
    8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 6, 15: 6, 16: 6,
}
PIECE_VALUE = {0: 0.0, 1: 2.0, 2: 2.0, 3: 4.0, 4: 9.0, 5: 4.5, 6: 1.0}


def move_to_index(move: str) -> int:
    coords = algebraic_to_board_coords(move)
    if coords is None:
        raise ValueError(f"bad move: {move!r}")
    (fr, fc), (tr, tc) = coords
    return (fr * 9 + fc) * 90 + (tr * 9 + tc)


def index_to_move(idx: int) -> str:
    frm, to = divmod(int(idx), 90)
    return board_coords_to_algebraic(frm // 9, frm % 9, to // 9, to % 9)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_encoding.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): move <-> flat 8100 action index"
```

---

### Task 3: Observation planes + material balance

**Files:**
- Modify: `muzero/encoding.py` (append)
- Test: `muzero/tests/test_encoding.py` (append)

- [ ] **Step 1: Write the failing tests (append to `muzero/tests/test_encoding.py`)**

```python
from muzero.encoding import board_planes, encode_observation, material_balance


def _start_board():
    board = np.zeros((10, 9), dtype=np.int8)
    back = [8, 6, 4, 2, 1, 3, 5, 7, 9]  # r n b a k a b n r piece ids
    for c, pid in enumerate(back):
        board[0, c] = -pid  # black top
        board[9, c] = pid  # red bottom
    board[2, 1], board[2, 7] = -10, -11  # black cannons
    board[7, 1], board[7, 7] = 10, 11  # red cannons
    for c in range(0, 9, 2):
        board[3, c] = -(12 + c // 2)  # black pawns
        board[6, c] = 12 + c // 2  # red pawns
    return board


def test_board_planes_start_position():
    planes = board_planes(_start_board())
    assert planes.shape == (14, 10, 9)
    assert planes.sum() == 32  # 16 pieces per side
    assert planes[0, 9, 4] == 1.0  # red king plane
    assert planes[7, 0, 4] == 1.0  # black king plane


def test_encode_observation_shape_and_padding():
    board = _start_board()
    obs = encode_observation([board], "w", 1, 0, history_length=8)
    assert obs.shape == (115, 10, 9)
    assert obs[:14].sum() == 0  # padded oldest history slot is empty
    assert obs[98:112].sum() == 32  # newest slot holds the board
    assert obs[112].max() == 1.0  # side-to-move plane (red)


def test_material_balance():
    board = _start_board()
    assert material_balance(board) == 0.0
    board[0, 0] = 0  # remove a black rook
    assert material_balance(board) == 9.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_encoding.py -v`
Expected: new tests FAIL with `ImportError` (board_planes not defined)

- [ ] **Step 3: Implement (append to `muzero/encoding.py`)**

```python
def board_planes(board: np.ndarray) -> np.ndarray:
    """One position -> 14 binary planes (7 red types, then 7 black)."""
    planes = np.zeros((14, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            v = int(board[r, c])
            if v == 0:
                continue
            t = PIECE_TYPE[abs(v)]
            planes[t if v > 0 else 7 + t, r, c] = 1.0
    return planes


def encode_observation(
    boards: list,
    side_to_move: str,
    repetition_count: int,
    no_progress: int,
    history_length: int = 8,
) -> np.ndarray:
    """Stack of the last `history_length` boards (oldest first, zero-padded)
    plus side-to-move / repetition / no-progress broadcast planes."""
    hist = list(boards)[-history_length:]
    stacks = [np.zeros((14, 10, 9), dtype=np.float32)] * (history_length - len(hist))
    stacks = stacks + [board_planes(b) for b in hist]
    stm = np.full((1, 10, 9), 1.0 if side_to_move == "w" else 0.0, dtype=np.float32)
    rep = np.full((1, 10, 9), min(int(repetition_count), 3) / 3.0, dtype=np.float32)
    nop = np.full((1, 10, 9), min(int(no_progress), 100) / 100.0, dtype=np.float32)
    return np.concatenate(stacks + [stm, rep, nop], axis=0)


def material_balance(board: np.ndarray) -> float:
    """Red minus Black piece value (kings worth 0)."""
    total = 0.0
    for v in board.flatten():
        v = int(v)
        if v == 0:
            continue
        val = PIECE_VALUE[PIECE_TYPE[abs(v)]]
        total += val if v > 0 else -val
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_encoding.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): observation planes + material balance"
```

---

### Task 4: Scalar ↔ categorical support transforms

**Files:**
- Create: `muzero/transforms.py`
- Test: `muzero/tests/test_transforms.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_transforms.py`:
```python
import torch

from muzero.transforms import h_inverse, h_transform, scalar_to_support, support_to_scalar


def test_h_transform_round_trip():
    x = torch.tensor([-250.0, -1.0, 0.0, 0.5, 1.0, 250.0])
    assert torch.allclose(h_inverse(h_transform(x)), x, atol=1e-3)


def test_support_round_trip():
    x = torch.tensor([[-1.7, 0.0, 0.31, 1.9]])
    support = scalar_to_support(x, -2.0, 2.0, 21)
    assert support.shape == (1, 4, 21)
    assert torch.allclose(support.sum(-1), torch.ones(1, 4), atol=1e-6)
    # support_to_scalar expects logits; log of the two-hot distribution works
    back = support_to_scalar(torch.log(support + 1e-12), -2.0, 2.0, 21)
    assert torch.allclose(back, x, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_transforms.py -v`
Expected: FAIL with `No module named 'muzero.transforms'`

- [ ] **Step 3: Implement**

`muzero/transforms.py`:
```python
"""MuZero invertible value scaling and scalar <-> categorical support."""

from __future__ import annotations

import torch

EPS = 0.001


def h_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + EPS * x


def h_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (
        ((torch.sqrt(1.0 + 4.0 * EPS * (torch.abs(x) + 1.0 + EPS)) - 1.0) / (2.0 * EPS)) ** 2
        - 1.0
    )


def scalar_to_support(x: torch.Tensor, vmin: float, vmax: float, bins: int) -> torch.Tensor:
    """Two-hot encoding of scalars onto a linear support. Output: x.shape + (bins,)."""
    x = x.clamp(vmin, vmax)
    step = (vmax - vmin) / (bins - 1)
    pos = (x - vmin) / step
    low = pos.floor().long().clamp(0, bins - 1)
    high = (low + 1).clamp(0, bins - 1)
    frac = pos - low.float()
    out = torch.zeros(*x.shape, bins, dtype=torch.float32, device=x.device)
    out.scatter_(-1, low.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    out.scatter_add_(-1, high.unsqueeze(-1), frac.unsqueeze(-1))
    return out


def support_to_scalar(logits: torch.Tensor, vmin: float, vmax: float, bins: int) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    atoms = torch.linspace(vmin, vmax, bins, device=logits.device)
    return (probs * atoms).sum(dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_transforms.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): scalar/categorical support transforms"
```

---

### Task 5: Networks (representation / dynamics / prediction + SimSiam)

**Files:**
- Create: `muzero/network.py`
- Test: `muzero/tests/test_network.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_network.py`:
```python
import torch

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet, action_planes


def tiny_config():
    return MuZeroConfig(channels=16, repr_blocks=1, dyn_blocks=1, device="cpu")


def test_action_planes():
    planes = action_planes(torch.tensor([9]), "cpu")  # a0a1: from sq 0, to sq 9
    assert planes.shape == (1, 2, 10, 9)
    assert planes[0, 0, 0, 0] == 1.0  # from-square a0 -> row 0, col 0
    assert planes[0, 1, 1, 0] == 1.0  # to-square a1 -> row 1, col 0


def test_inference_shapes():
    cfg = tiny_config()
    net = MuZeroNet(cfg)
    obs = torch.randn(4, 115, 10, 9)
    out = net.initial_inference(obs)
    assert out["hidden"].shape == (4, 16, 10, 9)
    assert out["policy_logits"].shape == (4, 8100)
    assert out["value_logits"].shape == (4, cfg.value_bins)
    assert out["value"].shape == (4,)
    assert out["moves_left_logits"].shape == (4, cfg.moves_left_max + 1)
    assert out["material"].shape == (4,)

    out2 = net.recurrent_inference(out["hidden"], torch.tensor([9, 9, 9, 9]))
    assert out2["hidden"].shape == (4, 16, 10, 9)
    assert out2["reward_logits"].shape == (4, cfg.reward_bins)
    assert out2["reward"].shape == (4,)


def test_projection_shapes():
    cfg = tiny_config()
    net = MuZeroNet(cfg)
    hidden = torch.randn(4, 16, 10, 9)
    assert net.project(hidden, with_predictor=False).shape == (4, 1024)
    assert net.project(hidden, with_predictor=True).shape == (4, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_network.py -v`
Expected: FAIL with `No module named 'muzero.network'`

- [ ] **Step 3: Implement**

`muzero/network.py`:
```python
"""Representation / dynamics / prediction networks + SimSiam projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.transforms import h_inverse, support_to_scalar


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


def action_planes(actions: torch.Tensor, device) -> torch.Tensor:
    """Action indices (B,) -> (B, 2, 10, 9): from-square and to-square one-hots."""
    b = actions.shape[0]
    planes = torch.zeros(b, 2, 90, device=device)
    frm = (actions // 90).long()
    to = (actions % 90).long()
    idx = torch.arange(b, device=device)
    planes[idx, 0, frm] = 1.0
    planes[idx, 1, to] = 1.0
    return planes.view(b, 2, 10, 9)


def normalize_hidden(h: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max scaling to [0, 1] (MuZero appendix G)."""
    flat = h.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-8)
    return flat.view_as(h)


def _tower(in_ch: int, ch: int, blocks: int) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()]
    layers += [ResBlock(ch) for _ in range(blocks)]
    return nn.Sequential(*layers)


def _head(ch: int, reduced: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ch, reduced, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(reduced * 90, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out),
    )


class MuZeroNet(nn.Module):
    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        self.representation = _tower(cfg.input_planes, ch, cfg.repr_blocks)
        self.dynamics = _tower(ch + 2, ch, cfg.dyn_blocks)
        self.reward_head = _head(ch, 2, 128, cfg.reward_bins)
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 90, cfg.action_space)
        )
        self.value_head = _head(ch, 2, 256, cfg.value_bins)
        self.moves_left_head = _head(ch, 2, 128, cfg.moves_left_max + 1)
        self.material_head = _head(ch, 2, 128, 1)
        self.projector = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * 90, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.predictor = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1024))

    def _predict(self, hidden: torch.Tensor) -> dict:
        value_logits = self.value_head(hidden)
        value = h_inverse(
            support_to_scalar(value_logits, -self.cfg.value_max, self.cfg.value_max, self.cfg.value_bins)
        )
        return {
            "hidden": hidden,
            "policy_logits": self.policy_head(hidden),
            "value_logits": value_logits,
            "value": value,
            "moves_left_logits": self.moves_left_head(hidden),
            "material": self.material_head(hidden).squeeze(-1),
        }

    def initial_inference(self, obs: torch.Tensor) -> dict:
        return self._predict(normalize_hidden(self.representation(obs)))

    def recurrent_inference(self, hidden: torch.Tensor, actions: torch.Tensor) -> dict:
        x = torch.cat([hidden, action_planes(actions, hidden.device)], dim=1)
        next_hidden = normalize_hidden(self.dynamics(x))
        reward_logits = self.reward_head(next_hidden)
        out = self._predict(next_hidden)
        out["reward_logits"] = reward_logits
        out["reward"] = support_to_scalar(
            reward_logits, -self.cfg.reward_max, self.cfg.reward_max, self.cfg.reward_bins
        )
        return out

    def project(self, hidden: torch.Tensor, with_predictor: bool) -> torch.Tensor:
        p = self.projector(hidden)
        return self.predictor(p) if with_predictor else p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_network.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): repr/dynamics/prediction nets with aux heads + SimSiam"
```

---

### Task 6: Environment wrapper — core (Pikafish legality, stepping, shaping)

**Files:**
- Create: `muzero/env.py`
- Test: `muzero/tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

`muzero/tests/test_env.py`:
```python
import numpy as np

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from muzero.tests.helpers import FakeEvaluator, make_evaluator, requires_engine


@requires_engine
def test_reset_and_legal_moves():
    env = XiangqiEnv(MuZeroConfig(), make_evaluator())
    board = env.reset()
    assert board.shape == (10, 9)
    assert env.side_to_move == "w"
    legal = env.legal_moves()
    assert len(legal) == 44  # known perft(1) of the start position
    assert all(len(m) == 4 for m in legal)


@requires_engine
def test_step_pawn_push():
    env = XiangqiEnv(MuZeroConfig(), make_evaluator())
    env.reset()
    board, reward, done, info = env.step("e6e5")  # red central pawn (engine e3e4)
    assert not done
    assert env.side_to_move == "b"
    assert env.plies == 1
    assert board[6, 4] == 0 and board[5, 4] > 0
    assert isinstance(reward, float)


def test_observation_shape_with_fake_engine():
    env = XiangqiEnv(MuZeroConfig(), FakeEvaluator())
    env.reset()
    obs = env.observation()
    assert obs.shape == (115, 10, 9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_env.py -v`
Expected: FAIL with `No module named 'muzero.env'`

- [ ] **Step 3: Implement**

`muzero/env.py`:
```python
"""Xiangqi env wrapper: gym_xiangqi applies moves; Pikafish is the sole
legality and evaluation source. Enforces repetition-draw and hopeless-game
truncation per the design spec."""

from __future__ import annotations

import gym
import numpy as np
from gym_xiangqi.utils import move_to_action_space

from muzero.config import MuZeroConfig
from src.xiangqi_board import (
    algebraic_to_board_coords,
    board_to_uci_fen,
    engine_uci_to_algebraic,
)

from muzero.encoding import encode_observation


class XiangqiEnv:
    def __init__(self, config: MuZeroConfig, evaluator):
        self.config = config
        self.evaluator = evaluator
        self._gym = None

    # -- lifecycle -----------------------------------------------------------

    def reset(self, ally_side: str = "w") -> np.ndarray:
        if self._gym is not None:
            self._gym.close()
        self._gym = gym.make("gym_xiangqi:xiangqi-v0")
        res = self._gym.reset()
        obs = res[0] if isinstance(res, tuple) else res
        self.board = self._extract_board(obs)
        self.side_to_move = "w"
        self.ally_side = ally_side
        self.plies = 0
        self.no_progress = 0
        self.result = None
        self.truncated = False
        self._rep_counts: dict = {}
        self._rep_cps: dict = {}
        self._sat_streak = 0
        # Per-state histories (index t = state before ply t), used to rebuild
        # observations in the replay buffer.
        self.boards = [self.board.copy()]
        self.to_play_history = ["w"]
        self.rep_history = [self._bump_repetition()]
        self.no_progress_history = [0]
        return self.board

    def _extract_board(self, obs) -> np.ndarray:
        state = getattr(self._gym.unwrapped, "state", None)
        if state is None:
            state = obs
        return np.array(state, dtype=np.int8).reshape(10, 9)

    # -- queries -------------------------------------------------------------

    def fen(self) -> str:
        return board_to_uci_fen(self.board, self.side_to_move)

    def legal_moves(self) -> list:
        engine_moves = self.evaluator.list_legal_moves(self.fen()) or []
        moves = []
        for u in engine_moves:
            m = engine_uci_to_algebraic(u)
            if m is not None:
                moves.append(m)
        return moves

    def red_cp(self):
        cp = self.evaluator.evaluate_cp(self.fen())
        if cp is None:
            return None
        return float(cp) if self.side_to_move == "w" else -float(cp)

    def observation(self) -> np.ndarray:
        return encode_observation(
            self.boards,
            self.side_to_move,
            self.rep_history[-1],
            self.no_progress,
            self.config.history_length,
        )

    # -- stepping ------------------------------------------------------------

    def step(self, move: str):
        """Apply an internal-algebraic move for the side to move.

        Returns (board, reward, done, info); reward is mover-perspective and
        includes shaping, terminal +/-1, and any repetition penalty."""
        assert self.result is None, "game already over"
        mover = self.side_to_move
        (fr, fc), (tr, tc) = algebraic_to_board_coords(move)
        piece_id = int(self.board[fr, fc])
        captured = int(self.board[tr, tc]) != 0
        cp_before_red = self.red_cp()

        action = int(move_to_action_space(piece_id, (fr, fc), (tr, tc)))
        res = self._gym.step(action)
        if len(res) == 5:
            obs, _, term, trunc, _ = res
            gym_done = bool(term or trunc)
        else:
            obs, _, gym_done, _ = res
        self.board = self._extract_board(obs)
        self.side_to_move = "b" if mover == "w" else "w"
        self.plies += 1
        self.no_progress = 0 if captured else self.no_progress + 1
        rep = self._bump_repetition()
        self.boards.append(self.board.copy())
        self.to_play_history.append(self.side_to_move)
        self.rep_history.append(rep)
        self.no_progress_history.append(self.no_progress)

        cp_after_red = self.red_cp()
        reward = self._shaping_reward(mover, cp_before_red, cp_after_red)
        info = {"red_cp": cp_after_red, "truncated": False, "repetition_penalized": None}

        if gym_done or not self.legal_moves():
            # King captured, or opponent has no legal move (mate/stalemate):
            # mover wins in Xiangqi.
            self.result = "red_win" if mover == "w" else "black_win"
            reward += 1.0
        elif rep >= 3 and self._no_threat():
            self.result = "draw_repetition"
            mover_cp = self._mover_cp(mover, cp_after_red)
            if mover_cp is not None and mover_cp >= self.config.repetition_cp_ok:
                reward += self.config.repetition_penalty
                info["repetition_penalized"] = mover
        elif self.plies >= self.config.max_game_plies:
            self.result = "draw_max_plies"
        elif self._check_truncation(mover, cp_after_red):
            self.result = "black_win" if self.ally_side == "w" else "red_win"
            self.truncated = True
            info["truncated"] = True
            reward += -1.0  # mover here is always the saturated ally

        info["result"] = self.result
        return self.board, float(reward), self.result is not None, info

    # -- internals -----------------------------------------------------------

    def _mover_cp(self, mover: str, red_cp):
        if red_cp is None:
            return None
        return red_cp if mover == "w" else -red_cp

    def _shaping_reward(self, mover: str, cp_before_red, cp_after_red) -> float:
        if cp_before_red is None or cp_after_red is None:
            return 0.0
        delta_red = cp_after_red - cp_before_red
        delta = delta_red if mover == "w" else -delta_red
        return self.config.shaping_weight * float(np.tanh(delta / self.config.shaping_cp_scale))

    def _position_key(self) -> str:
        return self.fen().rsplit(" ", 2)[0]  # board + stm (drop move counters)

    def _bump_repetition(self) -> int:
        key = self._position_key()
        self._rep_counts[key] = self._rep_counts.get(key, 0) + 1
        cp = None
        if self._rep_counts[key] >= 2:  # only pay for cp once repeats start
            cp = self.red_cp()
        if cp is not None:
            self._rep_cps.setdefault(key, []).append(cp)
        return self._rep_counts[key]

    def _no_threat(self) -> bool:
        cps = self._rep_cps.get(self._position_key(), [])
        if len(cps) < 2:
            return True  # no eval signal -> treat as threat-free shuffle
        return (max(cps) - min(cps)) < self.config.repetition_swing_cp

    def _check_truncation(self, mover: str, cp_after_red) -> bool:
        if mover != self.ally_side:
            return False
        mover_cp = self._mover_cp(mover, cp_after_red)
        if mover_cp is None:
            return False
        if mover_cp <= self.config.truncation_cp:
            self._sat_streak += 1
        else:
            self._sat_streak = 0
        return self._sat_streak >= self.config.truncation_consecutive
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_env.py -v`
Expected: PASS (engine tests skipped if `PIKAFISH_BIN` unset; the fake-engine test must pass everywhere)

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): Pikafish-legal env wrapper with cp shaping"
```

---

### Task 7: Environment — repetition-draw and truncation adjudication

**Files:**
- Modify: none (behavior exists from Task 6; this task locks it in with tests)
- Test: `muzero/tests/test_env_adjudication.py`

- [ ] **Step 1: Write the failing tests**

`muzero/tests/test_env_adjudication.py`:
```python
from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from muzero.tests.helpers import FakeEvaluator

# Horse shuffle in internal coords (red horse h9<->g7, black horse h0<->g2).
SHUFFLE = ["h9g7", "h0g2", "g7h9", "g2h0"]


def test_threefold_repetition_is_draw_with_penalty():
    cfg = MuZeroConfig()
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 0.0))
    env.reset()
    done = False
    outputs = []
    for i in range(8):  # start position recurs at plies 0, 4, 8
        assert not done
        _, reward, done, info = env.step(SHUFFLE[i % 4])
        outputs.append((reward, info))
    assert done
    assert env.result == "draw_repetition"
    reward, info = outputs[-1]
    assert info["repetition_penalized"] == "b"  # black completed the repetition
    assert reward <= cfg.repetition_penalty


def test_hopeless_ally_game_truncates_as_loss():
    import numpy as np
    import pytest

    cfg = replace(MuZeroConfig(), truncation_consecutive=3)
    # evaluate_cp is side-to-move perspective; +900 after every move means the
    # mover always left the opponent at +900, i.e. red is at -900 after red moves.
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 900.0))
    env.reset(ally_side="w")
    done = False
    plies = 0
    while not done:
        _, reward, done, info = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "black_win"
    assert env.truncated and info["truncated"]
    assert plies == 5  # red's 3rd saturated move
    # Terminal -1 plus shaping on the final move: red-perspective cp swings
    # +900 -> -900 (delta -1800) because the fake engine always reports +900
    # for the side to move.
    shaping = cfg.shaping_weight * float(np.tanh(-1800.0 / cfg.shaping_cp_scale))
    assert reward == pytest.approx(-1.0 + shaping, abs=1e-6)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest muzero/tests/test_env_adjudication.py -v`
Expected: PASS if Task 6 is correct. If either fails, fix `muzero/env.py` (not the tests) until they pass — likely spots: repetition key must include side-to-move; `_check_truncation` must only count the ally's own post-move evals.

- [ ] **Step 3: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "test(muzero): repetition-draw + truncation adjudication"
```

---

### Task 8: GameHistory + replay buffer with PER and unroll targets

**Files:**
- Create: `muzero/replay_buffer.py`
- Test: `muzero/tests/test_replay_buffer.py`

- [ ] **Step 1: Write the failing tests**

`muzero/tests/test_replay_buffer.py`:
```python
import numpy as np

from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.replay_buffer import GameHistory, ReplayBuffer


def make_game(length=20, result="red_win", truncated=False):
    g = GameHistory()
    rng = np.random.default_rng(0)
    for t in range(length + 1):
        g.boards.append(rng.integers(-16, 17, size=(10, 9)).astype(np.int8))
        g.to_play_history.append("w" if t % 2 == 0 else "b")
        g.rep_history.append(1)
        g.no_progress_history.append(t)
    for t in range(length):
        g.actions.append(int(rng.integers(0, 8100)))
        g.rewards.append(0.0)
        g.policy_indices.append(np.array([1, 2, 3], dtype=np.int64))
        g.policy_probs.append(np.array([0.5, 0.3, 0.2], dtype=np.float32))
        g.root_values.append(0.1)
    g.rewards[-1] = 1.0
    g.result = result
    g.truncated = truncated
    return g


def test_add_and_fifo_eviction():
    cfg = replace(MuZeroConfig(), buffer_games=3)
    buf = ReplayBuffer(cfg)
    for _ in range(5):
        buf.add(make_game())
    assert len(buf.games) == 3


def test_n_step_value_perspective_signs():
    cfg = replace(MuZeroConfig(), td_steps=10)
    g = make_game(length=4)  # rewards [0,0,0,1.0], no bootstrap past end
    buf = ReplayBuffer(cfg)
    # From state 3 the mover receives +1 immediately.
    assert buf.n_step_value(g, 3) == 1.0
    # From state 2 the opponent gets +1 next ply -> -1 for the state-2 mover.
    assert buf.n_step_value(g, 2) == -1.0


def test_sample_batch_shapes():
    cfg = replace(MuZeroConfig(), batch_size=8, unroll_steps=8)
    buf = ReplayBuffer(cfg)
    for _ in range(4):
        buf.add(make_game())
    batch = buf.sample_batch(8)
    assert batch["obs"].shape == (8, 115, 10, 9)
    assert batch["actions"].shape == (8, 8)
    assert batch["target_policy"].shape == (8, 9, 8100)
    assert batch["policy_mask"].shape == (8, 9)
    assert batch["target_value"].shape == (8, 9)
    assert batch["target_reward"].shape == (8, 8)
    assert batch["target_moves_left"].shape == (8, 9)
    assert batch["target_material"].shape == (8, 9)
    assert batch["consistency_obs"].shape == (8, 115, 10, 9)
    assert batch["consistency_k"].shape == (8,)
    np.testing.assert_allclose(
        np.asarray(batch["target_policy"]).sum(-1), 1.0, rtol=1e-4
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_replay_buffer.py -v`
Expected: FAIL with `No module named 'muzero.replay_buffer'`

- [ ] **Step 3: Implement**

`muzero/replay_buffer.py`:
```python
"""Whole-game FIFO replay buffer with proportional prioritized sampling.

Observations are reconstructed from stored int8 boards at sample time
(storing encoded 115-plane stacks for 5000 games would need ~30 GB)."""

from __future__ import annotations

from collections import deque

import numpy as np

from muzero.config import MuZeroConfig
from muzero.encoding import encode_observation, material_balance


class GameHistory:
    """One finished game. Index t: boards/to_play/rep/no_progress have L+1
    entries (state before ply t, plus terminal state); the rest have L."""

    def __init__(self):
        self.boards = []
        self.to_play_history = []
        self.rep_history = []
        self.no_progress_history = []
        self.actions = []
        self.rewards = []  # mover-perspective, shaping + terminal + penalties
        self.policy_indices = []  # sparse root visit distributions
        self.policy_probs = []
        self.root_values = []  # mover-perspective MCTS root values
        self.result = None
        self.truncated = False
        self.ally_side = "w"

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.games: deque = deque(maxlen=config.buffer_games)
        self.priorities: deque = deque(maxlen=config.buffer_games)
        self.total_games_added = 0
        self.rng = np.random.default_rng(config.seed)

    # -- adding ---------------------------------------------------------------

    def add(self, game: GameHistory) -> None:
        pri = np.array(
            [abs(game.root_values[t] - self.n_step_value(game, t)) + 1e-3 for t in range(len(game))],
            dtype=np.float32,
        ) ** self.config.per_alpha
        if game.truncated:
            tail = 2 * self.config.truncation_consecutive
            pri[-tail:] *= self.config.truncated_tail_weight
        self.games.append(game)
        self.priorities.append(pri)
        self.total_games_added += 1

    # -- targets ----------------------------------------------------------------

    def n_step_value(self, game: GameHistory, t: int) -> float:
        """Mover-perspective n-step return for state t. Rewards and bootstrap
        values alternate sign because each ply flips whose move it is."""
        cfg = self.config
        g = 0.0
        for j in range(cfg.td_steps):
            k = t + j
            if k >= len(game):
                return g
            g += (cfg.discount ** j) * ((-1) ** j) * game.rewards[k]
        k = t + cfg.td_steps
        if k < len(game.root_values):
            g += (cfg.discount ** cfg.td_steps) * ((-1) ** cfg.td_steps) * game.root_values[k]
        return g

    def _dense_policy(self, game: GameHistory, t: int) -> np.ndarray:
        dense = np.zeros(self.config.action_space, dtype=np.float32)
        dense[game.policy_indices[t]] = game.policy_probs[t]
        s = dense.sum()
        return dense / s if s > 0 else np.full_like(dense, 1.0 / dense.shape[0])

    def make_target(self, game: GameHistory, t: int) -> dict:
        cfg = self.config
        K, L = cfg.unroll_steps, len(game)
        uniform = np.full(cfg.action_space, 1.0 / cfg.action_space, dtype=np.float32)

        obs = encode_observation(
            game.boards[: t + 1], game.to_play_history[t],
            game.rep_history[t], game.no_progress_history[t], cfg.history_length,
        )
        actions = np.array(
            [game.actions[t + k] if t + k < L else int(self.rng.integers(0, cfg.action_space))
             for k in range(K)],
            dtype=np.int64,
        )
        policies, pmask, values, moves_left, material = [], [], [], [], []
        for k in range(K + 1):
            s = t + k
            if s < L:
                policies.append(self._dense_policy(game, s))
                pmask.append(1.0)
                values.append(self.n_step_value(game, s))
            else:
                policies.append(uniform)
                pmask.append(0.0)
                values.append(0.0)  # absorbing states train value to 0
            moves_left.append(min(max(L - s, 0), cfg.moves_left_max))
            material.append(material_balance(game.boards[min(s, L)]) / 10.0)
        rewards = np.array(
            [game.rewards[t + k] if t + k < L else 0.0 for k in range(K)], dtype=np.float32
        )
        # SimSiam target: one random real future observation within the unroll.
        max_kc = min(K, L - t)
        k_c = int(self.rng.integers(1, max_kc + 1)) if max_kc >= 1 else 0
        c_obs = (
            encode_observation(
                game.boards[: t + k_c + 1], game.to_play_history[t + k_c],
                game.rep_history[t + k_c], game.no_progress_history[t + k_c],
                cfg.history_length,
            )
            if k_c > 0
            else np.zeros((cfg.input_planes, 10, 9), dtype=np.float32)
        )
        return {
            "obs": obs, "actions": actions,
            "target_policy": np.stack(policies), "policy_mask": np.array(pmask, dtype=np.float32),
            "target_value": np.array(values, dtype=np.float32),
            "target_reward": rewards,
            "target_moves_left": np.array(moves_left, dtype=np.int64),
            "target_material": np.array(material, dtype=np.float32),
            "consistency_obs": c_obs, "consistency_k": k_c,
        }

    # -- sampling ---------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> dict:
        flat, owners = [], []
        for gi, pri in enumerate(self.priorities):
            flat.append(pri)
            owners.append(np.full(len(pri), gi, dtype=np.int64))
        flat = np.concatenate(flat)
        owners = np.concatenate(owners)
        offsets = np.concatenate([[0], np.cumsum([len(p) for p in self.priorities])])
        probs = flat / flat.sum()
        picks = self.rng.choice(flat.shape[0], size=batch_size, p=probs)
        samples = [
            self.make_target(self.games[owners[i]], int(i - offsets[owners[i]])) for i in picks
        ]
        return {k: np.stack([s[k] for s in samples]) for k in samples[0]}

    def mean_game_length(self) -> float:
        return float(np.mean([len(g) for g in self.games])) if self.games else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_replay_buffer.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): PER replay buffer with K=8 unroll targets"
```

---

### Task 9: Batched MCTS

**Files:**
- Create: `muzero/mcts.py`
- Test: `muzero/tests/test_mcts.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_mcts.py`:
```python
import numpy as np
import torch

from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet


def test_mcts_respects_mask_and_visit_budget():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1,
        num_simulations=16, interior_topk=8, device="cpu",
    )
    torch.manual_seed(0)
    runner = NetRunner(MuZeroNet(cfg), "cpu")
    rng = np.random.default_rng(0)
    legal_a = np.array([5, 100, 8099], dtype=np.int64)
    legal_b = np.array([0, 1, 2, 3], dtype=np.int64)
    roots = [
        (rng.standard_normal((115, 10, 9)).astype(np.float32), legal_a),
        (rng.standard_normal((115, 10, 9)).astype(np.float32), legal_b),
    ]
    results = MCTS(cfg).run(runner, roots, add_noise=True)
    assert len(results) == 2
    for (visits, root_value), legal in zip(results, [legal_a, legal_b]):
        assert sum(visits.values()) == 16
        assert set(visits.keys()) <= set(legal.tolist())
        assert np.isfinite(root_value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_mcts.py -v`
Expected: FAIL with `No module named 'muzero.mcts'`

- [ ] **Step 3: Implement**

`muzero/mcts.py`:
```python
"""Batched pUCT MCTS over learned dynamics (negamax backup, MinMax-normalized Q).

Legality masking is applied at the root only; interior nodes expand the
top-k prior actions of the full policy (MuZero standard — the learned
dynamics cannot produce a legal-move list)."""

from __future__ import annotations

import math
import threading

import numpy as np
import torch

from muzero.config import MuZeroConfig


class NetRunner:
    """Thread-safe batched inference wrapper around MuZeroNet."""

    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.lock = threading.Lock()

    def initial(self, obs_batch: np.ndarray) -> dict:
        with self.lock, torch.inference_mode():
            self.net.eval()  # BatchNorm: batch-stats + running-stat mutation otherwise
            obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(self.device)
            out = self.net.initial_inference(obs)
        return self._detach(out)

    def recurrent(self, hidden: torch.Tensor, actions: np.ndarray) -> dict:
        with self.lock, torch.inference_mode():
            self.net.eval()  # see initial()
            acts = torch.from_numpy(np.ascontiguousarray(actions)).to(self.device)
            out = self.net.recurrent_inference(hidden, acts)
        return self._detach(out)

    @staticmethod
    def _detach(out: dict) -> dict:
        keep = {"hidden": out["hidden"]}
        for k in ("policy_logits", "value", "reward"):
            if k in out:
                keep[k] = out[k].float().cpu().numpy()
        return keep


class MinMaxStats:
    def __init__(self):
        self.minimum, self.maximum = float("inf"), float("-inf")

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "reward", "hidden",
                 "cand_actions", "cand_priors", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.hidden = None  # row-tensor in the net's hidden batch
        self.cand_actions = None  # np.ndarray of action indices
        self.cand_priors = None  # np.ndarray aligned with cand_actions
        self.children = {}  # candidate position -> Node

    def expanded(self) -> bool:
        return self.cand_actions is not None

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, config: MuZeroConfig):
        self.config = config

    def run(self, runner: NetRunner, roots_data: list, add_noise: bool) -> list:
        """roots_data: list of (obs (115,10,9) float32, legal action indices).
        Returns per game: ({action: visit_count}, root_value)."""
        cfg = self.config
        obs_batch = np.stack([obs for obs, _ in roots_data])
        out = runner.initial(obs_batch)
        roots, stats = [], []
        for g, (_, legal) in enumerate(roots_data):
            root = Node(0.0)
            priors = _masked_softmax(out["policy_logits"][g], legal)
            if add_noise:
                noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(legal))
                priors = (1 - cfg.exploration_fraction) * priors + cfg.exploration_fraction * noise
            _expand(root, legal, priors, out["hidden"][g], reward=0.0)
            roots.append(root)
            stats.append(MinMaxStats())

        for _ in range(cfg.num_simulations):
            paths, hiddens, actions = [], [], []
            for g, root in enumerate(roots):
                node, path = root, [root]
                while node.expanded():
                    node = self._select_child(node, stats[g])
                    path.append(node)
                paths.append(path)
                hiddens.append(path[-2].hidden)
                actions.append(path[-1].prior_action)
            out = runner.recurrent(torch.stack(list(hiddens)), np.array(actions, dtype=np.int64))
            for g, path in enumerate(paths):
                leaf = path[-1]
                logits = out["policy_logits"][g]
                topk = np.argpartition(logits, -cfg.interior_topk)[-cfg.interior_topk:]
                _expand(leaf, topk.astype(np.int64), _masked_softmax(logits, topk),
                        out["hidden"][g], reward=float(out["reward"][g]))
                self._backup(path, float(out["value"][g]), stats[g])
        return [
            ({int(root.cand_actions[p]): ch.visit_count for p, ch in root.children.items()},
             root.value())
            for root in roots
        ]

    def _select_child(self, node: Node, stats: MinMaxStats) -> Node:
        cfg = self.config
        n = node.cand_priors.shape[0]
        q = np.zeros(n, dtype=np.float32)
        visits = np.zeros(n, dtype=np.float32)
        for pos, ch in node.children.items():
            visits[pos] = ch.visit_count
            if ch.visit_count > 0:
                q[pos] = stats.normalize(ch.reward + cfg.discount * -ch.value())
        pb_c = (
            math.log((node.visit_count + cfg.pb_c_base + 1) / cfg.pb_c_base) + cfg.pb_c_init
        ) * math.sqrt(max(node.visit_count, 1)) / (1.0 + visits)
        pos = int(np.argmax(q + pb_c * node.cand_priors))
        child = node.children.get(pos)
        if child is None:
            child = Node(float(node.cand_priors[pos]))
            child.prior_action = int(node.cand_actions[pos])
            node.children[pos] = child
        return child

    def _backup(self, path: list, leaf_value: float, stats: MinMaxStats):
        cfg = self.config
        v = leaf_value  # perspective of the player to move at the leaf
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            stats.update(node.reward + cfg.discount * -v)
            v = node.reward + cfg.discount * -v


def _expand(node: Node, actions: np.ndarray, priors: np.ndarray, hidden, reward: float):
    node.cand_actions = np.asarray(actions, dtype=np.int64)
    node.cand_priors = np.asarray(priors, dtype=np.float32)
    node.hidden = hidden
    node.reward = reward


def _masked_softmax(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    x = logits[indices].astype(np.float64)
    x = np.exp(x - x.max())
    return (x / x.sum()).astype(np.float32)
```

Note: `Node.prior_action` is assigned dynamically in `_select_child`; add `"prior_action"` to `__slots__`:
```python
    __slots__ = ("prior", "visit_count", "value_sum", "reward", "hidden",
                 "cand_actions", "cand_priors", "children", "prior_action")
```
(Use this final version of `__slots__` when writing the file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_mcts.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): batched pUCT MCTS with root legality masking"
```

---

### Task 10: Self-play workers + ally/enemy promotion

**Files:**
- Create: `muzero/selfplay.py`
- Test: `muzero/tests/test_selfplay.py`

- [ ] **Step 1: Write the failing tests**

`muzero/tests/test_selfplay.py`:
```python
import numpy as np
import torch

from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.mcts import NetRunner
from muzero.network import MuZeroNet
from muzero.replay_buffer import ReplayBuffer
from muzero.selfplay import SelfPlayCoordinator, SelfPlayWorker, select_action
from muzero.tests.helpers import make_evaluator, requires_engine


def test_coordinator_promotes_after_streak():
    cfg = replace(MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu")
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    for _ in range(2):
        coord.report_result(ally_won=True, draw=False)
    assert coord.era == 0
    coord.report_result(ally_won=False, draw=False)  # streak resets
    for _ in range(3):
        coord.report_result(ally_won=True, draw=False)
    assert coord.era == 1
    for pa, pe in zip(ally.parameters(), enemy.parameters()):
        assert torch.equal(pa, pe)


def test_select_action_temperature():
    visits = {7: 10, 9: 2}
    rng = np.random.default_rng(0)
    assert select_action(visits, ply=100, temperature_moves=30, rng=rng) == 7  # argmax
    picks = {select_action(visits, ply=0, temperature_moves=30, rng=rng) for _ in range(50)}
    assert picks == {7, 9}  # sampling explores both


@requires_engine
def test_selfplay_smoke_generates_games():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1,
        num_simulations=8, interior_topk=8, games_per_worker=2,
        max_game_plies=6, device="cpu",
    )
    torch.manual_seed(0)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    worker = SelfPlayWorker(
        cfg, NetRunner(ally, "cpu"), NetRunner(enemy, "cpu"),
        buf, coord, make_evaluator(), worker_id=0,
    )
    summaries = worker.generate(num_games=2)
    assert len(summaries) == 2
    assert len(buf.games) == 2
    game = buf.games[0]
    assert 1 <= len(game) <= 6
    assert len(game.boards) == len(game) + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_selfplay.py -v`
Expected: FAIL with `No module named 'muzero.selfplay'`

- [ ] **Step 3: Implement**

`muzero/selfplay.py`:
```python
"""Self-play: lockstep multi-game workers, opening book, frozen-enemy promotion.

Workers are threads (engine I/O and GPU calls release the GIL); each worker
owns one PikafishEvaluator and `games_per_worker` concurrent games. The ally
net moves for `ally_side`; the frozen enemy net moves for the other side."""

from __future__ import annotations

import threading

import numpy as np
import torch

from muzero.config import MuZeroConfig
from muzero.encoding import move_to_index, index_to_move
from muzero.env import XiangqiEnv
from muzero.mcts import MCTS, NetRunner
from muzero.replay_buffer import GameHistory, ReplayBuffer
from src.xiangqi_board import engine_uci_to_algebraic


def select_action(visits: dict, ply: int, temperature_moves: int, rng) -> int:
    actions = np.array(list(visits.keys()), dtype=np.int64)
    counts = np.array(list(visits.values()), dtype=np.float64)
    if ply >= temperature_moves:
        return int(actions[np.argmax(counts)])
    probs = counts / counts.sum()
    return int(rng.choice(actions, p=probs))


class SelfPlayCoordinator:
    """Tracks consecutive ally wins across all workers; promotes the enemy."""

    def __init__(self, config: MuZeroConfig, ally_net, enemy_net):
        self.config = config
        self.ally_net = ally_net
        self.enemy_net = enemy_net
        self.lock = threading.Lock()
        self.streak = 0
        self.era = 0
        self.games_this_era = 0

    def report_result(self, ally_won: bool, draw: bool) -> bool:
        with self.lock:
            self.games_this_era += 1
            self.streak = self.streak + 1 if (ally_won and not draw) else 0
            if self.streak >= self.config.promote_after_consecutive_wins:
                with torch.no_grad():
                    self.enemy_net.load_state_dict(self.ally_net.state_dict())
                self.enemy_net.eval()
                self.streak = 0
                self.era += 1
                self.games_this_era = 0
                return True
        return False


class _Game:
    def __init__(self, env: XiangqiEnv, history: GameHistory, opening_uci: str):
        self.env = env
        self.history = history
        self.opening_uci = opening_uci


class SelfPlayWorker:
    def __init__(self, config, ally_runner: NetRunner, enemy_runner: NetRunner,
                 buffer: ReplayBuffer, coordinator: SelfPlayCoordinator,
                 evaluator, worker_id: int):
        self.cfg = config
        self.ally_runner = ally_runner
        self.enemy_runner = enemy_runner
        self.buffer = buffer
        self.coordinator = coordinator
        self.evaluator = evaluator
        self.worker_id = worker_id
        self.rng = np.random.default_rng(config.seed + worker_id + 1)
        self.mcts = MCTS(config, rng=self.rng)  # seeded noise, thread-local RNG
        self.games_started = 0

    def _new_game(self) -> _Game:
        ally_side = "w" if self.games_started % 2 == 0 else "b"
        opening = self.cfg.opening_book[self.games_started % len(self.cfg.opening_book)]
        self.games_started += 1
        env = XiangqiEnv(self.cfg, self.evaluator)
        env.reset(ally_side=ally_side)
        history = GameHistory()
        history.ally_side = ally_side
        game = _Game(env, history, opening)
        self._play_forced_opening(game)
        return game

    def _play_forced_opening(self, game: _Game):
        move = engine_uci_to_algebraic(game.opening_uci)
        idx = move_to_index(move)
        self._record_and_step(game, idx, {idx: 1}, root_value=0.0)

    def _record_and_step(self, game: _Game, action: int, visits: dict, root_value: float):
        h = game.history
        total = sum(visits.values())
        h.actions.append(action)
        h.policy_indices.append(np.array(list(visits.keys()), dtype=np.int64))
        h.policy_probs.append(
            np.array([v / total for v in visits.values()], dtype=np.float32)
        )
        h.root_values.append(float(root_value))
        _, reward, done, info = game.env.step(index_to_move(action))
        h.rewards.append(reward)
        return done, info

    def _finish(self, game: _Game) -> dict:
        env, h = game.env, game.history
        h.boards = [b.copy() for b in env.boards]
        h.to_play_history = list(env.to_play_history)
        h.rep_history = list(env.rep_history)
        h.no_progress_history = list(env.no_progress_history)
        h.result = env.result
        h.truncated = env.truncated
        ally_won = (env.result == "red_win") == (env.ally_side == "w") and env.result in (
            "red_win", "black_win"
        )
        draw = env.result not in ("red_win", "black_win")
        promoted = self.coordinator.report_result(ally_won=ally_won, draw=draw)
        self.buffer.add(h)
        final_red_cp = env.red_cp()
        return {
            "result": env.result, "ally_side": env.ally_side, "ally_won": ally_won,
            "draw": draw, "plies": len(h), "truncated": env.truncated,
            "promoted": promoted, "final_red_cp": final_red_cp,
            "era": self.coordinator.era,
        }

    def generate(self, num_games: int) -> list:
        """Play `num_games` to completion in lockstep; returns game summaries."""
        summaries = []
        active = [self._new_game() for _ in range(min(self.cfg.games_per_worker, num_games))]
        while active:
            for runner, want_ally in ((self.ally_runner, True), (self.enemy_runner, False)):
                group = [
                    g for g in active
                    if (g.env.side_to_move == g.env.ally_side) == want_ally
                ]
                if not group:
                    continue
                roots = []
                for g in group:
                    legal = np.array(
                        [move_to_index(m) for m in g.env.legal_moves()], dtype=np.int64
                    )
                    roots.append((g.env.observation().astype(np.float32), legal))
                results = self.mcts.run(runner, roots, add_noise=want_ally)
                for g, (visits, root_value) in zip(group, results):
                    action = select_action(
                        visits, g.env.plies, self.cfg.temperature_moves, self.rng
                    )
                    done, _ = self._record_and_step(g, action, visits, root_value)
                    if done:
                        summaries.append(self._finish(g))
                        active.remove(g)
                        if self.games_started < num_games:
                            active.append(self._new_game())
        return summaries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_selfplay.py -v`
Expected: PASS (smoke test skipped without `PIKAFISH_BIN`)

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): lockstep self-play workers with frozen-enemy promotion"
```

---

### Task 11: Warm start (Pikafish games + MultiPV soft targets)

**Files:**
- Create: `muzero/warmstart.py`
- Test: `muzero/tests/test_warmstart.py`

- [ ] **Step 1: Write the failing tests**

`muzero/tests/test_warmstart.py`:
```python
from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.replay_buffer import ReplayBuffer
from muzero.tests.helpers import PIKAFISH_BIN, make_evaluator, requires_engine
from muzero.warmstart import SimpleUciEngine, generate_warmstart_games

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


@requires_engine
def test_multipv_returns_scored_moves():
    eng = SimpleUciEngine(PIKAFISH_BIN, movetime_ms=50, multipv=4)
    try:
        lines = eng.search(START_FEN)
    finally:
        eng.close()
    assert 1 <= len(lines) <= 4
    for uci, cp in lines:
        assert len(uci) == 4 and isinstance(cp, float)


@requires_engine
def test_warmstart_fills_buffer():
    cfg = replace(MuZeroConfig(), warmstart_plies=8, max_game_plies=6, warmstart_movetime_ms=20)
    buf = ReplayBuffer(cfg)
    stats = generate_warmstart_games(cfg, buf, make_evaluator())
    assert stats["plies"] >= 8
    assert len(buf.games) >= 1
    g = buf.games[0]
    assert len(g.policy_indices[0]) >= 1
    assert abs(g.root_values[0]) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_warmstart.py -v`
Expected: FAIL with `No module named 'muzero.warmstart'` (or SKIP without engine — in that case proceed; CI-less repo relies on the engine machine)

- [ ] **Step 3: Implement**

`muzero/warmstart.py`:
```python
"""Cold-start buffer fill: Pikafish-vs-Pikafish games with MultiPV soft
policy targets. Uses a small dedicated UCI wrapper so the shared
PikafishEvaluator (and the LLM pipeline that depends on it) stays untouched."""

from __future__ import annotations

import re
import subprocess

import numpy as np

from muzero.config import MuZeroConfig
from muzero.encoding import move_to_index
from muzero.env import XiangqiEnv
from muzero.replay_buffer import GameHistory, ReplayBuffer
from src.xiangqi_board import engine_uci_to_algebraic

_INFO_RE = re.compile(
    r"multipv (\d+) score (cp|mate) (-?\d+).* pv ([a-i]\d[a-i]\d)"
)


class SimpleUciEngine:
    def __init__(self, binary_path: str, movetime_ms: int, multipv: int):
        self.movetime_ms = movetime_ms
        self.proc = subprocess.Popen(
            [binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._cmd("uci")
        self._wait("uciok")
        self._cmd(f"setoption name MultiPV value {multipv}")
        self._cmd("isready")
        self._wait("readyok")

    def _cmd(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str) -> list:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def search(self, fen: str) -> list:
        """Returns [(engine_uci, cp_side_to_move)] best-first, one per multipv."""
        self._cmd(f"position fen {fen}")
        self._cmd(f"go movetime {self.movetime_ms}")
        lines = self._wait("bestmove")
        best: dict = {}
        for line in lines:
            m = _INFO_RE.search(line)
            if m:
                rank = int(m.group(1))
                cp = float(m.group(3)) if m.group(2) == "cp" else float(
                    np.sign(int(m.group(3))) * 30000
                )
                best[rank] = (m.group(4), cp)
        return [best[r] for r in sorted(best)]

    def close(self):
        try:
            self._cmd("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def generate_warmstart_games(cfg: MuZeroConfig, buffer: ReplayBuffer, evaluator) -> dict:
    """Play engine-vs-engine games until >= cfg.warmstart_plies plies are stored."""
    engine = SimpleUciEngine(cfg.pikafish_bin, cfg.warmstart_movetime_ms, cfg.warmstart_multipv)
    rng = np.random.default_rng(cfg.seed)
    total_plies = games = 0
    try:
        while total_plies < cfg.warmstart_plies:
            env = XiangqiEnv(cfg, evaluator)
            env.reset(ally_side="w")
            history = GameHistory()
            opening = cfg.opening_book[int(rng.integers(len(cfg.opening_book)))]
            done = _play_move(env, history, engine_uci_to_algebraic(opening), None)
            while not done:
                lines = engine.search(env.fen())
                if not lines:
                    break
                move = engine_uci_to_algebraic(lines[0][0])
                done = _play_move(env, history, move, lines)
            history.boards = [b.copy() for b in env.boards]
            history.to_play_history = list(env.to_play_history)
            history.rep_history = list(env.rep_history)
            history.no_progress_history = list(env.no_progress_history)
            history.result = env.result or "draw_max_plies"
            buffer.add(history)
            total_plies += len(history)
            games += 1
    finally:
        engine.close()
    return {"plies": total_plies, "games": games}


def _play_move(env: XiangqiEnv, history: GameHistory, move: str, multipv_lines) -> bool:
    if multipv_lines:
        idx = np.array(
            [move_to_index(engine_uci_to_algebraic(u)) for u, _ in multipv_lines],
            dtype=np.int64,
        )
        cps = np.array([cp for _, cp in multipv_lines], dtype=np.float64)
        probs = np.exp((cps - cps.max()) / 200.0)
        probs = (probs / probs.sum()).astype(np.float32)
        root_value = float(np.tanh(cps[0] / 600.0))  # mover perspective
    else:  # forced opening ply: one-hot
        idx = np.array([move_to_index(move)], dtype=np.int64)
        probs = np.array([1.0], dtype=np.float32)
        root_value = 0.0
    history.actions.append(move_to_index(move))
    history.policy_indices.append(idx)
    history.policy_probs.append(probs)
    history.root_values.append(root_value)
    _, reward, done, _ = env.step(move)
    history.rewards.append(reward)
    return done
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_warmstart.py -v`
Expected: PASS on a machine with `PIKAFISH_BIN`; SKIP otherwise

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): Pikafish warm-start games with MultiPV soft targets"
```

---

### Task 12: Training step (combined loss)

**Files:**
- Create: `muzero/train.py` (trainer class only; the main loop comes in Task 13)
- Test: `muzero/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_train.py`:
```python
import torch

from dataclasses import replace

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.replay_buffer import ReplayBuffer
from muzero.tests.test_replay_buffer import make_game
from muzero.train import MuZeroTrainer


def test_train_batch_runs_and_updates_params():
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1,
        batch_size=4, unroll_steps=8, device="cpu",
    )
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    buf = ReplayBuffer(cfg)
    for _ in range(2):
        buf.add(make_game())
    trainer = MuZeroTrainer(cfg, net)
    before = [p.detach().clone() for p in net.parameters()]
    losses = trainer.train_batch(buf.sample_batch(4))
    for key in ("policy", "value", "reward", "moves_left", "material", "consistency", "total"):
        assert key in losses and torch.isfinite(torch.tensor(losses[key])), key
    changed = any(
        not torch.equal(b, p.detach()) for b, p in zip(before, net.parameters())
    )
    assert changed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_train.py -v`
Expected: FAIL with `No module named 'muzero.train'`

- [ ] **Step 3: Implement**

`muzero/train.py` (first part — Task 13 appends `main`):
```python
"""Training: K-step unrolled combined loss + the main orchestration loop."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.transforms import h_transform, scalar_to_support


def scale_gradient(t: torch.Tensor, scale: float) -> torch.Tensor:
    return t * scale + t.detach() * (1.0 - scale)


def _soft_ce(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(-1)


class MuZeroTrainer:
    def __init__(self, cfg: MuZeroConfig, net: MuZeroNet):
        self.cfg = cfg
        self.net = net
        self.optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.train_steps = 0

    def _to_tensors(self, batch: dict) -> dict:
        dev = next(self.net.parameters()).device
        out = {}
        for k, v in batch.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("actions", "target_moves_left", "consistency_k"):
                t = t.long()
            else:
                t = t.float()
            out[k] = t.to(dev)
        return out

    def train_batch(self, batch: dict) -> dict:
        cfg = self.cfg
        self.net.train()
        b = self._to_tensors(batch)
        B, K = b["actions"].shape

        value_support = scalar_to_support(
            h_transform(b["target_value"]), -cfg.value_max, cfg.value_max, cfg.value_bins
        )
        reward_support = scalar_to_support(
            b["target_reward"], -cfg.reward_max, cfg.reward_max, cfg.reward_bins
        )

        out = self.net.initial_inference(b["obs"])
        losses = {
            "policy": (_soft_ce(out["policy_logits"], b["target_policy"][:, 0]) * b["policy_mask"][:, 0]).mean(),
            "value": _soft_ce(out["value_logits"], value_support[:, 0]).mean(),
            "reward": torch.zeros((), device=b["obs"].device),
            "moves_left": F.cross_entropy(out["moves_left_logits"], b["target_moves_left"][:, 0]),
            "material": F.mse_loss(out["material"], b["target_material"][:, 0]),
        }
        hidden = out["hidden"]
        latents = []
        for k in range(1, K + 1):
            out_k = self.net.recurrent_inference(hidden, b["actions"][:, k - 1])
            hidden = scale_gradient(out_k["hidden"], 0.5)
            latents.append(hidden)
            losses["policy"] = losses["policy"] + (
                _soft_ce(out_k["policy_logits"], b["target_policy"][:, k]) * b["policy_mask"][:, k]
            ).mean() / K
            losses["value"] = losses["value"] + _soft_ce(out_k["value_logits"], value_support[:, k]).mean() / K
            losses["reward"] = losses["reward"] + _soft_ce(out_k["reward_logits"], reward_support[:, k - 1]).mean() / K
            losses["moves_left"] = losses["moves_left"] + F.cross_entropy(
                out_k["moves_left_logits"], b["target_moves_left"][:, k]
            ) / K
            losses["material"] = losses["material"] + F.mse_loss(
                out_k["material"], b["target_material"][:, k]
            ) / K

        # SimSiam consistency at one sampled unroll offset per sample.
        k_c = b["consistency_k"]
        mask = (k_c > 0).float()
        if mask.sum() > 0:
            stacked = torch.stack(latents, dim=1)  # (B, K, ch, 10, 9)
            gather = (k_c.clamp(min=1) - 1).view(B, 1, 1, 1, 1).expand(
                -1, 1, *stacked.shape[2:]
            )
            dyn_latent = stacked.gather(1, gather).squeeze(1)
            with torch.no_grad():
                target_latent = self.net.representation(b["consistency_obs"])
                from muzero.network import normalize_hidden

                target_proj = self.net.project(normalize_hidden(target_latent), with_predictor=False)
            pred = self.net.project(dyn_latent, with_predictor=True)
            cos = F.cosine_similarity(pred, target_proj.detach(), dim=-1)
            losses["consistency"] = (-(cos * mask).sum() / mask.sum().clamp(min=1.0))
        else:
            losses["consistency"] = torch.zeros((), device=b["obs"].device)

        w = cfg.loss_weights
        total = (
            w[0] * losses["policy"] + w[1] * losses["value"] + w[2] * losses["reward"]
            + w[3] * losses["moves_left"] + w[4] * losses["material"] + w[5] * losses["consistency"]
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.train_steps += 1
        result = {k: float(v.detach()) for k, v in losses.items()}
        result["total"] = float(total.detach())
        return result
```

Move the `from muzero.network import normalize_hidden` import to the top of the file with the other imports when writing it (shown inline above only for reading flow).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_train.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): unrolled combined loss + SimSiam consistency"
```

---

### Task 13: Metrics, fixed-opponent gate, main loop

**Files:**
- Create: `muzero/metrics.py`
- Modify: `muzero/train.py` (append gate + main loop)
- Test: `muzero/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`muzero/tests/test_metrics.py`:
```python
from muzero.metrics import aggregate_game_summaries


def test_aggregate_game_summaries():
    summaries = [
        {"result": "red_win", "ally_side": "w", "ally_won": True, "draw": False,
         "plies": 40, "truncated": False, "promoted": False, "final_red_cp": 250.0, "era": 0},
        {"result": "draw_repetition", "ally_side": "b", "ally_won": False, "draw": True,
         "plies": 60, "truncated": False, "promoted": False, "final_red_cp": 0.0, "era": 0},
        {"result": "black_win", "ally_side": "w", "ally_won": False, "draw": False,
         "plies": 30, "truncated": True, "promoted": False, "final_red_cp": None, "era": 0},
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/win_rate"] == 1 / 3
    assert m["selfplay/draw_rate"] == 1 / 3
    assert m["selfplay/loss_rate"] == 1 / 3
    assert m["selfplay/repetition_draw_rate"] == 1 / 3
    assert m["selfplay/truncation_rate"] == 1 / 3
    assert m["selfplay/mean_plies"] == 130 / 3
    assert m["selfplay/mean_final_ally_cp"] == 125.0  # (250 + 0)/2, None skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_metrics.py -v`
Expected: FAIL with `No module named 'muzero.metrics'`

- [ ] **Step 3: Implement `muzero/metrics.py`**

```python
"""wandb logging + per-loop aggregation of self-play game summaries."""

from __future__ import annotations

from muzero.config import MuZeroConfig


def aggregate_game_summaries(summaries: list) -> dict:
    n = max(len(summaries), 1)
    wins = sum(1 for s in summaries if s["ally_won"])
    draws = sum(1 for s in summaries if s["draw"])
    ally_cps = [
        (s["final_red_cp"] if s["ally_side"] == "w" else -s["final_red_cp"])
        for s in summaries
        if s["final_red_cp"] is not None
    ]
    return {
        "selfplay/win_rate": wins / n,
        "selfplay/draw_rate": draws / n,
        "selfplay/loss_rate": (len(summaries) - wins - draws) / n,
        "selfplay/repetition_draw_rate": sum(
            1 for s in summaries if s["result"] == "draw_repetition"
        ) / n,
        "selfplay/truncation_rate": sum(1 for s in summaries if s["truncated"]) / n,
        "selfplay/mean_plies": sum(s["plies"] for s in summaries) / n,
        "selfplay/mean_final_ally_cp": (
            sum(ally_cps) / len(ally_cps) if ally_cps else 0.0
        ),
        "selfplay/promotions": sum(1 for s in summaries if s["promoted"]),
        "selfplay/era": max((s["era"] for s in summaries), default=0),
        "selfplay/games": len(summaries),
    }


class MetricsLogger:
    def __init__(self, cfg: MuZeroConfig, enabled: bool = True):
        self.enabled = enabled
        self.wandb = None
        if enabled:
            import wandb

            self.wandb = wandb
            wandb.init(project=cfg.wandb_project, config=vars(cfg))

    def log(self, metrics: dict, step: int):
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
```

- [ ] **Step 4: Append the gate and main loop to `muzero/train.py`**

```python
def run_gate(cfg: MuZeroConfig, runner, evaluator) -> dict:
    """Ally (MCTS, no noise, argmax) vs raw Pikafish at gate movetime."""
    from muzero.encoding import index_to_move, move_to_index
    from muzero.env import XiangqiEnv
    from muzero.mcts import MCTS
    from muzero.warmstart import SimpleUciEngine
    from src.xiangqi_board import algebraic_to_engine_move, engine_uci_to_algebraic

    engine = SimpleUciEngine(cfg.pikafish_bin, cfg.gate_movetime_ms, multipv=1)
    mcts = MCTS(cfg)
    wins = draws = 0
    try:
        for i in range(cfg.gate_games):
            ally_side = "w" if i % 2 == 0 else "b"
            env = XiangqiEnv(cfg, evaluator)
            env.reset(ally_side=ally_side)
            done = False
            while not done:
                if env.side_to_move == ally_side:
                    legal = np.array(
                        [move_to_index(m) for m in env.legal_moves()], dtype=np.int64
                    )
                    (visits, _), = mcts.run(
                        runner, [(env.observation().astype(np.float32), legal)], add_noise=False
                    )
                    move = index_to_move(max(visits, key=visits.get))
                else:
                    lines = engine.search(env.fen())
                    if not lines:
                        break
                    move = engine_uci_to_algebraic(lines[0][0])
                _, _, done, _ = env.step(move)
            if env.result in ("red_win", "black_win"):
                if (env.result == "red_win") == (ally_side == "w"):
                    wins += 1
            else:
                draws += 1
    finally:
        engine.close()
    n = cfg.gate_games
    return {"gate/win_rate": wins / n, "gate/draw_rate": draws / n,
            "gate/loss_rate": (n - wins - draws) / n}


def main():
    import argparse
    import copy
    import os

    from src.pikafish_eval import PikafishEvaluator

    from muzero.mcts import NetRunner
    from muzero.metrics import MetricsLogger, aggregate_game_summaries
    from muzero.replay_buffer import ReplayBuffer
    from muzero.selfplay import SelfPlayCoordinator, SelfPlayWorker
    from muzero.warmstart import generate_warmstart_games

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    args = parser.parse_args()

    cfg = MuZeroConfig()
    if args.device:
        cfg.device = args.device
    if args.smoke:
        cfg.channels, cfg.repr_blocks, cfg.dyn_blocks = 16, 1, 1
        cfg.num_simulations, cfg.interior_topk = 8, 8
        cfg.num_workers, cfg.games_per_worker = 1, 2
        cfg.max_game_plies, cfg.batch_size = 6, 4
        cfg.warmstart_plies, cfg.warmstart_train_batches = 8, 2
        cfg.games_per_train_loop, cfg.gate_every_loops = 2, 10**9

    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    ally = MuZeroNet(cfg).to(device)
    enemy = copy.deepcopy(ally).to(device)
    enemy.eval()
    trainer = MuZeroTrainer(cfg, ally)
    start_iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ally.load_state_dict(ckpt["ally"])
        enemy.load_state_dict(ckpt["enemy"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"]

    buffer = ReplayBuffer(cfg)
    ally_runner, enemy_runner = NetRunner(ally, device), NetRunner(enemy, device)
    # enemy_lock: promotion must not swap weights mid-forward-pass (see Task 10)
    coordinator = SelfPlayCoordinator(cfg, ally, enemy, enemy_lock=enemy_runner.lock)
    if args.resume and "era" in torch.load(args.resume, map_location="cpu"):
        coordinator.era = torch.load(args.resume, map_location="cpu")["era"]

    def make_evaluator():
        return PikafishEvaluator(
            binary_path=cfg.pikafish_bin, depth=cfg.pikafish_depth,
            timeout_sec=cfg.pikafish_timeout_sec, movetime_ms=cfg.pikafish_movetime_ms,
            verbose=False,
        )

    workers = [
        SelfPlayWorker(cfg, ally_runner, enemy_runner, buffer, coordinator,
                       make_evaluator(), worker_id=w)
        for w in range(cfg.num_workers)
    ]
    gate_evaluator = make_evaluator()
    logger = MetricsLogger(cfg, enabled=not args.no_wandb)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if not buffer.games:
        print("[warmstart] generating Pikafish games ...")
        stats = generate_warmstart_games(cfg, buffer, workers[0].evaluator)
        print(f"[warmstart] {stats['games']} games / {stats['plies']} plies")
        for _ in range(cfg.warmstart_train_batches):
            trainer.train_batch(buffer.sample_batch(cfg.batch_size))

    import threading

    for it in range(start_iteration, args.iterations):
        # -- generate --
        games_per_worker_now = cfg.games_per_worker
        results, threads = [], []
        for w in workers:
            t = threading.Thread(
                target=lambda w=w: results.extend(w.generate(games_per_worker_now))
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        metrics = aggregate_game_summaries(results)

        # -- train: ~games_per_train_loop games' worth of positions --
        num_batches = max(
            1, int(cfg.games_per_train_loop * buffer.mean_game_length() // cfg.batch_size)
        )
        loss_sums: dict = {}
        for _ in range(num_batches):
            for k, v in trainer.train_batch(buffer.sample_batch(cfg.batch_size)).items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
        metrics.update({f"loss/{k}": v / num_batches for k, v in loss_sums.items()})
        metrics.update({
            "buffer/games": len(buffer.games),
            "buffer/mean_game_length": buffer.mean_game_length(),
            "buffer/total_games_added": buffer.total_games_added,
            "train/batches": num_batches,
            "train/steps": trainer.train_steps,
        })

        # -- gate --
        if (it + 1) % cfg.gate_every_loops == 0:
            metrics.update(run_gate(cfg, ally_runner, gate_evaluator))

        logger.log(metrics, step=it)
        print(f"[iter {it}] " + " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())
                                          if isinstance(v, float)))
        torch.save(
            {"ally": ally.state_dict(), "enemy": enemy.state_dict(),
             "optimizer": trainer.optimizer.state_dict(), "iteration": it + 1,
             "era": coordinator.era},
            os.path.join(cfg.checkpoint_dir, "latest.pt"),
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the metrics test and the full suite**

Run: `uv run pytest muzero/tests -v`
Expected: all PASS (engine tests skipped without `PIKAFISH_BIN`)

- [ ] **Step 6: End-to-end smoke run (engine machine only)**

Run: `uv run python -m muzero.train --smoke --no-wandb --iterations 1 --device cpu`
Expected: warmstart prints, one `[iter 0]` line with finite losses, `checkpoints/muzero_xiangqi/latest.pt` created. If `PIKAFISH_BIN` is unavailable on this machine, defer this step to the training machine and note it in the log.

- [ ] **Step 7: Lint and commit**

```bash
ruff check muzero --fix && ruff format muzero
git add muzero
git commit -m "feat(muzero): metrics, fixed-opponent gate, training main loop"
```

---

### Task 14: Docs handoff

**Files:**
- Modify: `docs/ARCHITECTURE.md` (repo map + new component section)
- Modify: `docs/AGENT_TODO.md` (close the MuZero task, add follow-ups)
- Create: `docs/logs/<today>-log-muzero-implementation.md` (from `docs/logs/template.md`)

- [ ] **Step 1: Update `docs/ARCHITECTURE.md`**

Add to the Repository Map list:
```markdown
- `muzero/` — MuZero/EfficientZero-style Xiangqi agent (tensor world model + MCTS self-play); Pikafish-only legality; independent of the LLM pipeline. Entrypoint `python -m muzero.train`. Spec: `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`.
```

Add a new component section after §3e:
```markdown
### 3f. MuZero Xiangqi (tensor world model + MCTS)
- **Description:** `muzero/` implements an EfficientZero-style agent per
  `docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`: 115×10×9 board
  tensors, 8100-action space masked to Pikafish-legal moves, 800-sim pUCT MCTS,
  K=8 unrolled training with policy/value/reward/moves-left/material/SimSiam
  losses, frozen-enemy self-play with promotion after 3 consecutive ally wins,
  repetition-draw + hopeless-truncation adjudication, Pikafish warm start, and
  a periodic fixed-Pikafish gate. Entrypoint: `python -m muzero.train`
  (`--smoke` for a tiny end-to-end run). Tests: `uv run pytest muzero/tests`.
```

- [ ] **Step 2: Update `docs/AGENT_TODO.md`**

Move the MuZero task to Completed with today's date; add follow-up backlog items discovered during implementation (at minimum: "first real training run on the 5090 + wandb review" and any deferred items).

- [ ] **Step 3: Write the dated log**

Create `docs/logs/<today>-log-muzero-implementation.md` following `docs/logs/template.md`: goal, what was built per task, test results (paste the pytest summary line), whether the smoke run executed, and next steps.

- [ ] **Step 4: Final verification + commit**

```bash
ruff check . --fix && ruff format muzero
uv run pytest muzero/tests -v
git add docs muzero
git commit -m "docs: MuZero subsystem architecture + handoff accounting"
```

---

## Self-Review Notes (already applied)

- Spec coverage: every spec section maps to a task — §3 layout (T1), §4 env (T6–T7), §5 encoding (T2–T3), §6 networks (T4–T5), §7 MCTS/self-play (T9–T10), §8 buffer/loss (T8, T12), §9 warm start (T11), §10 metrics/gate (T13), §11 testing (each task + smoke), docs handoff (T14).
- Known simplifications (documented, spec-compatible): workers are threads sharing a locked `NetRunner` instead of processes with an IPC inference server (same batching benefit, far less complexity — revisit only if GPU utilization is poor); PER importance-sampling correction weights omitted (priorities only bias sampling); SimSiam consistency applied at one random unroll offset per sample to keep batch memory bounded; interior MCTS nodes expand top-64 prior actions to bound tree memory at 800 sims × 84 games.
- Type consistency verified: `GameHistory` field names match between `replay_buffer.py`, `selfplay.py`, and `warmstart.py`; `NetRunner.initial/recurrent` signatures match `mcts.py` and `train.py` call sites; `MuZeroConfig` field names match all usages.
