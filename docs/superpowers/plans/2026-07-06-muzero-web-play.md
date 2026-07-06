# Human-vs-MuZero Web Play Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play Xiangqi against the canonical MuZero checkpoint through the existing `web/` UI, as either color, at gate strength (800 sims, argmax, no noise), CPU-first.

**Architecture:** Approach A per `docs/superpowers/specs/2026-07-06-muzero-web-play-design.md` — a `MuZeroPlayer` (checkpoint + the exact gate search path: `canonical_root` → `MCTS.run(add_noise=False)` → `absolute_visits` → argmax) and a `MuZeroGameSession` built on `muzero.env.XiangqiEnv`, selected at server launch via `XIANGQI_PLAY_ENGINE=muzero`. The LLM path is untouched. The UI gains a color picker (muzero mode only) and human-side-aware piece selection.

**Tech Stack:** Python 3.12, FastAPI (existing), PyTorch (muzero net), vanilla JS board UI, `uv` for everything, pytest.

**Key facts an implementer must know:**
- Branch: work on `muzero-web-play`, cut from current `main`.
- `muzero.env.XiangqiEnv(cfg, evaluator)` — `reset(ally_side="w"|"b")`, `side_to_move` ∈ "w"/"b", `legal_moves()` returns internal-algebraic strings, `step(move)` (asserts game not over), `result` ∈ {None, "red_win", "black_win", "draw_repetition", "draw_max_plies"}, `board` is signed int8 (positive=red) — same format as the gym `env.state` the existing session renders.
- `web/server/game_session.py` imports `web.server.engine_player`, which pulls in the 7B/transformers stack at import time — the new MuZero modules must NOT import `game_session` (Task 1 extracts the shared board-grid helper into a light module instead).
- `FakeEvaluator` (muzero/tests/helpers.py) returns legal moves as ENGINE-UCI strings; `env.legal_moves()` converts them (rank r → 9−r): engine `"a3a4"` ⇒ internal `"a6a5"` (a red pawn push), engine `"i6i5"` ⇒ internal `"i3i4"` (a black pawn push). `env.fen()`'s second whitespace token is the side to move (`"w"`/`"b"`).
- The board UI drives itself off snapshot `turn` ("human"/"greedy"/"engine"/"none"); `winnerLabel` in board.js already renders `"draw"`.
- Conventions: `uv run ruff check . --fix && uv run ruff format .` on changed Python; commit messages end with a blank line + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Web tests need package inits: create empty `web/__init__.py` and `web/tests/__init__.py` in Task 2.

---

### Task 0: Branch

- [ ] **Step 0.1:**

```bash
cd "/Users/fionachow/Documents/NYU/CDS/Spring 2024/DS-GA 3001.005 - Reinforcement Learning/Projects"
git checkout main && git pull --ff-only && git checkout -b muzero-web-play
```

---

### Task 1: Extract light board-view helpers; tag LLM session snapshots

The MuZero session must render the board grid without importing the 7B stack.

**Files:**
- Create: `web/server/board_view.py`
- Modify: `web/server/game_session.py` (lines 26-43 `_PIECE_FEN`, lines 94-106 `_board_grid`, snapshot dict ~line 164)

- [ ] **Step 1.1: Create `web/server/board_view.py`** (moved verbatim from game_session.py, names made public):

```python
"""Board-grid rendering shared by LLM and MuZero sessions (no heavy imports)."""

from __future__ import annotations

from typing import List

import numpy as np

PIECE_FEN = {
    1: "k",
    2: "a",
    3: "a",
    4: "b",
    5: "b",
    6: "n",
    7: "n",
    8: "r",
    9: "r",
    10: "c",
    11: "c",
    12: "p",
    13: "p",
    14: "p",
    15: "p",
    16: "p",
}


def board_grid(state: np.ndarray) -> List[List[str]]:
    """Signed int8 board -> UI letter grid (upper=red, lower=black, '.'=empty)."""
    rows: List[List[str]] = []
    for row in state:
        cells: List[str] = []
        for cell in row:
            val = int(cell)
            if val == 0:
                cells.append(".")
            else:
                base = PIECE_FEN.get(abs(val), "?")
                cells.append(base.upper() if val > 0 else base)
        rows.append(cells)
    return rows
```

- [ ] **Step 1.2: Update `web/server/game_session.py`** — delete its `_PIECE_FEN` dict and `_board_grid` function; add `from web.server.board_view import board_grid` to the imports; replace the one call site `_board_grid(self.env.state)` with `board_grid(self.env.state)`. Then, in `GameSession.snapshot()`'s returned dict, add two keys (UI mode detection, per spec §3d):

```python
            "engineKind": "llm",
            "humanSide": "red",
```

- [ ] **Step 1.3: Verify imports stay light and nothing broke**

```bash
uv run python -c "from web.server.board_view import PIECE_FEN, board_grid; import numpy as np; print(board_grid(np.zeros((10,9),dtype=np.int8))[0])"
uv run ruff check web --fix && uv run ruff format web
uv run pytest muzero/tests -q
```

Expected: `['.', '.', ...]` row printed; ruff clean; muzero suite unchanged (68 passed, 6 skipped). Do NOT try to import `web.server.game_session` as a smoke test — it pulls the 7B stack.

- [ ] **Step 1.4: Commit**

```bash
git add web/server/board_view.py web/server/game_session.py
git commit -m "refactor(web): extract light board-view helpers, tag session engineKind"
```

---

### Task 2: MuZeroPlayer

**Files:**
- Create: `web/__init__.py` (empty), `web/tests/__init__.py` (empty)
- Create: `web/server/muzero_player.py`
- Test: `web/tests/test_muzero_player.py`

- [ ] **Step 2.1: Write the failing tests** — create `web/tests/test_muzero_player.py`:

```python
from dataclasses import replace

import pytest
import torch

from muzero.config import MuZeroConfig
from muzero.network import MuZeroNet
from muzero.tests.helpers import FakeEvaluator
from web.server.muzero_player import MuZeroPlayer


def tiny_cfg(**over):
    return replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        max_game_plies=4,
        device="cpu",
        **over,
    )


def build_player(tmp_path, cfg):
    torch.manual_seed(0)
    net = MuZeroNet(cfg)
    ckpt = tmp_path / "latest.pt"
    torch.save({"ally": net.state_dict()}, ckpt)
    return MuZeroPlayer(str(ckpt), device="cpu", config=cfg)


def fake_evaluator():
    # ENGINE-UCI legals; env converts rank r -> 9-r to internal algebraic:
    # red to move gets internal "a6a5", black gets internal "i3i4".
    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    return FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)


def test_choose_move_returns_absolute_legal_move_for_both_colors(tmp_path):
    from muzero.env import XiangqiEnv

    cfg = tiny_cfg()
    player = build_player(tmp_path, cfg)
    env = XiangqiEnv(cfg, fake_evaluator())
    env.reset()
    assert player.choose_move(env) == "a6a5"  # red: single legal, absolute
    env.step("a6a5")
    assert player.choose_move(env) == "i3i4"  # black: single legal, absolute


def test_missing_checkpoint_raises_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
        MuZeroPlayer(str(tmp_path / "nonexistent.pt"))


def test_incompatible_checkpoint_raises_actionable_error(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"ally": {}}, ckpt)  # empty state dict -> load_state_dict fails
    with pytest.raises(RuntimeError, match="Incompatible"):
        MuZeroPlayer(str(ckpt), config=tiny_cfg())
```

Also create empty `web/__init__.py` and `web/tests/__init__.py` in this step so the imports resolve.

- [ ] **Step 2.2: Run to verify failure**

Run: `uv run pytest web/tests/test_muzero_player.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'web.server.muzero_player'`.

- [ ] **Step 2.3: Implement** — create `web/server/muzero_player.py`:

```python
"""MuZero checkpoint as a web-play opponent (exact gate-strength search)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from muzero.config import MuZeroConfig
from muzero.encoding import absolute_visits, index_to_move
from muzero.mcts import MCTS, NetRunner
from muzero.network import MuZeroNet
from muzero.selfplay import canonical_root


class MuZeroPlayer:
    """Loads a canonical (114-plane) checkpoint and picks argmax-MCTS moves.

    `num_simulations`/`config` are test hooks only; production uses the
    MuZeroConfig defaults (800 sims — "always full strength" per spec)."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        num_simulations: Optional[int] = None,
        config: Optional[MuZeroConfig] = None,
    ):
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"MuZero checkpoint not found: {path} — copy one from the "
                "training box or set XIANGQI_MUZERO_CKPT"
            )
        cfg = config or MuZeroConfig()
        cfg.device = device
        if num_simulations is not None:
            cfg.num_simulations = num_simulations
        self.cfg = cfg
        net = MuZeroNet(cfg).to(device)
        ckpt = torch.load(str(path), map_location=device)
        try:
            net.load_state_dict(ckpt["ally"])
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"Incompatible MuZero checkpoint {path} (pre-canonicalization "
                f"115-plane checkpoints cannot be loaded): {exc}"
            ) from exc
        net.eval()
        self.runner = NetRunner(net, device)
        self.mcts = MCTS(cfg)

    def choose_move(self, env) -> str:
        """Gate-strength move: no noise, argmax visits, ABSOLUTE algebraic."""
        obs, legal = canonical_root(env)
        ((visits, _, _),) = self.mcts.run(
            self.runner, [(obs, legal)], add_noise=False
        )
        visits = absolute_visits(visits, env.side_to_move)
        return index_to_move(max(visits, key=visits.get))
```

- [ ] **Step 2.4: Run tests**

Run: `uv run pytest web/tests -v` → 3 passed. Then `uv run ruff check web --fix && uv run ruff format web`.

- [ ] **Step 2.5: Commit**

```bash
git add web/__init__.py web/tests web/server/muzero_player.py
git commit -m "feat(web): MuZeroPlayer — checkpoint + gate-strength argmax MCTS"
```

---

### Task 3: MuZeroGameSession

**Files:**
- Create: `web/server/muzero_session.py`
- Test: `web/tests/test_muzero_session.py`

- [ ] **Step 3.1: Write the failing tests** — create `web/tests/test_muzero_session.py`:

```python
from web.server.muzero_session import MuZeroGameSession
from web.tests.test_muzero_player import build_player, fake_evaluator, tiny_cfg

SNAPSHOT_KEYS = {
    "board",
    "graphic",
    "fen",
    "allyMode",
    "engineKind",
    "humanSide",
    "sideToMove",
    "turn",
    "gameOver",
    "winner",
    "lastAllyMove",
    "lastEngineMove",
    "engineThinking",
}


def make_session(tmp_path, **cfg_over):
    cfg = tiny_cfg(**cfg_over)
    player = build_player(tmp_path, cfg)
    return MuZeroGameSession(fake_evaluator(), player, config=cfg)


def test_snapshot_contract_and_red_start(tmp_path):
    sess = make_session(tmp_path)
    snap = sess.reset(human_side="red")
    assert SNAPSHOT_KEYS <= set(snap)
    assert snap["engineKind"] == "muzero"
    assert snap["humanSide"] == "red"
    assert snap["turn"] == "human"  # red moves first
    assert snap["allyMode"] == "human"
    assert snap["gameOver"] is False


def test_red_full_exchange(tmp_path):
    sess = make_session(tmp_path)
    sess.reset(human_side="red")
    snap, err = sess.apply_human_move("a6a5")
    assert err is None
    assert snap["lastAllyMove"] == "a6a5"
    assert snap["turn"] == "engine"
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["lastEngineMove"] == "i3i4"  # absolute, model's only legal
    assert snap["turn"] == "human"


def test_black_engine_moves_first(tmp_path):
    sess = make_session(tmp_path)
    snap = sess.reset(human_side="black")
    assert snap["humanSide"] == "black"
    assert snap["turn"] == "engine"  # model is Red, moves first
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["lastEngineMove"] == "a6a5"
    assert snap["turn"] == "human"


def test_illegal_move_and_wrong_turn_rejected(tmp_path):
    sess = make_session(tmp_path)
    sess.reset(human_side="red")
    _, err = sess.apply_human_move("e6e5")
    assert err == "Move is not legal (Pikafish)"
    _, err = sess.apply_engine_move()
    assert err == "Not engine turn"
    _, err = sess.apply_greedy_ally()
    assert err == "Greedy ally is not supported with the MuZero engine"


def test_max_plies_draw_maps_to_draw_winner(tmp_path):
    sess = make_session(tmp_path, max_game_plies=2)
    sess.reset(human_side="red")
    sess.apply_human_move("a6a5")
    snap, err = sess.apply_engine_move()
    assert err is None
    assert snap["gameOver"] is True
    assert snap["winner"] == "draw"  # draw_max_plies -> "draw"
    assert snap["turn"] == "none"
    _, err = sess.apply_human_move("a6a5")
    assert err == "Game is already over"
```

- [ ] **Step 3.2: Run to verify failure**

Run: `uv run pytest web/tests/test_muzero_session.py -v`
Expected: `ModuleNotFoundError: No module named 'web.server.muzero_session'`.

- [ ] **Step 3.3: Implement** — create `web/server/muzero_session.py`:

```python
"""Human vs MuZero session on muzero's own env (spec 2026-07-06).

Snapshot contract mirrors web.server.game_session.GameSession so the same
board.js works: `lastAllyMove` = the HUMAN's move and `lastEngineMove` =
the model's move, whichever color each is playing."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from muzero.config import MuZeroConfig
from muzero.env import XiangqiEnv
from src.xiangqi_board import board_to_fen, board_to_graphic
from web.server.board_view import board_grid
from web.server.muzero_player import MuZeroPlayer

HumanSide = Literal["red", "black"]

_RESULT_WINNER = {
    "red_win": "red",
    "black_win": "black",
    "draw_repetition": "draw",
    "draw_max_plies": "draw",
}


class MuZeroGameSession:
    engine_kind = "muzero"

    def __init__(
        self,
        pikafish,
        player: Optional[MuZeroPlayer],
        config: Optional[MuZeroConfig] = None,
    ):
        cfg = config or MuZeroConfig()
        # Humans get to finish (or save) lost games: disable the training-time
        # hopeless-cp auto-adjudication. Repetition draws + ply cap remain.
        cfg.truncation_consecutive = 10**9
        self.cfg = cfg
        self.pikafish = pikafish
        self.player = player
        self.env = XiangqiEnv(cfg, pikafish)
        self.human_side: HumanSide = "red"
        self.game_over = False
        self.winner: Optional[str] = None
        self.last_ally_move: Optional[str] = None  # human's last move
        self.last_engine_move: Optional[str] = None  # model's last move
        self.engine_thinking = False
        self.reset(human_side="red")

    def _human_to_move(self) -> bool:
        return self.env.side_to_move == ("w" if self.human_side == "red" else "b")

    def reset(self, human_side: HumanSide = "red") -> Dict[str, Any]:
        self.human_side = human_side if human_side in ("red", "black") else "red"
        self.env.reset(ally_side="w" if self.human_side == "red" else "b")
        self.game_over = False
        self.winner = None
        self.last_ally_move = None
        self.last_engine_move = None
        self.engine_thinking = False
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        if self.game_over:
            turn = "none"
        elif self._human_to_move():
            turn = "human"
        else:
            turn = "engine"
        return {
            "board": board_grid(self.env.board),
            "graphic": board_to_graphic(self.env.board),
            "fen": board_to_fen(self.env.board),
            "allyMode": "human",
            "engineKind": self.engine_kind,
            "humanSide": self.human_side,
            "sideToMove": "ally" if self._human_to_move() else "enemy",
            "turn": turn,
            "gameOver": self.game_over,
            "winner": self.winner,
            "lastAllyMove": self.last_ally_move,
            "lastEngineMove": self.last_engine_move,
            "engineThinking": self.engine_thinking,
        }

    def legal_targets_from(self, from_sq: str) -> List[str]:
        if self.game_over or not self._human_to_move():
            return []
        from_sq = (from_sq or "").strip().lower()
        if len(from_sq) != 2:
            return []
        return sorted(
            {m[2:] for m in self.env.legal_moves() if m.startswith(from_sq)}
        )

    def _finish_if_over(self) -> None:
        if self.env.result is not None:
            self.game_over = True
            self.winner = _RESULT_WINNER.get(self.env.result, "draw")

    def apply_human_move(self, move: str) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if not self._human_to_move():
            return self.snapshot(), "Not your turn"
        move = (move or "").strip().lower()
        if move not in self.env.legal_moves():
            return self.snapshot(), "Move is not legal (Pikafish)"
        self.env.step(move)
        self.last_ally_move = move
        self._finish_if_over()
        return self.snapshot(), None

    def apply_engine_move(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if self.game_over:
            return self.snapshot(), "Game is already over"
        if self._human_to_move():
            return self.snapshot(), "Not engine turn"
        if self.player is None:
            return self.snapshot(), "MuZero engine not loaded"
        self.engine_thinking = True
        try:
            move = self.player.choose_move(self.env)
            if move not in self.env.legal_moves():
                return self.snapshot(), f"Engine produced illegal move {move!r}"
            self.env.step(move)
            self.last_engine_move = move
            self._finish_if_over()
        finally:
            self.engine_thinking = False
        return self.snapshot(), None

    def apply_greedy_ally(self) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.snapshot(), "Greedy ally is not supported with the MuZero engine"
```

(The file ends at `apply_greedy_ally` — do not add methods beyond the GameSession duck-type interface.)

- [ ] **Step 3.4: Run tests**

Run: `uv run pytest web/tests -v` → 8 passed. `uv run ruff check web --fix && uv run ruff format web`. Also re-run `uv run pytest muzero/tests -q` (68 passed, 6 skipped — untouched).

- [ ] **Step 3.5: Commit**

```bash
git add web/server/muzero_session.py web/tests/test_muzero_session.py
git commit -m "feat(web): MuZeroGameSession — either-color play on muzero env"
```

---

### Task 4: Server wiring — engine selection + humanSide

**Files:**
- Modify: `web/server/app.py` (lifespan ~lines 40-64, `NewGameRequest` ~lines 33-38, `new_game` ~lines 99-102)
- Modify: `scripts/serve_xiangqi_play.py`

- [ ] **Step 4.1: Modify `web/server/app.py`.**

Replace the `NewGameRequest` class with:

```python
class NewGameRequest(BaseModel):
    allyMode: Literal["human", "greedy"] = Field(
        "human",
        description="LLM mode only. human: click to move; greedy: capture-greedy ally",
    )
    humanSide: Literal["red", "black"] = Field(
        "red", description="MuZero mode only: which color the human plays"
    )
```

Replace the `lifespan` body's engine-loading section (keep `build_pikafish()` first) with:

```python
    global _session, _engine
    pikafish = build_pikafish()
    engine_kind = os.environ.get("XIANGQI_PLAY_ENGINE", "llm").strip().lower()
    if engine_kind == "muzero":
        from web.server.muzero_player import MuZeroPlayer
        from web.server.muzero_session import MuZeroGameSession

        ckpt = os.environ.get(
            "XIANGQI_MUZERO_CKPT",
            str(_REPO_ROOT / "checkpoints/muzero_xiangqi/latest.pt"),
        )
        device = os.environ.get("XIANGQI_PLAY_DEVICE", "cpu")
        print(f"[xiangqi-play] Loading MuZero ckpt={ckpt} device={device}", flush=True)
        player = MuZeroPlayer(ckpt, device=device)
        _session = MuZeroGameSession(pikafish=pikafish, player=player)
        print("[xiangqi-play] MuZero engine ready.", flush=True)
    else:
        skip_engine = os.environ.get(
            "XIANGQI_PLAY_SKIP_ENGINE", ""
        ).strip().lower() in {"1", "true", "yes"}
        if not skip_engine:
            adapter = os.environ.get(
                "XIANGQI_PLAY_ADAPTER",
                str(_REPO_ROOT / "checkpoints/xiangqi_grpo_v2/ep_40"),
            )
            device = os.environ.get("XIANGQI_PLAY_DEVICE", "cuda")
            print(
                f"[xiangqi-play] Loading engine adapter={adapter} device={device}",
                flush=True,
            )
            _engine = EnginePlayer(adapter_path=adapter, device=device)
            print("[xiangqi-play] Engine ready.", flush=True)
        else:
            print(
                "[xiangqi-play] Engine skipped (XIANGQI_PLAY_SKIP_ENGINE).",
                flush=True,
            )
        _session = GameSession(pikafish=pikafish, engine=_engine)
    yield
```

Also move the module-level `from web.server.engine_player import EnginePlayer` import INSIDE the `else:` branch (as `from web.server.engine_player import EnginePlayer`) so muzero mode never imports the 7B stack. Replace the `new_game` endpoint body with:

```python
@app.post("/api/game/new")
async def new_game(body: NewGameRequest | None = None) -> Dict[str, Any]:
    sess = _require_session()
    if getattr(sess, "engine_kind", "llm") == "muzero":
        return sess.reset(human_side=body.humanSide if body else "red")
    return sess.reset(ally_mode=body.allyMode if body else "human")
```

- [ ] **Step 4.2: Modify `scripts/serve_xiangqi_play.py`** — add after the `--skip-engine` argument:

```python
    ap.add_argument(
        "--engine",
        choices=["llm", "muzero"],
        default="llm",
        help="Opponent: llm (7B LoRA, default) or muzero (canonical checkpoint)",
    )
    ap.add_argument(
        "--ckpt",
        default=os.path.join(_ROOT, "checkpoints/muzero_xiangqi/latest.pt"),
        help="MuZero checkpoint path (muzero mode only)",
    )
```

Change the `--device` argument's `default="cuda"` to `default=None` with help `"cuda/cpu; default: cuda for llm, cpu for muzero"`, and replace the env-var block with:

```python
    os.environ["XIANGQI_PLAY_ADAPTER"] = args.adapter
    os.environ["XIANGQI_PLAY_ENGINE"] = args.engine
    os.environ["XIANGQI_MUZERO_CKPT"] = args.ckpt
    os.environ["XIANGQI_PLAY_DEVICE"] = args.device or (
        "cpu" if args.engine == "muzero" else "cuda"
    )
    if args.skip_engine:
        os.environ["XIANGQI_PLAY_SKIP_ENGINE"] = "1"
```

- [ ] **Step 4.3: Verify**

```bash
uv run ruff check web scripts --fix && uv run ruff format web scripts
uv run pytest web/tests muzero/tests -q
uv run python scripts/serve_xiangqi_play.py --help
```

Expected: ruff clean (ignore the pre-existing `scripts/claude_plays.py` failures if ruff is run repo-wide — scope to `web scripts/serve_xiangqi_play.py` if needed); tests pass; help text shows `--engine {llm,muzero}` and `--ckpt`. (Full server startup needs PIKAFISH_BIN + fastapi — that's the Task 6 manual smoke on the box.)

- [ ] **Step 4.4: Commit**

```bash
git add web/server/app.py scripts/serve_xiangqi_play.py
git commit -m "feat(web): engine selection (llm|muzero) + humanSide in new-game API"
```

---

### Task 5: UI — color picker + human-side-aware board

**Files:**
- Modify: `web/static/index.html` (controls block, lines 12-24; title line 6)
- Modify: `web/static/board.js` (status line ~205, selectable ~292, click filter ~478, newGame ~505-525, init ~534-550)

- [ ] **Step 5.1: `index.html`** — change `<title>` to `Xiangqi — Play` and the subtitle line to `<p class="subtitle" id="subtitle">Loading…</p>`. Wrap the existing ally-mode controls and add the side picker, replacing the current `.controls` div content with:

```html
    <div class="controls">
      <span id="ally-controls">
        <span>Red ally:</span>
        <label class="mode-label">
          <input type="radio" name="ally-mode" value="human" checked />
          Human (click moves)
        </label>
        <label class="mode-label">
          <input type="radio" name="ally-mode" value="greedy" />
          Greedy agent (ε=0)
        </label>
      </span>
      <span id="side-controls" class="hidden">
        <span>You play:</span>
        <label class="mode-label">
          <input type="radio" name="human-side" value="red" checked />
          Red
        </label>
        <label class="mode-label">
          <input type="radio" name="human-side" value="black" />
          Black
        </label>
      </span>
      <button type="button" id="btn-new-game" class="btn">New game</button>
    </div>
```

- [ ] **Step 5.2: `board.js`** — five edits:

(a) Add helpers after the `allyMode()` function (~line 68):

```js
function humanSide() {
  return state?.humanSide || "red";
}

function engineKind() {
  return state?.engineKind || "llm";
}

function isHumanPiece(ch, isRed) {
  if (ch === ".") return false;
  return humanSide() === "red" ? isRed : !isRed;
}

function selectedHumanSide() {
  const el = document.querySelector('input[name="human-side"]:checked');
  return el ? el.value : "red";
}

function applyEngineKindUI() {
  const kind = engineKind();
  const sideCtl = document.getElementById("side-controls");
  const allyCtl = document.getElementById("ally-controls");
  if (sideCtl) sideCtl.classList.toggle("hidden", kind !== "muzero");
  if (allyCtl) allyCtl.classList.toggle("hidden", kind === "muzero");
  const sub = document.getElementById("subtitle");
  if (sub) {
    sub.textContent =
      kind === "muzero"
        ? "Human vs MuZero (canonical net, full-strength search)"
        : "Red (ally) vs Black (ep_40 engine)";
  }
}
```

(b) In `updateStatus()` replace the `state.turn === "human"` line's message:

```js
  if (state.turn === "human") {
    statusEl.textContent =
      humanSide() === "red"
        ? "Your turn — click a Red piece"
        : "Your turn — click a Black piece";
  }
```

(c) In `renderBoard()` replace `if (humanTurn && isRed) btn.classList.add("selectable");` with:

```js
      if (humanTurn && isHumanPiece(ch, isRed)) btn.classList.add("selectable");
```

(d) In `onIntersectionClick()` replace `if (!isRed || ch === ".") {` with:

```js
  if (!isHumanPiece(ch, isRed)) {
```

(e) In `newGame()`: include the side in the POST body and auto-trigger the model's first move when the human takes Black; replace the body from `const mode = ...` to the end of the function with:

```js
  const mode = selectedAllyMode();
  const side = selectedHumanSide();
  state = await postWithRetry(
    "/api/game/new",
    { allyMode: mode, humanSide: side },
    { gen, timeoutMs: 30000 }
  );
  applyEngineKindUI();
  refresh();

  if (!state.gameOver && state.turn === "engine" && isActive(gen)) {
    await runEngineMove(gen); // human plays Black: model (Red) opens
  }
  if (mode === "greedy" && allyMode() === "greedy" && !state.gameOver && isActive(gen)) {
    await runGreedyGameLoop(gen, { firstPlies: true });
  }
```

And in `init()`, add `applyEngineKindUI();` immediately after the first `await syncState();`.

- [ ] **Step 5.3: Verify syntax** (node may not be installed; skip gracefully):

```bash
node --check web/static/board.js 2>/dev/null && echo "js ok" || echo "node unavailable — reviewed by eye"
```

- [ ] **Step 5.4: Commit**

```bash
git add web/static/index.html web/static/board.js
git commit -m "feat(web): color picker + human-side-aware board for muzero mode"
```

---

### Task 6: Docs, merge, and runbooks

**Files:**
- Modify: `web/README.md`, `docs/ARCHITECTURE.md` (§3e), `docs/AGENT_TODO.md`
- Create: `docs/logs/2026-07-06-log-muzero-web-play.md` (per `docs/logs/template.md`)

- [ ] **Step 6.1: `web/README.md`** — retitle to cover both engines and add a MuZero section:

```markdown
## Play vs MuZero (either color, CPU-friendly)

```bash
export PIKAFISH_BIN=/path/to/pikafish
uv run --group web python scripts/serve_xiangqi_play.py --engine muzero
```

- Uses `checkpoints/muzero_xiangqi/latest.pt` by default (`--ckpt` to override);
  runs on CPU by default so training can keep going. Old pre-canonicalization
  (115-plane) checkpoints are rejected at startup.
- Pick Red or Black next to "New game" — as Black, the model moves first.
- Full training strength (800 simulations): expect ~5–20 s of thinking per
  move on a laptop CPU. Startup is seconds (no 7B load).
- On the Mac: install a macOS Pikafish build, set `PIKAFISH_BIN`, and copy the
  checkpoint:
  `scp <box>:~/Documents/FineTune-LLM-OnlineRL/checkpoints/muzero_xiangqi/latest.pt checkpoints/muzero_xiangqi/`
```

- [ ] **Step 6.2: Docs housekeeping** — `docs/ARCHITECTURE.md` §3e: note the web UI now serves two engines (LLM Black-only; MuZero either color via `MuZeroGameSession` on muzero's env, gate-strength search, `--engine muzero`). `docs/AGENT_TODO.md`: move the "human-vs-MuZero web play adapter" backlog item to Completed with a pointer to the spec/plan/log. Write `docs/logs/2026-07-06-log-muzero-web-play.md` per the template (goal, files, test counts, manual-smoke status, next steps).

- [ ] **Step 6.3: Full verification + merge**

```bash
uv run ruff check web scripts/serve_xiangqi_play.py muzero --fix && uv run ruff format web scripts/serve_xiangqi_play.py muzero
uv run pytest web/tests muzero/tests -q
git add docs web/README.md
git commit -m "docs(web): muzero play runbook + architecture/TODO/log updates"
git checkout main && git merge --no-ff muzero-web-play -m "merge: human-vs-muzero web play adapter" && git push
```

- [ ] **Step 6.4: Manual smoke on the 5090 box (user or agent-with-access runs this; record results in the log):**

```bash
git pull
PIKAFISH_BIN=<path> uv run --group web python scripts/serve_xiangqi_play.py --engine muzero
# open http://127.0.0.1:8765 — play a few moves as Red; New game as Black
# (model should open automatically); confirm think time is acceptable on CPU.
```

---

## Self-Review (completed)

- **Spec coverage:** §3a player → Task 2; §3b session (incl. truncation-off, draw mapping, snapshot contract + engineKind) → Tasks 1 & 3; §3c wiring + serve flags → Task 4; §3d UI → Task 5; §4 error handling → Tasks 2-4 (startup errors in player ctor, 400-path via existing `_play_error`); §5 testing → Tasks 2-3 (6 session tests + 3 player tests, FakeEvaluator only, engine-gated CI not extended); §6 runbook → Task 6.
- **Placeholder scan:** clean — no TBDs, every code step shows complete code.
- **Type consistency:** `MuZeroPlayer(ckpt_path, device, num_simulations, config)` matches usage in tests and app.py; `reset(human_side=...)` vs `reset(ally_mode=...)` dispatch keyed on `engine_kind` attr present on both sessions (LLM session gets `engineKind` in snapshot only — app.py dispatch uses `getattr(sess, "engine_kind", "llm")`, and `GameSession` has no such attr → correctly falls to the LLM branch).
- **Heavy-import trap:** muzero modules never import `game_session`/`engine_player` (Task 1 extraction); app.py's `EnginePlayer` import moved inside the LLM branch.
