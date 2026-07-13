# Node-Limited Pikafish Gate Rung Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the saturated uniform-random gate rung with a Pikafish rung limited to `cfg.gate_pika_nodes` search nodes — a graded, extensible strength dial between the greedy rung and full Pikafish.

**Architecture:** One new config value (`gate_pika_nodes=128`), a `nodes` mode on the existing `SimpleUciEngine` UCI wrapper (`go nodes N` instead of `go movetime M`), and a rung swap inside `run_gate` that reuses the existing `_run_gate_rung` machinery with a second engine instance. Measurement-only: no training rules change.

**Tech Stack:** Python 3.12, pytest, `uv` for all commands. Spec: `docs/superpowers/specs/2026-07-13-node-limited-pika-rung-design.md`.

**Conventions (read first):**
- Run every command through `uv run ...` from the repo root.
- Engine-gated tests use the existing helpers: `from muzero.tests.helpers import PIKAFISH_BIN, requires_engine` — they skip automatically when `PIKAFISH_BIN` is not set. Without a local Pikafish binary those tests SKIP; that is expected and fine.
- Before finishing: `uv run ruff check muzero --fix && uv run ruff format muzero` (scope to `muzero/` — repo-wide ruff fails on a pre-existing `scripts/claude_plays.py` issue that is out of scope).
- Never stage `__pycache__` / `.pyc` files (some are tracked legacy noise; `git add` specific file paths only).

---

### Task 1: Config value `gate_pika_nodes`

**Files:**
- Modify: `muzero/config.py` (the "Fixed-opponent gate" block, around line 111)
- Test: `muzero/tests/test_config.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `muzero/tests/test_config.py`:

```python
def test_gate_pika_nodes_default():
    assert MuZeroConfig().gate_pika_nodes == 128
```

(If the file's existing imports lack `MuZeroConfig`, match the file's existing import style — it already tests `MuZeroConfig`, so the import exists.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_config.py::test_gate_pika_nodes_default -q`
Expected: FAIL with `AttributeError: 'MuZeroConfig' object has no attribute 'gate_pika_nodes'`

- [ ] **Step 3: Add the config field**

In `muzero/config.py`, the `# Fixed-opponent gate` block currently reads:

```python
    # Fixed-opponent gate
    gate_every_loops: int = 10
    gate_games: int = 20
    gate_movetime_ms: int = 10
```

Change it to:

```python
    # Fixed-opponent gate
    gate_every_loops: int = 10
    gate_games: int = 20
    gate_movetime_ms: int = 10
    # 2026-07-13: graded mid-rung — Pikafish limited to this many search
    # nodes per move (the binary has no UCI_Elo/Skill Level; `go nodes N`
    # needs no engine option and each halving is a roughly constant strength
    # step). Bump protocol: if gate/win_rate_pika_nodes > ~0.85 for 3+
    # consecutive gates, DOUBLE this and note it in a dated log (each change
    # redefines the metric — never silent); if < ~0.15 for 3+ gates, halve.
    gate_pika_nodes: int = 128
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_config.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add muzero/config.py muzero/tests/test_config.py
git commit -m "feat(muzero): gate_pika_nodes config for node-limited gate rung"
```

---

### Task 2: `nodes` mode on `SimpleUciEngine`

**Files:**
- Modify: `muzero/warmstart.py:21-53` (`SimpleUciEngine`)
- Test: `muzero/tests/test_warmstart.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `muzero/tests/test_warmstart.py`:

```python
def test_go_command_modes():
    """nodes=None keeps movetime search; nodes=N switches to node-limited.
    Constructed via __new__ so no engine process is spawned."""
    eng = SimpleUciEngine.__new__(SimpleUciEngine)
    eng.movetime_ms = 10
    eng.nodes = None
    assert eng._go_command() == "go movetime 10"
    eng.nodes = 128
    assert eng._go_command() == "go nodes 128"


@requires_engine
def test_node_limited_search_returns_move():
    eng = SimpleUciEngine(PIKAFISH_BIN, movetime_ms=10, multipv=1, nodes=8)
    try:
        lines = eng.search(START_FEN)
    finally:
        eng.close()
    assert lines and len(lines[0][0]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_warmstart.py::test_go_command_modes -q`
Expected: FAIL with `AttributeError: 'SimpleUciEngine' object has no attribute '_go_command'`

- [ ] **Step 3: Implement the nodes mode**

In `muzero/warmstart.py`, change the constructor signature and store the new field:

```python
    def __init__(
        self, binary_path: str, movetime_ms: int, multipv: int, nodes: int | None = None
    ):
        self.movetime_ms = movetime_ms
        self.nodes = nodes
```

(rest of `__init__` unchanged). Add the command builder and use it in `search()`:

```python
    def _go_command(self) -> str:
        if self.nodes is not None:
            return f"go nodes {self.nodes}"
        return f"go movetime {self.movetime_ms}"
```

In `search()`, replace:

```python
        self._cmd(f"go movetime {self.movetime_ms}")
```

with:

```python
        self._cmd(self._go_command())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_warmstart.py -q`
Expected: `test_go_command_modes` PASS; engine-gated tests PASS with `PIKAFISH_BIN` set, otherwise SKIP.

- [ ] **Step 5: Commit**

```bash
git add muzero/warmstart.py muzero/tests/test_warmstart.py
git commit -m "feat(muzero): node-limited search mode on SimpleUciEngine"
```

---

### Task 3: Rung swap in `run_gate`

**Files:**
- Modify: `muzero/train.py:190-243` (`run_gate`)
- Test: `muzero/tests/test_gate_opponents.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `muzero/tests/test_gate_opponents.py`:

```python
def test_run_gate_reports_pika_nodes_rung(monkeypatch):
    """run_gate constructs the weak engine with nodes=cfg.gate_pika_nodes,
    closes both engines, emits the new metric keys, and no random-rung
    keys. Engines are faked at muzero.warmstart (run_gate's import site)."""
    import torch
    from dataclasses import replace

    import muzero.warmstart as warmstart
    from muzero.config import MuZeroConfig
    from muzero.mcts import NetRunner
    from muzero.network import MuZeroNet
    from muzero.tests.helpers import FakeEvaluator
    from muzero.train import run_gate

    instances = []

    class FakeEngine:
        def __init__(self, binary_path, movetime_ms, multipv, nodes=None):
            self.nodes = nodes
            self.closed = False
            instances.append(self)

        def search(self, fen):
            stm = fen.split()[1]
            # ENGINE-UCI (bottom-origin ranks): engine_uci_to_algebraic
            # maps "a6a5" -> "a3a4" and "i3i4" -> "i6i5", the FakeEvaluator's
            # legal algebraic moves below.
            return [("a6a5" if stm == "w" else "i3i4", 0.0)]

        def close(self):
            self.closed = True

    monkeypatch.setattr(warmstart, "SimpleUciEngine", FakeEngine)

    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        num_simulations=4,
        interior_topk=4,
        gate_games=2,
        max_game_plies=2,
        gate_pika_nodes=128,
        device="cpu",
    )
    torch.manual_seed(0)
    runner = NetRunner(MuZeroNet(cfg), "cpu")

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    metrics = run_gate(cfg, runner, evaluator)

    # full-strength engine first (nodes=None), then the weak rung's engine
    assert [e.nodes for e in instances] == [None, 128]
    assert all(e.closed for e in instances)
    for key in (
        "gate/win_rate_pika_nodes",
        "gate/draw_rate_pika_nodes",
        "gate/loss_rate_pika_nodes",
        "gate/win_rate_greedy",
        "gate/win_rate",
        "gate/seconds",
    ):
        assert key in metrics
    assert metrics["gate/pika_nodes"] == 128.0
    assert not any(k.endswith("_random") for k in metrics)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_gate_opponents.py::test_run_gate_reports_pika_nodes_rung -q`
Expected: FAIL — `run_gate` still emits `gate/win_rate_random` (the `not any(k.endswith("_random"))` assertion, or the missing `gate/win_rate_pika_nodes` key, depending on assertion order).

- [ ] **Step 3: Replace `run_gate`**

In `muzero/train.py`, replace the entire `run_gate` function (currently lines ~190–243) with:

```python
def run_gate(cfg: MuZeroConfig, runner, evaluator) -> dict:
    """Gate ladder: capture-greedy heuristic mover, then Pikafish limited to
    cfg.gate_pika_nodes search nodes (graded mid-rung; bump protocol
    documented in config.py), then raw Pikafish at gate movetime. The
    uniform-random rung was retired 2026-07-13 after ~300 iterations pinned
    at 1.0 (spec: docs/superpowers/specs/
    2026-07-13-node-limited-pika-rung-design.md); its gate/*_random history
    remains in wandb."""
    import time

    from muzero.gate_opponents import greedy_capture_move
    from muzero.warmstart import SimpleUciEngine
    from src.xiangqi_board import engine_uci_to_algebraic

    t0 = time.monotonic()
    # seed+1 preserved from the retired-random-rung era so the greedy rung's
    # historical move sequence is unchanged.
    greedy_rng = np.random.default_rng(cfg.seed + 1)

    def greedy_move(env):
        return greedy_capture_move(env, greedy_rng)

    engine = SimpleUciEngine(cfg.pikafish_bin, cfg.gate_movetime_ms, multipv=1)
    try:
        weak_engine = SimpleUciEngine(
            cfg.pikafish_bin,
            cfg.gate_movetime_ms,
            multipv=1,
            nodes=cfg.gate_pika_nodes,
        )
    except Exception:
        engine.close()
        raise

    def engine_mover(eng):
        def move(env):
            lines = eng.search(env.fen())
            if not lines:
                return None
            return engine_uci_to_algebraic(lines[0][0])

        return move

    n = cfg.gate_games
    try:
        greedy_wins, greedy_draws = _run_gate_rung(
            cfg, runner, evaluator, greedy_move
        )
        weak_wins, weak_draws = _run_gate_rung(
            cfg, runner, evaluator, engine_mover(weak_engine)
        )
        pika_wins, pika_draws = _run_gate_rung(
            cfg, runner, evaluator, engine_mover(engine)
        )
    finally:
        engine.close()
        weak_engine.close()
    return {
        "gate/win_rate_greedy": greedy_wins / n,
        "gate/draw_rate_greedy": greedy_draws / n,
        "gate/loss_rate_greedy": (n - greedy_wins - greedy_draws) / n,
        "gate/win_rate_pika_nodes": weak_wins / n,
        "gate/draw_rate_pika_nodes": weak_draws / n,
        "gate/loss_rate_pika_nodes": (n - weak_wins - weak_draws) / n,
        "gate/pika_nodes": float(cfg.gate_pika_nodes),
        "gate/win_rate": pika_wins / n,
        "gate/draw_rate": pika_draws / n,
        "gate/loss_rate": (n - pika_wins - pika_draws) / n,
        "gate/seconds": time.monotonic() - t0,
    }
```

Notes for the implementer:
- The old function built a `rng = np.random.default_rng(cfg.seed)` and a `random_move` mover — both are deleted with the random rung.
- Rung order in the returned dict and in execution: greedy, pika_nodes, full Pikafish.
- `engine` (full strength) is constructed FIRST — the wiring test asserts `[None, 128]` construction order.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest muzero/tests/test_gate_opponents.py -q`
Expected: all PASS (the pre-existing greedy rung test must still pass — it uses `_run_gate_rung` directly, which this task does not touch).

- [ ] **Step 5: Run the full muzero suite**

Run: `uv run pytest muzero/tests -q`
Expected: all PASS (engine-gated tests SKIP without `PIKAFISH_BIN`). If anything else references `gate/win_rate_random` (grep to confirm: `grep -rn "win_rate_random" muzero/ --include="*.py"`), fix it in this task — as of plan-writing, `run_gate` is the only Python producer/consumer.

- [ ] **Step 6: Commit**

```bash
git add muzero/train.py muzero/tests/test_gate_opponents.py
git commit -m "feat(muzero): node-limited Pikafish rung replaces saturated random rung"
```

---

### Task 4: Lint, docs, and handoff accounting

**Files:**
- Modify: `docs/ARCHITECTURE.md` (gate/rung description, §3e or wherever `run_gate`'s ladder is described — grep for "random" / "greedy" / "gate")
- Modify: `docs/AGENT_TODO.md` (Active Tasks: add rollout task)
- Create: `docs/logs/2026-07-13-log-node-limited-pika-rung.md` (use `docs/logs/template.md`)

- [ ] **Step 1: Lint**

Run: `uv run ruff check muzero --fix && uv run ruff format muzero`
Expected: "All checks passed!" and 0-or-more files reformatted; re-run `uv run pytest muzero/tests -q` if anything reformatted.

- [ ] **Step 2: Update ARCHITECTURE.md**

Find the gate ladder description (`grep -n "random" docs/ARCHITECTURE.md`) and update it to describe the three rungs as: capture-greedy → Pikafish@`gate_pika_nodes` (graded dial, bump protocol in config) → full Pikafish@`gate_movetime_ms`. Mention `gate/*_random` was retired 2026-07-13.

- [ ] **Step 3: Write the dated log**

Create `docs/logs/2026-07-13-log-node-limited-pika-rung.md` following `docs/logs/template.md`, covering: hypothesis (dead rung + cliff → graded dial), config change (`gate_pika_nodes=128`, bump protocol), expected first readings (30–70% band is the target; adjust dial per protocol), rollout checklist from the spec §4, and the note that this is measurement-only (safe mid-experiment; activates on next training restart).

- [ ] **Step 4: Update AGENT_TODO.md**

Add to Active Tasks:

```markdown
- `[ ]` (2026-07-13) **MuZero: node-limited pika rung — restart to activate.** Rung ladder now greedy → pika@128-nodes → full pika (spec `docs/superpowers/specs/2026-07-13-node-limited-pika-rung-design.md`). On the box: `git pull && uv run pytest muzero/tests -q`, stop training, `uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt`. Watch the first 2–3 `gate/win_rate_pika_nodes` readings; apply the double/halve protocol (config comment) if outside 30–70%. Measurement-only — does not disturb experiment #2's banked state.
```

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/AGENT_TODO.md docs/logs/2026-07-13-log-node-limited-pika-rung.md
git commit -m "docs: node-limited pika rung — architecture, log, rollout task"
```

---

## Self-Review (done at plan-writing)

- Spec §2a → Task 1; §2b → Task 2; §2c → Task 3; §2d/§4 → Task 4 (log + TODO). Testing section → Tasks 1–3 test steps. No gaps.
- No placeholders; every code step shows the code.
- Signature consistency checked: `SimpleUciEngine(binary_path, movetime_ms, multipv, nodes=None)` matches between Task 2 implementation, Task 3 call sites, and both fakes/tests. `_go_command` defined in Task 2, used only there. Metric keys in Task 3's implementation match its test's assertions.
- ENGINE-UCI ↔ algebraic conversion verified against `src/xiangqi_board.py:108-118`: `"a6a5"` → `"a3a4"`, `"i3i4"` → `"i6i5"`.
