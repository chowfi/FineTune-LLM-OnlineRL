# Engine-Game Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every training iteration, add a small number (default 4, ~5%) of Pikafish-vs-Pikafish games to the replay buffer as expert demonstrations, with temperature-style sampling among the engine's top moves for game variety.

**Architecture:** Generalize the existing warmstart game loop (`muzero/warmstart.py`) into a shared `play_engine_game` helper with a pure `_pick_move_index` sampling rule; add `generate_seed_games` beside `generate_warmstart_games`; wire a never-raises `seed_engine_games` helper into the training loop that logs `buffer/seeded_games`. Seeded games flow through the identical env/target/buffer pipeline, so `selfplay/*` metrics stay pure automatically.

**Tech Stack:** Python 3.12, pytest, `uv`. Spec: `docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md`.

**Conventions (read first):**
- Every command through `uv run ...` from the repo root.
- Engine-gated tests use `from muzero.tests.helpers import PIKAFISH_BIN, requires_engine` and SKIP without a local binary — expected, fine.
- Lint scoped to `muzero/` only: `uv run ruff check muzero --fix && uv run ruff format muzero` (repo-wide ruff fails on pre-existing out-of-scope issues).
- NEVER stage `__pycache__` / `.pyc` / `.venv` paths — `git add` explicit file paths only.
- Baseline before this plan: `uv run pytest muzero/tests -q` → 89 passed, 7 skipped.

---

### Task 1: Config value `seed_games_per_loop`

**Files:**
- Modify: `muzero/config.py` (the `# Self-play` block, after `max_game_plies`)
- Test: `muzero/tests/test_config.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `muzero/tests/test_config.py`:

```python
def test_seed_games_per_loop_default():
    assert MuZeroConfig().seed_games_per_loop == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_config.py::test_seed_games_per_loop_default -q`
Expected: FAIL with `AttributeError: 'MuZeroConfig' object has no attribute 'seed_games_per_loop'`

- [ ] **Step 3: Add the config field**

In `muzero/config.py`, inside the `# Self-play` block, directly after the line `max_game_plies: int = 300`, insert:

```python
    # 2026-07-14 experiment #3: engine-game seeding. Every iteration, this
    # many Pikafish-vs-Pikafish games (warmstart strength: 50 ms/move,
    # MultiPV 4; first temperature_moves plies sampled from the MultiPV
    # softmax for variety) are added to the buffer as expert demonstrations
    # (~5% of an 84-game loop). Targets the self-play echo chamber:
    # refutation-blindness + endgame technique against resistance (spec:
    # docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md).
    # 0 disables = the REVERT setting. Revert if loss/value or
    # value_cp_correlation degrade sustained (~15+ iters beyond churn) or
    # either engine-gate band falls below its pre-change band for 3+ gates.
    seed_games_per_loop: int = 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_config.py -q`
Expected: all PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add muzero/config.py muzero/tests/test_config.py
git commit -m "feat(muzero): seed_games_per_loop config for engine-game seeding"
```

---

### Task 2: Shared `play_engine_game` helper with temperature sampling

**Files:**
- Modify: `muzero/warmstart.py` (extract helper from `generate_warmstart_games`; add `_multipv_probs`, `_pick_move_index`)
- Test: `muzero/tests/test_warmstart.py` (append)

**Context for this task:** `generate_warmstart_games` currently contains the whole game loop inline and always plays the engine's best move (`lines[0]`). `_play_move` already computes a softmax over MultiPV centipawn scores for the POLICY TARGET — the same distribution the new sampling uses to pick the PLAYED move during the first `cfg.temperature_moves` plies. The policy target stays the full MultiPV distribution regardless of which move is played.

- [ ] **Step 1: Write the failing tests**

Append to `muzero/tests/test_warmstart.py`:

```python
def test_pick_move_index_samples_early_and_plays_best_late():
    from muzero.warmstart import _pick_move_index

    lines = [("a6a5", 0.0), ("b6b5", -30.0)]
    rng = np.random.default_rng(0)
    early_picks = {_pick_move_index(lines, ply=0, temperature_moves=30, rng=rng)
                   for _ in range(200)}
    assert early_picks == {0, 1}  # both engine choices actually get played
    counts = [0, 0]
    rng = np.random.default_rng(1)
    for _ in range(400):
        counts[_pick_move_index(lines, 0, 30, rng)] += 1
    assert counts[0] > counts[1]  # better-scored move favored
    # at/after temperature_moves: always the best line
    assert all(
        _pick_move_index(lines, ply, 30, np.random.default_rng(i)) == 0
        for i, ply in enumerate((30, 31, 100))
    )
    # a single candidate is always index 0, any ply
    assert _pick_move_index([("a6a5", 0.0)], 0, 30, np.random.default_rng(2)) == 0


def test_play_engine_game_produces_buffer_ready_history():
    from dataclasses import replace

    from muzero.config import MuZeroConfig
    from muzero.tests.helpers import FakeEvaluator
    from muzero.warmstart import play_engine_game

    class ScriptedEngine:
        def search(self, fen):
            stm = fen.split()[1]
            # ENGINE-UCI; converts to the legal algebraic moves below
            return [("a6a5", 0.0)] if stm == "w" else [("i3i4", 0.0)]

    cfg = replace(
        MuZeroConfig(),
        max_game_plies=2,
        temperature_moves=0,  # deterministic: always best line
        opening_book=("a6a5",),  # -> algebraic "a3a4", legal for white below
    )

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    history = play_engine_game(
        cfg, ScriptedEngine(), evaluator, np.random.default_rng(0)
    )
    assert len(history) == 2  # opening ply + one engine ply, then ply cap
    assert len(history.rewards) == 2
    assert len(history.policy_indices) == 2
    assert history.result is not None
    buf = ReplayBuffer(cfg)
    buf.add(history)  # buffer-compatible record
    assert len(buf.games) == 1
```

(`np`, `ReplayBuffer`, `MuZeroConfig`, and `replace` availability: the file already imports `replace`, `MuZeroConfig`, `ReplayBuffer`, and `SimpleUciEngine` at module level; it does NOT import numpy — add `import numpy as np` to the module imports in this step, keeping import order ruff-clean.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_warmstart.py -q`
Expected: the two new tests FAIL with `ImportError: cannot import name '_pick_move_index'` (and `play_engine_game`); pre-existing tests still pass/skip.

- [ ] **Step 3: Implement the refactor**

In `muzero/warmstart.py`:

(a) Add the two helpers after the `_INFO_RE` definition:

```python
def _multipv_probs(lines) -> np.ndarray:
    """Softmax over MultiPV centipawn scores (temperature 200 cp)."""
    cps = np.array([cp for _, cp in lines], dtype=np.float64)
    probs = np.exp((cps - cps.max()) / 200.0)
    return (probs / probs.sum()).astype(np.float32)


def _pick_move_index(lines, ply: int, temperature_moves: int, rng) -> int:
    """Index of the move to PLAY: sampled from the MultiPV softmax for the
    first temperature_moves plies (variety — a deterministic engine would
    otherwise replay one game per opening line forever), best line after."""
    if ply < temperature_moves and len(lines) > 1:
        return int(rng.choice(len(lines), p=_multipv_probs(lines)))
    return 0
```

(b) Add `play_engine_game` (the body is the inner loop currently inlined in `generate_warmstart_games`, with the sampling rule applied):

```python
def play_engine_game(cfg: MuZeroConfig, engine, evaluator, rng) -> GameHistory:
    """One engine-vs-engine game through the standard env pipeline (same
    referee, rewards, and targets as self-play games). Used by warmstart
    (cold start) and by per-iteration seeding (experiment #3)."""
    env = XiangqiEnv(cfg, evaluator)
    env.reset(ally_side="w")
    history = GameHistory()
    opening = cfg.opening_book[int(rng.integers(len(cfg.opening_book)))]
    done = _play_move(env, history, engine_uci_to_algebraic(opening), None)
    while not done:
        lines = engine.search(env.fen())
        if not lines:
            break
        pick = _pick_move_index(lines, len(history), cfg.temperature_moves, rng)
        move = engine_uci_to_algebraic(lines[pick][0])
        done = _play_move(env, history, move, lines)
    history.boards = [b.copy() for b in env.boards]
    history.to_play_history = list(env.to_play_history)
    history.rep_history = list(env.rep_history)
    history.no_progress_history = list(env.no_progress_history)
    history.truncated = env.truncated
    history.ally_side = env.ally_side
    history.result = env.result or "engine_aborted"
    return history
```

(c) Replace the body of `generate_warmstart_games` so it delegates:

```python
def generate_warmstart_games(
    cfg: MuZeroConfig, buffer: ReplayBuffer, evaluator
) -> dict:
    """Play engine-vs-engine games until >= cfg.warmstart_plies plies are stored."""
    engine = SimpleUciEngine(
        cfg.pikafish_bin, cfg.warmstart_movetime_ms, cfg.warmstart_multipv
    )
    rng = np.random.default_rng(cfg.seed)
    total_plies = games = 0
    try:
        while total_plies < cfg.warmstart_plies:
            history = play_engine_game(cfg, engine, evaluator, rng)
            buffer.add(history)
            total_plies += len(history)
            games += 1
            print(
                f"[warmstart] game {games}: {len(history)} plies ({history.result}) "
                f"— {total_plies}/{cfg.warmstart_plies} plies",
                flush=True,
            )
    finally:
        engine.close()
    return {"plies": total_plies, "games": games}
```

`_play_move` is unchanged. IMPORTANT: `_play_move` keeps computing the policy target from the FULL MultiPV lines (`probs` over all candidates) and `root_values` from `cps[0]` — the sampling only chooses which move is *played*; if you find yourself changing `_play_move`, stop (that would corrupt targets).

Note the one deliberate behavior change to warmstart itself: it now also samples early moves (the spec accepts this — variety is harmless-to-helpful there, and one code path beats two).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_warmstart.py -q` — new tests PASS, engine-gated tests SKIP locally.
Then the full suite: `uv run pytest muzero/tests -q` — expect 91 passed, 7 skipped.

- [ ] **Step 5: Commit**

```bash
git add muzero/warmstart.py muzero/tests/test_warmstart.py
git commit -m "refactor(muzero): shared play_engine_game with temperature sampling"
```

---

### Task 3: `generate_seed_games`

**Files:**
- Modify: `muzero/warmstart.py` (new function after `generate_warmstart_games`)
- Test: `muzero/tests/test_warmstart.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `muzero/tests/test_warmstart.py`:

```python
def _seed_test_fixture(monkeypatch, n_games):
    """Run generate_seed_games with a scripted engine; returns (stats, buffer,
    constructed) where constructed counts fake-engine instantiations."""
    from dataclasses import replace

    import muzero.warmstart as warmstart
    from muzero.config import MuZeroConfig
    from muzero.tests.helpers import FakeEvaluator
    from muzero.warmstart import generate_seed_games

    constructed = []

    class FakeUci:
        def __init__(self, binary_path, movetime_ms, multipv):
            constructed.append(self)
            self.closed = False

        def search(self, fen):
            stm = fen.split()[1]
            return [("a6a5", 0.0)] if stm == "w" else [("i3i4", 0.0)]

        def close(self):
            self.closed = True

    monkeypatch.setattr(warmstart, "SimpleUciEngine", FakeUci)
    cfg = replace(
        MuZeroConfig(),
        max_game_plies=2,
        temperature_moves=0,
        opening_book=("a6a5",),
        seed_games_per_loop=n_games,
    )

    def legal(fen):
        stm = fen.split()[1]
        return ["a3a4"] if stm == "w" else ["i6i5"]

    evaluator = FakeEvaluator(cp_fn=lambda fen: 0.0, legal_fn=legal)
    buf = ReplayBuffer(cfg)
    stats = generate_seed_games(cfg, buf, evaluator, n_games, np.random.default_rng(0))
    return stats, buf, constructed


def test_generate_seed_games_plays_n_and_fills_buffer(monkeypatch):
    stats, buf, constructed = _seed_test_fixture(monkeypatch, n_games=3)
    assert stats == {"games": 3, "plies": 6}  # 3 games x 2-ply cap
    assert len(buf.games) == 3
    assert len(constructed) == 1 and constructed[0].closed


def test_generate_seed_games_zero_is_a_noop(monkeypatch):
    stats, buf, constructed = _seed_test_fixture(monkeypatch, n_games=0)
    assert stats == {"games": 0, "plies": 0}
    assert len(buf.games) == 0
    assert constructed == []  # engine never constructed


@requires_engine
def test_generate_seed_games_real_engine_smoke():
    from dataclasses import replace

    from muzero.config import MuZeroConfig
    from muzero.tests.helpers import make_evaluator
    from muzero.warmstart import generate_seed_games

    cfg = replace(MuZeroConfig(), max_game_plies=6, warmstart_movetime_ms=20)
    buf = ReplayBuffer(cfg)
    stats = generate_seed_games(
        cfg, buf, make_evaluator(), 1, np.random.default_rng(0)
    )
    assert stats["games"] == 1 and stats["plies"] >= 1
    assert len(buf.games) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest muzero/tests/test_warmstart.py -q`
Expected: the two new non-gated tests FAIL with `ImportError: cannot import name 'generate_seed_games'`; smoke SKIPs locally.

- [ ] **Step 3: Implement**

Add to `muzero/warmstart.py` after `generate_warmstart_games`:

```python
def generate_seed_games(
    cfg: MuZeroConfig, buffer: ReplayBuffer, evaluator, n_games: int, rng
) -> dict:
    """Experiment #3 (2026-07-14): per-iteration expert-demonstration
    trickle — n_games engine-vs-engine games into the buffer. Unlike
    warmstart this runs every training loop, so the buffer permanently
    holds ~seed_games_per_loop/84 expert data instead of washing it out."""
    if n_games <= 0:
        return {"games": 0, "plies": 0}
    engine = SimpleUciEngine(
        cfg.pikafish_bin, cfg.warmstart_movetime_ms, cfg.warmstart_multipv
    )
    total_plies = games = 0
    try:
        for _ in range(n_games):
            history = play_engine_game(cfg, engine, evaluator, rng)
            buffer.add(history)
            total_plies += len(history)
            games += 1
    finally:
        engine.close()
    return {"games": games, "plies": total_plies}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_warmstart.py -q` — PASS (gated ones SKIP locally).
Full suite: `uv run pytest muzero/tests -q` — expect 93 passed, 8 skipped.

- [ ] **Step 5: Commit**

```bash
git add muzero/warmstart.py muzero/tests/test_warmstart.py
git commit -m "feat(muzero): generate_seed_games per-iteration expert trickle"
```

---

### Task 4: Training-loop wiring with outage fallback

**Files:**
- Modify: `muzero/train.py` (new module-level function + two lines in the loop)
- Test: `muzero/tests/test_train.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `muzero/tests/test_train.py`:

```python
def test_seed_engine_games_paths(monkeypatch):
    """Disabled -> 0.0 without calling the seeder; success -> game count;
    engine outage -> 0.0 and no exception."""
    import muzero.warmstart as warmstart
    from muzero.train import seed_engine_games

    calls = []

    def fake_seeder(cfg, buffer, evaluator, n_games, rng):
        calls.append(n_games)
        return {"games": n_games, "plies": n_games * 100}

    monkeypatch.setattr(warmstart, "generate_seed_games", fake_seeder)

    disabled = replace(MuZeroConfig(), seed_games_per_loop=0)
    assert seed_engine_games(disabled, None, None, iteration=7) == 0.0
    assert calls == []  # seeder never invoked when disabled

    enabled = replace(MuZeroConfig(), seed_games_per_loop=4)
    assert seed_engine_games(enabled, None, None, iteration=7) == 4.0
    assert calls == [4]

    def dying_seeder(cfg, buffer, evaluator, n_games, rng):
        raise RuntimeError("engine died")

    monkeypatch.setattr(warmstart, "generate_seed_games", dying_seeder)
    assert seed_engine_games(enabled, None, None, iteration=8) == 0.0  # no raise
```

(The file already imports `replace` and `MuZeroConfig` at module level.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest muzero/tests/test_train.py::test_seed_engine_games_paths -q`
Expected: FAIL with `ImportError: cannot import name 'seed_engine_games'`

- [ ] **Step 3: Implement**

(a) In `muzero/train.py`, add this module-level function directly after `run_gate`:

```python
def seed_engine_games(cfg: MuZeroConfig, buffer, evaluator, iteration: int) -> float:
    """Experiment #3 trickle (spec 2026-07-14). Never raises: an engine
    outage degrades the iteration to self-play-only with a warning. The
    per-iteration rng (cfg.seed + iteration) is reproducible without
    repeating the same sampled engine games every loop."""
    if cfg.seed_games_per_loop <= 0:
        return 0.0
    from muzero.warmstart import generate_seed_games

    try:
        stats = generate_seed_games(
            cfg,
            buffer,
            evaluator,
            cfg.seed_games_per_loop,
            np.random.default_rng(cfg.seed + iteration),
        )
        return float(stats["games"])
    except Exception as exc:
        print(f"[seed] engine seeding failed this iteration: {exc!r}", flush=True)
        return 0.0
```

(b) In `main()`'s training loop, directly after the line `metrics = aggregate_game_summaries(results)`, insert:

```python
        # -- seed: expert-demonstration trickle (experiment #3) --
        metrics["buffer/seeded_games"] = seed_engine_games(
            cfg, buffer, workers[0].evaluator, it
        )
```

(`workers[0].evaluator` is the same long-lived evaluator handle warmstart already uses at line ~378.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest muzero/tests/test_train.py -q` — all PASS.
Full suite: `uv run pytest muzero/tests -q` — expect 94 passed, 8 skipped.

- [ ] **Step 5: Commit**

```bash
git add muzero/train.py muzero/tests/test_train.py
git commit -m "feat(muzero): per-iteration engine-game seeding wired into training loop"
```

---

### Task 5: Lint, docs, and handoff accounting

**Files:**
- Modify: `docs/ARCHITECTURE.md` (self-play/training-loop description — grep for "warmstart" / "self-play" to find the section)
- Modify: `docs/AGENT_TODO.md` (Active Tasks)
- Create: `docs/logs/2026-07-14-log-engine-game-seeding.md` (use `docs/logs/template.md`; give it a real H1 title, NOT the template placeholder)

- [ ] **Step 1: Lint**

Run: `uv run ruff check muzero --fix && uv run ruff format muzero`
Expected: clean; if anything reformats, re-run `uv run pytest muzero/tests -q` (94 passed, 8 skipped).

- [ ] **Step 2: Update ARCHITECTURE.md**

In the training-loop / warmstart component descriptions: note that `muzero/warmstart.py` now exposes a shared `play_engine_game` (temperature-sampled engine-vs-engine games) used by both cold-start warmstart AND per-iteration seeding (`generate_seed_games`, `seed_games_per_loop=4`, experiment #3, spec `docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md`); the training loop logs `buffer/seeded_games`; `selfplay/*` metrics remain model-play-only because seeded games bypass the summary path. Surgical edits only.

- [ ] **Step 3: Write the dated log**

Create `docs/logs/2026-07-14-log-engine-game-seeding.md` per the template, titled `# Engine-Game Seeding — experiment #3`, covering: hypothesis (arena flat 380–480; user-observed refutation-blindness + endgame technique vs resistance; self-play echo chamber — see spec §1); the change (`seed_games_per_loop=4`, teacher = warmstart-strength engine, top-4 sampling for first 30 plies); expected mechanical shifts (loss/policy may drop — sharper targets; buffer/seeded_games≈4; ~25s/iter engine cost); pre-registered success criteria (both engine-gate bands trending up over 5+ gates; arena slope positive across ~500–560; fewer refuted-plan moments at the board) and revert criteria (config 0 + restart if loss/value or value_cp_correlation degrade sustained ~15+ iters, or either engine gate below pre-change band 3+ gates); gate-reading caveat (interpret gate/*_pika_nodes against the logged gate/pika_nodes dial — the rung may still be calibrating); rollout checklist (pull, test, stop, `--resume`, confirm `buffer/seeded_games=4` in the next log line); quantitative-results section left as "fill in as data arrives" with the pre-change baseline (greedy band ~0.45–0.80 gates 13–16; pika_nodes rung baseline TBD from its first readings; arena ~700–770 flat; mate rate ~0.20).

- [ ] **Step 4: Update AGENT_TODO.md**

Replace the current "decide experiment #3" active task's leading text: mark it resolved by appending "→ DECIDED 2026-07-14: engine-game seeding (built; see below)." at the end of that bullet, and add a NEW first bullet:

```markdown
- `[ ]` (2026-07-14) **MuZero: engine-game seeding (experiment #3) — restart to activate.** `seed_games_per_loop=4` engine games/iteration into the buffer (spec `docs/superpowers/specs/2026-07-14-engine-game-seeding-design.md`, log `docs/logs/2026-07-14-log-engine-game-seeding.md`). On the box: `git pull && uv run pytest muzero/tests -q`, stop training, `uv run python -m muzero.train --resume checkpoints/muzero_xiangqi/latest.pt`, confirm `buffer/seeded_games=4` in the next iteration line. Judge after 5+ gates + next two arena top-ups per the log's criteria; revert = `seed_games_per_loop: 0` + restart. Note: same restart also activates the node-limited pika rung if not already running.
```

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/AGENT_TODO.md docs/logs/2026-07-14-log-engine-game-seeding.md
git commit -m "docs: engine-game seeding — architecture, experiment log, rollout task"
```

---

## Self-Review (done at plan-writing)

- Spec §3a → Task 1; §2/§3b (sampling + shared helper) → Task 2; §3b (`generate_seed_games`) → Task 3; §3c (loop wiring, outage fallback, metric) → Task 4; §4/§6 (criteria, rollout) → Task 5 log; §5 testing → Tasks 2–4 test steps (sampling rule incl. distribution + late-ply cases; n_games/zero/no-engine; wiring paths incl. outage; warmstart regression = existing engine-gated test; engine-gated seed smoke).
- No placeholders; every code step shows the code.
- Signature consistency: `play_engine_game(cfg, engine, evaluator, rng)`, `_pick_move_index(lines, ply, temperature_moves, rng)`, `generate_seed_games(cfg, buffer, evaluator, n_games, rng)`, `seed_engine_games(cfg, buffer, evaluator, iteration)` — used identically across Tasks 2/3/4 and their tests. `FakeUci.__init__(binary_path, movetime_ms, multipv)` matches `generate_seed_games`'s construction (3 positional args, no `nodes`).
- ENGINE-UCI ↔ algebraic pairs reused from the verified mapping: `"a6a5"`→`"a3a4"`, `"i3i4"`→`"i6i5"`.
- Expected test counts: 89→91 (Task 2: +2) → 93 passed, 8 skipped (Task 3: +2 non-gated, +1 gated skip) → 94 (Task 4: +1).
