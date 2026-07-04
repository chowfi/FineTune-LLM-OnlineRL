# MuZero-Style Latest-Weights Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stock MuZero self-play (latest network plays both sides) the default, with the existing frozen-enemy promotion scheme preserved behind `self_play_mode="frozen_enemy"`, per `docs/superpowers/specs/2026-07-03-muzero-latest-selfplay-design.md`.

**Architecture:** Runner-level switch. In latest mode no enemy net exists (`enemy_runner is ally_runner`), the worker's per-round MCTS groups merge into one batched call with Dirichlet noise at every root, truncation becomes symmetric (per-color saturation streaks in the env), and the coordinator's promotion is disabled. Frozen mode stays byte-identical to current behavior.

**Tech Stack:** Existing `muzero/` package (PyTorch, numpy, pytest via `uv run`).

**Conventions:** repo root `"/Users/fionachow/Documents/NYU/CDS/Spring 2024/DS-GA 3001.005 - Reinforcement Learning/Projects"` (quote the path). Branch: work directly on `main` unless the user says otherwise (matches how post-merge fixes have been landing). After each task: `uv run ruff check muzero --fix && uv run ruff format muzero`. Engine-gated tests skip without `PIKAFISH_BIN`. Baseline suite before Task 1: **45 passed, 5 skipped**.

**Cross-cutting caution:** the default mode FLIPS to `"latest"` in Task 1, which changes behavior of code/tests written for the frozen scheme. Tasks 2–3 update the affected tests to pin each mode explicitly. Expect the suite to be green after every task's final step — Task 1 includes the two one-line test opt-ins needed to keep it green.

---

### Task 1: Config — `self_play_mode` + derived `truncation_symmetric`

**Files:**
- Modify: `muzero/config.py` (field near `promote_after_consecutive_wins`; `__post_init__`)
- Modify: `muzero/tests/test_config.py` (append)
- Modify: `muzero/tests/test_selfplay.py` (one-line opt-in in `test_coordinator_promotes_after_streak`)
- Modify: `muzero/tests/test_env_adjudication.py` (opt-in in `test_hopeless_ally_game_truncates_as_loss`)

- [ ] **Step 1: Write the failing tests (append to `muzero/tests/test_config.py`)**

```python
import pytest


def test_self_play_mode_defaults_and_derivation():
    cfg = MuZeroConfig()
    assert cfg.self_play_mode == "latest"
    assert cfg.truncation_symmetric is True
    frozen = MuZeroConfig(self_play_mode="frozen_enemy")
    assert frozen.truncation_symmetric is False


def test_self_play_mode_validation():
    with pytest.raises(ValueError):
        MuZeroConfig(self_play_mode="bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_config.py -v`
Expected: 2 new tests FAIL (`TypeError: unexpected keyword argument 'self_play_mode'` or `AttributeError`).

- [ ] **Step 3: Implement in `muzero/config.py`**

Add the field in the Self-play section (right above `promote_after_consecutive_wins`):

```python
    # "latest": stock MuZero — the newest network plays both sides, no enemy.
    # "frozen_enemy": learner vs frozen snapshot with streak promotion.
    self_play_mode: str = "latest"
    truncation_symmetric: bool = True  # derived from self_play_mode in __post_init__
```

Extend `__post_init__` (keep the existing `input_planes` derivation):

```python
    def __post_init__(self):
        self.input_planes = 14 * self.history_length + 3
        if self.self_play_mode not in ("latest", "frozen_enemy"):
            raise ValueError(f"self_play_mode: {self.self_play_mode!r}")
        self.truncation_symmetric = self.self_play_mode == "latest"
```

- [ ] **Step 4: Keep the two mode-sensitive existing tests pinned to frozen mode**

In `muzero/tests/test_selfplay.py`, `test_coordinator_promotes_after_streak`, change the config line to:

```python
    cfg = replace(
        MuZeroConfig(),
        channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        device="cpu",
        self_play_mode="frozen_enemy",
    )
```

In `muzero/tests/test_env_adjudication.py`, `test_hopeless_ally_game_truncates_as_loss`, change the config line to:

```python
    cfg = replace(MuZeroConfig(), truncation_consecutive=3, self_play_mode="frozen_enemy")
```

(`dataclasses.replace` re-runs `__post_init__`, so `truncation_symmetric` re-derives to False — this test now explicitly pins the asymmetric behavior. Note: promotion is not disabled until Task 3; this step just makes intent explicit and future-proof.)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest muzero/tests -v`
Expected: **47 passed, 5 skipped** (45 baseline + 2 new; nothing else breaks — env/selfplay code doesn't read the new fields yet).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
git add muzero
git commit -m "feat(muzero): self_play_mode config flag with derived truncation_symmetric"
```

---

### Task 2: Env — symmetric truncation (per-color saturation streaks)

**Files:**
- Modify: `muzero/env.py` (`reset`, `step` truncation branch, `_check_truncation`)
- Modify: `muzero/tests/test_env_adjudication.py` (append 2 tests)

- [ ] **Step 1: Write the failing tests (append to `muzero/tests/test_env_adjudication.py`)**

```python
def test_symmetric_truncation_fires_on_non_ally_side():
    # Default mode is "latest" -> symmetric truncation. Script the engine so
    # ONLY black is hopeless: evaluate_cp is side-to-move perspective, so
    # +900 with white to move means red is +900 (black just left itself lost),
    # and -900 with black to move means red is +900 again (red is fine).
    cfg = replace(MuZeroConfig(), truncation_consecutive=3)
    assert cfg.truncation_symmetric
    env = XiangqiEnv(
        cfg, FakeEvaluator(cp_fn=lambda fen: 900.0 if " w " in fen else -900.0)
    )
    env.reset(ally_side="w")  # the LOSER (black) is NOT the ally
    done = False
    plies = 0
    while not done:
        _, reward, done, info = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "red_win"  # black, the saturated side, loses
    assert env.truncated and info["truncated"]
    assert plies == 6  # black's 3rd saturated move (plies 2, 4, 6)
    assert reward < 0  # final mover is black, the loser: -1 + shaping


def test_asymmetric_mode_ignores_non_ally_saturation():
    # frozen_enemy mode: only the ally's streak counts. Make RED hopeless
    # while the ally is BLACK -> no truncation; the shuffle ends in the
    # 3-fold repetition draw at ply 8 instead.
    cfg = replace(
        MuZeroConfig(), truncation_consecutive=3, self_play_mode="frozen_enemy"
    )
    assert not cfg.truncation_symmetric
    env = XiangqiEnv(cfg, FakeEvaluator(cp_fn=lambda fen: 900.0))
    env.reset(ally_side="b")
    done = False
    plies = 0
    while not done:
        _, _, done, _ = env.step(SHUFFLE[plies % 4])
        plies += 1
    assert env.result == "draw_repetition"
    assert not env.truncated
    assert plies == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_env_adjudication.py -v`
Expected: `test_symmetric_truncation_fires_on_non_ally_side` FAILS (result is `draw_repetition` at ply 8, because current code ignores non-ally saturation). The asymmetric test may already pass — that's fine.

- [ ] **Step 3: Implement in `muzero/env.py`**

In `reset`, replace `self._sat_streak = 0` with:

```python
        self._sat_streaks = {"w": 0, "b": 0}
```

Replace `_check_truncation` entirely:

```python
    def _check_truncation(self, mover: str, cp_after_red) -> bool:
        # Symmetric mode (latest-weights self-play): either color's hopeless
        # streak truncates. Asymmetric mode (frozen_enemy): ally-only, so the
        # winning side still practices converting won positions.
        if not self.config.truncation_symmetric and mover != self.ally_side:
            return False
        mover_cp = self._mover_cp(mover, cp_after_red)
        if mover_cp is None:
            return False
        if mover_cp <= self.config.truncation_cp:
            self._sat_streaks[mover] += 1
        else:
            self._sat_streaks[mover] = 0
        return self._sat_streaks[mover] >= self.config.truncation_consecutive
```

In `step`, change the truncation branch's result line (the loser is always the mover — the streak only ever completes on the saturated side's own move):

```python
        elif self._check_truncation(mover, cp_after_red):
            self.result = "black_win" if mover == "w" else "red_win"
            self.truncated = True
            info["truncated"] = True
            reward += -1.0  # the mover is always the saturated, losing side
```

(In asymmetric mode the trigger only fires when `mover == ally_side`, so `"black_win" if mover == "w"` is identical to the old `"black_win" if self.ally_side == "w"` — frozen-mode behavior is unchanged.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest muzero/tests -v`
Expected: **49 passed, 5 skipped**. Watch specifically: the pre-existing `test_hopeless_ally_game_truncates_as_loss` (now frozen-mode) and `test_threefold_repetition_is_draw_with_penalty` must still pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
git add muzero
git commit -m "feat(muzero): symmetric hopeless-truncation for latest-mode self-play"
```

---

### Task 3: Self-play — single-group latest mode + promotion switch + all-move diagnostics

**Files:**
- Modify: `muzero/selfplay.py` (`SelfPlayCoordinator.__init__`/`report_result`, `_Game`, `SelfPlayWorker._record_and_step`/`_finish`/`generate`)
- Modify: `muzero/tests/test_selfplay.py` (append 2 tests)

- [ ] **Step 1: Write the failing tests (append to `muzero/tests/test_selfplay.py`)**

```python
def test_coordinator_promotion_disabled_in_latest_mode():
    cfg = replace(MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu")
    assert cfg.self_play_mode == "latest"
    torch.manual_seed(0)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    before = [p.detach().clone() for p in enemy.parameters()]
    coord = SelfPlayCoordinator(cfg, ally, enemy)
    for _ in range(5):
        promoted = coord.report_result(ally_won=True, draw=False)
        assert promoted is False
    assert coord.era == 0
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, enemy.parameters()))


def test_round_groups_by_mode():
    """Latest mode: one spec covering all games with noise; frozen mode: the
    original ally/enemy two-spec split (noise on ally roots only)."""

    def make_worker(mode):
        cfg = replace(
            MuZeroConfig(),
            channels=16,
            repr_blocks=1,
            dyn_blocks=1,
            device="cpu",
            self_play_mode=mode,
        )
        torch.manual_seed(0)
        net = MuZeroNet(cfg)
        ally_runner, enemy_runner = NetRunner(net, "cpu"), NetRunner(net, "cpu")
        coord = SelfPlayCoordinator(cfg, net, net)
        buf = ReplayBuffer(cfg)
        return SelfPlayWorker(
            cfg, ally_runner, enemy_runner, buf, coord, object(), worker_id=0
        )

    latest = make_worker("latest")
    assert latest._round_groups([]) == [(latest.ally_runner, None, True)]

    frozen = make_worker("frozen_enemy")
    assert frozen._round_groups([]) == [
        (frozen.ally_runner, True, True),
        (frozen.enemy_runner, False, False),
    ]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_selfplay.py -v`
Expected: first new test FAILS (promotion fires — coordinator has no mode switch); second FAILS (`AttributeError: ... has no attribute '_round_groups'`).

- [ ] **Step 3: Implement in `muzero/selfplay.py`**

`SelfPlayCoordinator.__init__` — add after `self.config = config`:

```python
        self.promotion_enabled = config.self_play_mode == "frozen_enemy"
```

`report_result` — gate the promotion block:

```python
    def report_result(self, ally_won: bool, draw: bool) -> bool:
        with self.lock:
            self.games_this_era += 1
            self.streak = self.streak + 1 if (ally_won and not draw) else 0
            if (
                self.promotion_enabled
                and self.streak >= self.config.promote_after_consecutive_wins
            ):
                with self.enemy_lock, torch.no_grad():
                    self.enemy_net.load_state_dict(self.ally_net.state_dict())
                self.enemy_net.eval()
                self.streak = 0
                self.era += 1
                self.games_this_era = 0
                return True
        return False
```

`SelfPlayWorker.__init__` — add after `self.cfg = config`:

```python
        self.latest_mode = config.self_play_mode == "latest"
```

Rename `_Game.ally_value_cp_pairs` usage: keep the attribute names (metrics continuity) but track an extra list. In `_Game.__init__` add:

```python
        self.ally_cps = []  # tracked-color cp after its own moves (both modes)
```

Replace the diagnostics in `_record_and_step` (the entropy block before `env.step` and the cp block after it):

```python
        mover = game.env.side_to_move  # about to move
        if self.latest_mode or mover == game.env.ally_side:
            p = np.array([v / total for v in visits.values()], dtype=np.float64)
            game.ally_entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
        _, reward, done, info = game.env.step(index_to_move(action))
        h.rewards.append(reward)
        if info.get("red_cp") is not None:
            mover_cp = info["red_cp"] if mover == "w" else -info["red_cp"]
            if self.latest_mode or mover == game.env.ally_side:
                # root_value is mover-perspective; pair it with mover-persp cp
                game.ally_value_cp_pairs.append((float(root_value), float(mover_cp)))
            if mover == game.env.ally_side:
                ally_cp = (
                    info["red_cp"]
                    if game.env.ally_side == "w"
                    else -info["red_cp"]
                )
                game.ally_cps.append(float(ally_cp))
        return done, info
```

In `_finish`, change `mean_ally_cp` to use the tracked-color list (identical values to today in frozen mode):

```python
            "mean_ally_cp": (
                float(np.mean(game.ally_cps)) if game.ally_cps else None
            ),
```

Add the round-grouping helper (returns lazy *specs*, not game lists — group membership is computed inside the round loop from live `active`, exactly matching the old inline loop's semantics where the enemy group was formed after ally steps flipped `side_to_move`):

```python
    def _round_groups(self, active: list) -> list:
        """Specs for this round's MCTS calls: (runner, side_filter, noise).

        side_filter None means "all active games" (latest mode)."""
        if self.latest_mode:
            return [(self.ally_runner, None, True)]
        return [(self.ally_runner, True, True), (self.enemy_runner, False, False)]
```

```python
        while active:
            for runner, want_ally, add_noise in self._round_groups(active):
                group = [
                    g
                    for g in active
                    if want_ally is None
                    or (g.env.side_to_move == g.env.ally_side) == want_ally
                ]
                if not group:
                    continue
                roots = []
                for g in group:
                    legal = np.array(
                        [move_to_index(m) for m in g.env.legal_moves()], dtype=np.int64
                    )
                    roots.append((g.env.observation().astype(np.float32), legal))
                results = self.mcts.run(runner, roots, add_noise=add_noise)
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
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest muzero/tests -v`
Expected: **51 passed, 5 skipped** (49 + 2 new). The engine-gated smoke test still constructs separate ally/enemy runners — in latest mode the worker simply never uses the enemy one; unchanged test stays valid.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
git add muzero
git commit -m "feat(muzero): latest-mode self-play — single batched group, noise on all roots, promotion switch"
```

---

### Task 4: Metrics — red/black win rates

**Files:**
- Modify: `muzero/metrics.py` (`aggregate_game_summaries`)
- Modify: `muzero/tests/test_metrics.py` (extend existing assertions)

- [ ] **Step 1: Write the failing test (append to `muzero/tests/test_metrics.py`, reusing its existing `summaries` fixture style)**

```python
def test_red_black_win_rates():
    summaries = [
        {"result": "red_win", "ally_side": "w", "ally_won": True, "draw": False,
         "plies": 40, "truncated": False, "promoted": False, "final_red_cp": 0.0, "era": 0},
        {"result": "black_win", "ally_side": "w", "ally_won": False, "draw": False,
         "plies": 40, "truncated": False, "promoted": False, "final_red_cp": 0.0, "era": 0},
        {"result": "draw_repetition", "ally_side": "b", "ally_won": False, "draw": True,
         "plies": 40, "truncated": False, "promoted": False, "final_red_cp": 0.0, "era": 0},
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/red_win_rate"] == 1 / 3
    assert m["selfplay/black_win_rate"] == 1 / 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_metrics.py -v` — new test FAILS with `KeyError`.

- [ ] **Step 3: Implement — add to the returned dict in `aggregate_game_summaries`**

```python
        "selfplay/red_win_rate": sum(1 for s in summaries if s["result"] == "red_win")
        / n,
        "selfplay/black_win_rate": sum(
            1 for s in summaries if s["result"] == "black_win"
        )
        / n,
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest muzero/tests -v` — expected **52 passed, 5 skipped**.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
git add muzero
git commit -m "feat(muzero): red/black win-rate metrics"
```

---

### Task 5: Training loop — wiring, checkpoint save/load per mode

**Files:**
- Modify: `muzero/train.py` (new module-level `load_checkpoint`; `main()` net/runner/coordinator wiring and checkpoint save)
- Modify: `muzero/tests/test_train.py` (append 1 test)

- [ ] **Step 1: Write the failing test (append to `muzero/tests/test_train.py`)**

```python
def test_load_checkpoint_without_enemy_key(tmp_path):
    cfg = replace(
        MuZeroConfig(), channels=16, repr_blocks=1, dyn_blocks=1, device="cpu"
    )
    torch.manual_seed(0)
    source = MuZeroNet(cfg)
    trainer_src = MuZeroTrainer(cfg, source)
    path = tmp_path / "latest.pt"
    torch.save(
        {
            "ally": source.state_dict(),
            "optimizer": trainer_src.optimizer.state_dict(),
            "iteration": 7,
            "era": 0,
            "streak": 1,
        },
        path,
    )  # note: no "enemy" key, as latest-mode checkpoints are written

    torch.manual_seed(1)
    ally, enemy = MuZeroNet(cfg), MuZeroNet(cfg)
    trainer = MuZeroTrainer(cfg, ally)
    from muzero.train import load_checkpoint

    ckpt = load_checkpoint(str(path), ally, enemy, trainer.optimizer, "cpu")
    assert ckpt["iteration"] == 7
    for pa, pe in zip(ally.parameters(), enemy.parameters()):
        assert torch.equal(pa, pe)  # enemy fell back to ally weights
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest muzero/tests/test_train.py -v` — FAILS (`ImportError: cannot import name 'load_checkpoint'`).

- [ ] **Step 3: Implement in `muzero/train.py`**

Add module-level (above `main`, after `run_gate`):

```python
def load_checkpoint(path: str, ally, enemy, optimizer, device) -> dict:
    """Resume helper. Latest-mode checkpoints carry no "enemy" entry; fall
    back to the ally weights so a checkpoint from either mode loads in
    either mode."""
    ckpt = torch.load(path, map_location=device)
    ally.load_state_dict(ckpt["ally"])
    if enemy is not ally:
        enemy.load_state_dict(ckpt.get("enemy", ckpt["ally"]))
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
```

In `main()`, replace the net-construction block:

```python
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    latest_mode = cfg.self_play_mode == "latest"
    ally = MuZeroNet(cfg).to(device)
    # latest mode: no separate enemy net — both sides share the ally.
    enemy = ally if latest_mode else copy.deepcopy(ally).to(device)
    if not latest_mode:
        enemy.eval()
    trainer = MuZeroTrainer(cfg, ally)
    start_iteration = 0
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume, ally, enemy, trainer.optimizer, device)
        start_iteration = ckpt["iteration"]
```

Replace the runner/coordinator block:

```python
    buffer = ReplayBuffer(cfg)
    ally_runner = NetRunner(ally, device)
    enemy_runner = ally_runner if latest_mode else NetRunner(enemy, device)
    # Runners are created before the coordinator so the coordinator can be
    # handed the enemy runner's lock: promotion must not swap weights
    # mid-forward-pass (promotion is disabled entirely in latest mode).
    coordinator = SelfPlayCoordinator(cfg, ally, enemy, enemy_lock=enemy_runner.lock)
```

Replace the checkpoint-save block at the end of the loop:

```python
        ckpt_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
        tmp_path = ckpt_path + ".tmp"
        ckpt_data = {
            "ally": ally.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "iteration": it + 1,
            "era": coordinator.era,
            "streak": coordinator.streak,
        }
        if not latest_mode:
            ckpt_data["enemy"] = enemy.state_dict()
        torch.save(ckpt_data, tmp_path)
        os.replace(tmp_path, ckpt_path)
```

(`copy` stays imported — still used in frozen mode.)

- [ ] **Step 4: Run the full suite + entrypoint sanity**

```bash
uv run pytest muzero/tests -v            # expect 53 passed, 5 skipped
uv run python -c "import muzero.train"
uv run python -m muzero.train --help
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
git add muzero
git commit -m "feat(muzero): latest-mode wiring — shared runner, no enemy net, mode-aware checkpoints"
```

---

### Task 6: Docs handoff + final verification

**Files:**
- Modify: `docs/ARCHITECTURE.md` §3f
- Modify: `docs/AGENT_TODO.md`
- Create: `docs/logs/2026-07-03-log-latest-selfplay-mode.md` (from `docs/logs/template.md`)

- [ ] **Step 1: Update `docs/ARCHITECTURE.md` §3f** — replace the sentence describing frozen-enemy self-play with:

```markdown
  Self-play defaults to stock MuZero (`self_play_mode="latest"`): the newest
  network plays both sides, Dirichlet noise at every root, one batched MCTS
  group per lockstep round, symmetric hopeless-truncation, no enemy net. The
  original frozen-enemy scheme (promotion after 3 consecutive ally wins,
  asymmetric ally-only truncation) remains available as a research ablation
  via `self_play_mode="frozen_enemy"`. Spec:
  `docs/superpowers/specs/2026-07-03-muzero-latest-selfplay-design.md`.
```

- [ ] **Step 2: Update `docs/AGENT_TODO.md`** — in the Active "first run" task, note that runs after this change use latest mode by default and the frozen-enemy run's checkpoint resumes cleanly into either mode; add a Completed entry for this change referencing the spec, plan, and log.

- [ ] **Step 3: Write the dated log** following `docs/logs/template.md`: goal, per-task summary, final pytest line, note that the in-flight frozen-enemy run should be restarted (or resumed with `--resume`, which now skips warm start) to pick up latest mode.

- [ ] **Step 4: Final verification + commit**

```bash
uv run ruff check muzero --fix && uv run ruff format muzero
uv run pytest muzero/tests -v      # expect 53 passed, 5 skipped
git add docs muzero
git commit -m "docs: latest-weights self-play mode — architecture, TODO, log"
```

---

## Self-Review Notes (already applied)

- **Spec coverage:** §3 config → Task 1; §4 env → Task 2; §5 selfplay/coordinator → Task 3; §7 metrics → Task 4; §6 train wiring + checkpoints → Task 5; §8 tests distributed across Tasks 1–5; §9 rollout → Task 6 docs.
- **Frozen-mode preservation verified by construction:** `_round_groups` returns the same two specs the old inline loop iterated (group membership recomputed inside the round loop from live `active`, exactly as before); truncation result string proven equivalent when trigger requires `mover == ally_side`; `value_cp_pairs` mover-perspective values coincide with the old ally-perspective values when only ally moves are recorded.
- **Type consistency:** `_round_groups` returns `(runner, want_ally|None, add_noise)` spec triples — Task 3's `generate`, its test, and the docstring all use that one shape.
- **Placeholder scan:** clean; every code step carries the actual code.
