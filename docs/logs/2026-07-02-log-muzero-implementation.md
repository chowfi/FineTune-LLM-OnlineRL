# MuZero for Xiangqi — Implementation (Tasks 1–14)

**Date:** 2026-07-02
**Agent/Author:** Claude Code (Fable 5), subagent-driven development

## 1. Hypothesis / Goal
Implement the approved MuZero/EfficientZero-style Xiangqi agent per
`docs/superpowers/specs/2026-07-02-muzero-xiangqi-design.md`, following
`docs/superpowers/plans/2026-07-02-muzero-xiangqi.md` task-by-task with a
fresh implementer subagent per task and two-stage review (spec compliance,
then code quality) after each.

## 2. Configuration Changes
- New `muzero/` package: `config.py`, `encoding.py`, `transforms.py`,
  `network.py`, `env.py`, `mcts.py`, `selfplay.py`, `replay_buffer.py`,
  `warmstart.py`, `train.py`, `metrics.py`, plus `muzero/tests/` (39 tests).
- `pyproject.toml`/`uv.lock`: `pytest` added as dev dependency.
- `.gitignore`: `__pycache__/`, `*.pyc` added.
- Branch: `muzero-xiangqi` (28 commits from `main`), user chose feature branch
  in place over a worktree.

## 3. Run Command
```bash
uv run pytest muzero/tests -v        # 34 passed, 5 skipped (engine-gated)
uv run python -m muzero.train --help # entrypoint sanity
```
End-to-end smoke (`--smoke --no-wandb --iterations 1 --device cpu`) **not run**:
`PIKAFISH_BIN` is unset on this machine. Deferred to the training machine —
tracked in `docs/AGENT_TODO.md` (Active).

## 4. Quantitative Results
- **Tests:** 34 passed, 5 skipped (skips = Pikafish-gated: env reset/step,
  self-play smoke, warmstart ×2). Ruff clean (scoped to `muzero/`).
- **Network:** 22.10M params at default config (192ch, 12+8 blocks) after
  channel-reducing the SimSiam projector (was 36.8M with an 18.75M projector).
- **Value support fix (review-driven, Critical):** n-step returns live in
  ≈[−2,2]; the spec's original [−300,300] support left ~1.5 of 601 bins active
  and amplified stray softmax mass ~300×. Rescaled to [−3,3] h-units →
  ~147 active bins; near-init decoded-value noise dropped from std ≈0.9–2.2 to
  ≈0.007–0.014.
- **Encoding cost (measured):** ~140 ms of pure-Python encoding per 512-sample
  training batch (make_target: 2× encode_observation + 9× material_balance per
  sample). Profiling watch-item, not fixed.

## 5. Qualitative Outcome
Review loop caught real defects at nearly every stage; the notable ones:
- Plan bug: gym_xiangqi `move_to_action_space` needs **unsigned** piece ids —
  the plan's signed version broke every Black move (fixed in `env.py`).
- BatchNorm mode: net shared between trainer (`.train()`) and MCTS inference —
  `NetRunner` now forces `.eval()` under its lock; SimSiam target branch in
  `train_batch` toggles eval too (running stats stay clean).
- MCTS `_backup` fed raw simulation values (and a spurious root sample) into
  `MinMaxStats`; now tracks exactly the child mean-Q that `_select_child`
  normalizes. Dirichlet noise now uses a seeded, per-worker RNG.
- Concurrency: enemy-promotion weight swap now takes the enemy `NetRunner`
  lock; `ReplayBuffer.add`'s paired appends are locked (regression test with
  4 threads × 20 adds).
- Resume path: single `torch.load`, `streak` persisted alongside `era`,
  atomic checkpoint writes (`.tmp` + `os.replace`), warmstart skipped on
  resume.
- Known accepted gaps: gym_xiangqi perpetual-check edge case; engine-abort
  counted as draw in gate; AdamW decays BN/bias params — all in the TODO
  backlog.

## 6. Repo / Handoff Updates
- `docs/ARCHITECTURE.md`: repo map entry + new §3f component section.
- `docs/AGENT_TODO.md`: implementation task closed; "first run on training
  machine" active task added; follow-ups + repo-hygiene backlog items added.
- Related: `docs/logs/2026-07-02-log-muzero-xiangqi-design.md` (design
  session), spec + plan under `docs/superpowers/` (plan amended 5× during
  execution — NetRunner eval, MCTS rng, coordinator enemy_lock, value_max).

## 7. Conclusion & Next Steps
Implementation complete and reviewed on branch `muzero-xiangqi`; not yet
merged to `main`. Next agent/user: run the engine-gated tests + smoke on the
5090 machine with `PIKAFISH_BIN` set, then start the real training run and log
findings (see Active task in `docs/AGENT_TODO.md`). Merge decision pending the
final whole-branch review.
