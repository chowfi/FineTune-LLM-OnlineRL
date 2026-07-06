# Human-vs-MuZero Web Play Adapter

**Date:** 2026-07-06
**Agent/Author:** Claude Code (subagent-driven development; coordinator + per-task implementer/spec-reviewer/quality-reviewer subagents)

## 1. Hypothesis / Goal

Let the user play against the canonical MuZero checkpoint through the existing
`web/` board UI, as either color, at full gate strength, CPU-first (playable on
the Mac and on the 5090 box without stopping training). Spec:
`docs/superpowers/specs/2026-07-06-muzero-web-play-design.md`; plan:
`docs/superpowers/plans/2026-07-06-muzero-web-play.md` (Approach A: parallel
MuZero session behind the same UI; LLM path untouched).

## 2. Configuration Changes

Branch `muzero-web-play`, 11 commits (3e9f394…7abb97f) + docs:

- `web/server/board_view.py` (new) — light board-grid rendering shared by both
  sessions (no 7B import chain).
- `web/server/muzero_player.py` (new) — loads `ckpt["ally"]` (rejects
  pre-canonicalization 115-plane checkpoints with an actionable error),
  `choose_move(env)` = `canonical_root` → 800-sim noiseless MCTS →
  `absolute_visits` → argmax (identical to the training gate). Defensive
  config copy; empty-legal guard; concurrency note.
- `web/server/muzero_session.py` (new) — either-color game session on
  `muzero.env.XiangqiEnv`; hopeless-cp adjudication disabled
  (`truncation_consecutive=10**9`, copied config); repetition draws + 300-ply
  cap kept; snapshot contract mirrors `GameSession` + `engineKind`/`humanSide`;
  engine exceptions surface as 400-contract errors, session stays usable.
- `web/server/pikafish_setup.py` (new) — `build_pikafish()` moved out of
  `game_session.py` so muzero-mode startup never imports
  transformers/unsloth/peft (verified via sys.modules check).
- `web/server/app.py` — lifespan dispatches on `XIANGQI_PLAY_ENGINE`
  (llm default unchanged; muzero: `XIANGQI_MUZERO_CKPT` default
  `checkpoints/muzero_xiangqi/latest.pt`, device default cpu); heavy imports
  lazy in the LLM branch; `new_game` routes `humanSide`/`allyMode` by
  `engine_kind`; concurrency-invariant comment.
- `scripts/serve_xiangqi_play.py` — `--engine {llm,muzero}`, `--ckpt`,
  device default cpu-for-muzero/cuda-for-llm.
- `web/static/` — color picker (`#side-controls`, muzero mode only),
  human-side-aware piece selection/status, auto engine-first-move when human
  takes Black, generic `.hidden` CSS rule, side-picker sync on page load,
  gen-guarded engine responses.
- Tests: `web/tests/` (new) — 12 tests (player + session) on FakeEvaluator +
  tiny nets; no GPU/engine needed.

## 3. Run Command

```bash
uv run pytest web/tests muzero/tests -q   # 80 passed, 6 skipped
uv run ruff check web scripts/serve_xiangqi_play.py
node --check web/static/board.js
```

## 4. Quantitative Results

- 12 web tests + 68 muzero tests pass; ruff clean; JS syntax-checked.
- Review catches worth recording: missing generic `.hidden` CSS rule (plan
  gap — controls would not have toggled), heavy-import chain via
  `game_session` (resolved with `pikafish_setup.py`), config-mutation
  footguns in player and session (both now `dataclasses.replace`),
  engine exceptions escaping as unretryable 500s, `engineThinking` ordering
  in error snapshots, side-picker desync after reload mid-Black-game.

## 5. Qualitative Outcome

- Manual smoke NOT yet run (needs Pikafish + a checkpoint): see §7.
- Known deferred items are in `docs/AGENT_TODO.md` (app.py endpoint tests;
  Pikafish-outage false-mate env behavior; illegal-engine-move retry cost).

## 6. Repo / Handoff Updates

- `docs/ARCHITECTURE.md` §3e rewritten for the two-engine web app.
- `docs/AGENT_TODO.md` — web-play adapter moved to Completed; follow-ups
  added under Backlog.
- `web/README.md` — MuZero runbook (box + Mac, checkpoint scp, think-time
  expectations).

## 7. Conclusion & Next Steps

Manual smoke on the 5090 box (or Mac with Pikafish + checkpoint):

```bash
git pull
PIKAFISH_BIN=<path> uv run --group web python scripts/serve_xiangqi_play.py --engine muzero
# open http://127.0.0.1:8765 — play a few moves as Red; New game as Black
# (model opens automatically); confirm CPU think time is acceptable.
```

Record smoke results in this log afterward. Training continues unaffected
(muzero web play is CPU-only by default).
