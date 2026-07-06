# Human-vs-MuZero Web Play Adapter — Design

**Date:** 2026-07-06
**Status:** Approved approach (A) from brainstorming session with user
**Related:** `docs/superpowers/specs/2026-07-05-muzero-color-canonicalization-design.md` (the canonical net this serves), `web/README.md` (existing LLM play UI)

## 1. Goal

Let the user play Xiangqi against the canonical MuZero checkpoint through the
existing `web/` board UI, as **either color**, at full training strength
(800 simulations, argmax, no exploration noise), running **CPU-first** so it
works both on the Mac and on the 5090 box without stopping training.

**User decisions (locked):** portable/CPU-by-default; always full strength (no
difficulty presets, no sims flag); board always drawn Red-at-bottom (no view
flip when playing Black).

**Non-goals:** touching the LLM play path; difficulty settings; board-flip
rendering; resign button; move takeback; multi-session support (the existing
app is single-session and stays that way).

## 2. Approach (A): parallel MuZero session behind the same UI

The existing `GameSession` is LLM-shaped (gym env, human=Red hardcoded, 7B
logprob scorer). Instead of bending it, add a sibling session built on
`muzero.env.XiangqiEnv`, which already maintains everything the MuZero encoder
needs (board history, repetition counts, no-progress counter) and already
adjudicates results. Model moves reuse the exact play path the training gate
uses — `selfplay.canonical_root` → `MCTS.run(add_noise=False)` →
`encoding.absolute_visits` → argmax — so web-play strength is literally gate
strength, with zero new frame-conversion code.

## 3. Components

### 3a. `web/server/muzero_player.py` — MuZeroPlayer

```python
class MuZeroPlayer:
    def __init__(self, ckpt_path: str, device: str = "cpu", num_simulations: int | None = None): ...
    def choose_move(self, env) -> str:  # absolute internal-algebraic move
```

- Builds a `MuZeroConfig` (overriding `device`; `num_simulations` stays the
  config default 800 unless the ctor overrides — ctor param exists ONLY so
  tests can run with 4 sims; no CLI/UI exposure).
- Constructs `MuZeroNet`, loads `ckpt["ally"]` (`map_location=device`),
  `eval()`, wraps in `NetRunner`; one `MCTS` instance.
- `choose_move(env)`: `obs, legal = canonical_root(env)`;
  `((visits, _, _),) = mcts.run(runner, [(obs, legal)], add_noise=False)`;
  `visits = absolute_visits(visits, env.side_to_move)`;
  return `index_to_move(max(visits, key=visits.get))`.
- Raises `FileNotFoundError`/`RuntimeError` with a clear message if the
  checkpoint is missing or has the wrong input-plane shape (old 115-plane
  checkpoints), naming the expected path.
- Expected think time at 800 sims, batch-1 CPU: roughly 5–20 s/move
  (laptop) — documented in README; acceptable per user decision.

### 3b. `web/server/muzero_session.py` — MuZeroGameSession

Owns one `muzero.env.XiangqiEnv` + one `MuZeroPlayer` + the shared
`PikafishEvaluator` (legality/eval source for the env, exactly as in
training). Mirrors the existing session's **snapshot JSON contract** so
`board.js` keeps working, with one addition (`humanSide`).

- **Play-time config overrides:** `truncation_consecutive = 10**9`
  (disables the hopeless-position auto-adjudication — the human gets to
  finish or attempt comebacks). Repetition draws and the 300-ply cap remain
  (both are real game-ending rules).
- `reset(human_side: "red"|"black")` → `env.reset(ally_side="w"|"b")`
  (ally = human; in latest-mode config the ally flag has no gameplay effect
  beyond bookkeeping). Snapshot `turn` = `"human"` when
  `env.side_to_move == human's side`, else `"engine"`; `"none"` when over.
- `snapshot()` keys (superset of the existing contract): `board` (grid via
  the same `_PIECE_FEN` letter mapping — `env.board` is the same signed-int8
  format as `env.state` in the gym session), `graphic`, `fen`, `humanSide`,
  `sideToMove` (`"ally"`/`"enemy"` mapped human/model for UI compatibility),
  `turn`, `gameOver`, `winner` (`"red"`/`"black"` from `env.result`;
  `"draw"` for `draw_repetition`/`draw_max_plies` — `board.js` already
  renders `"draw"`), `lastAllyMove` (= human's last move),
  `lastEngineMove` (= model's), `engineThinking`, `allyMode` (fixed
  `"human"`), and `engineKind: "muzero"` (the LLM session adds
  `engineKind: "llm"` to its snapshot — one-line change — so the UI can
  show/hide the color picker).
- `legal_targets_from(from_sq)`: filter `env.legal_moves()` by prefix —
  works for either color; only answers when it's the human's turn.
- `apply_human_move(move)`: guard turn/game-over; validate against
  `env.legal_moves()`; `env.step(move)`; update winner from
  `env.result`.
- `apply_engine_move()`: guard turn; set `engine_thinking` around
  `player.choose_move(env)` + `env.step(move)` (try/finally).
- No `apply_greedy_ally` — returns an "unsupported in MuZero mode" error.

### 3c. `web/server/app.py` — engine selection

- Lifespan reads `XIANGQI_PLAY_ENGINE` (`"llm"` default → existing behavior
  unchanged; `"muzero"` → build `MuZeroPlayer` +
  `MuZeroGameSession`). MuZero mode reads `XIANGQI_MUZERO_CKPT`
  (default `checkpoints/muzero_xiangqi/latest.pt`) and
  `XIANGQI_PLAY_DEVICE` (default `"cpu"` in muzero mode; the LLM default
  stays `"cuda"`).
- `NewGameRequest` gains optional `humanSide: "red"|"black" = "red"`
  (ignored by the LLM session; `allyMode` ignored by the MuZero session).
- Endpoints unchanged; both session types satisfy the same duck-typed
  interface (`reset`, `snapshot`, `legal_targets_from`, `apply_human_move`,
  `apply_engine_move`, `apply_greedy_ally`).
- `scripts/serve_xiangqi_play.py` gains `--engine {llm,muzero}` and
  `--ckpt PATH` flags that set the env vars.

### 3d. `web/static` — minimal UI changes

- `index.html`: a Red/Black radio next to "New game", visible only when the
  server reports muzero mode (snapshot `engineKind: "muzero"|"llm"` added for
  this); selection sent as `humanSide` in `/api/game/new`.
- `board.js`: (1) piece-click filter and status text keyed off snapshot
  `humanSide` instead of hardcoded Red ("click a Red piece" → "click one of
  your pieces"); (2) after `/api/game/new`, if `turn === "engine"` (human
  chose Black), trigger `/api/engine/move` — same call the existing
  post-human-move flow already makes; (3) board stays drawn Red-at-bottom
  in all cases (user decision).

## 4. Error handling

- Missing/incompatible checkpoint or missing `PIKAFISH_BIN` → server fails at
  startup with an actionable one-line message (path + how to fix).
- Illegal human move → HTTP 400 with the existing error-detail shape.
- Engine move that fails env legality (shouldn't happen — MCTS roots are
  masked to `env.legal_moves()`) → 400 with the offending move string, game
  not stepped.
- The env's engine calls can transiently fail (Pikafish restart); those
  surface as the same 400-with-message path — the UI already resyncs on 400.

## 5. Testing

- `web/tests/test_muzero_session.py` (pytest, run via `uv run pytest
  web/tests`): session-level tests using `muzero/tests/helpers.FakeEvaluator`
  and a tiny net (channels=16, 1 block, 4 sims) checkpointed to tmp —
  (1) snapshot contract keys + humanSide; (2) human=red full exchange
  (human move → engine reply, both legal, absolute); (3) human=black: new
  game reports engine turn first, engine move works before any human move;
  (4) illegal-move rejection; (5) result mapping incl. `"draw"`;
  (6) greedy endpoint rejected in muzero mode.
- Manual smoke (documented in README): muzero mode on CPU with the real
  checkpoint, one game each color.
- Engine-gated CI is NOT extended (web tests run with FakeEvaluator only).

## 6. Portability / runbook (README additions)

- 5090 box: `XIANGQI_PLAY_ENGINE=muzero uv run python
  scripts/serve_xiangqi_play.py --engine muzero` — CPU default, so training
  can keep running.
- Mac: install a macOS Pikafish binary + `export PIKAFISH_BIN=...`;
  `scp <box>:.../checkpoints/muzero_xiangqi/latest.pt
  checkpoints/muzero_xiangqi/`; same launch command. Note the ~5–20 s think
  time and the ~30 s startup (checkpoint load only — no 7B).
