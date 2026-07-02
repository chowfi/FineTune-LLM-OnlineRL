# Experiment Log: LLM Chess vs Xiangqi Elo Benchmark (inference only)

**Date:** 2026-05-13
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

**Hypothesis:** Base `Qwen/Qwen2.5-7B-Instruct` has had far more exposure to
Western chess in pretraining than to Xiangqi, so it should play chess much
better than Xiangqi out of the box, with no fine-tuning. We want **two
numbers + one cross-game plot** that we can defend to a colleague:

* an LLM Elo for chess (calibrated against Stockfish at different movetimes),
* an LLM Elo for Xiangqi (calibrated against Pikafish at the same movetimes),
* a win-rate-vs-movetime curve overlaying both games on the same axis.

Both engines use the exact same compute knob (`go movetime <ms>`), and we
self-calibrate each engine's rungs to within-pool Elo via Bradley-Terry MLE
with Rao-Kupper draws. The top rung (5 s/move) in each pool is anchored at
**3500 Elo** as a convention (near the top of published engine strength in
each game).

## 2. Configuration Changes

### New files

- `scripts/benchmark/__init__.py`
- `scripts/benchmark/xiangqi_prompt.py` — copied verbatim from
  `XiangqiAgent.system_prompt` / `format_turn_prompt` in
  `LLM_RL_agent_FSDP_v2.py` (so the bench uses the SAME prompt v2 trains
  against); takes the cchess piece-placement FEN + a v2-style top-origin
  graphic.
- `scripts/benchmark/chess_prompt.py` — parallel-shaped prompt for Western
  chess; same `<think>...</think>\nMove: <uci>` output contract; promotion
  suffix optional (`e7e8q`).
- `scripts/benchmark/boards.py` — `XiangqiBoard` (cchess-backed,
  bottom-origin Pikafish UCI internally; `engine_uci_to_algebraic` /
  `algebraic_to_engine_move` bridge to the LLM's top-origin format) and
  `ChessBoard` (python-chess-backed, standard UCI throughout). Both expose
  one interface: `fen`, `llm_fen`, `graphic`, `legal_moves_engine/llm`,
  `apply_engine_move/llm_move`, `is_terminal`, `side_to_move`.
- `scripts/benchmark/engines.py` — single `UciEngine` class (subprocess +
  blocking I/O, `setoption Threads/Hash`, Pikafish's `EvalFile pikafish.nnue`
  auto-detected) plus an `EnginePool` that keeps one process alive per
  `(game, movetime_ms)` rung.
- `scripts/benchmark/llm_player.py` — plain transformers
  `AutoModelForCausalLM` load (not Unsloth — keeps the SFT job's
  `unsloth_compiled_cache/` untouched), chat template, sampling
  (T=0.6, top_p=0.9, max_new_tokens=192), permissive Move regex covering
  both chess and xiangqi UCI, random-legal fallback on parse/legality fail
  (counted as `format_fail`).
- `scripts/benchmark/run_match.py` — `play_llm_vs_engine` and
  `play_engine_vs_engine`, plus per-move JSONL logging under
  `data/benchmark/games/`.
- `scripts/benchmark/elo_estimator.py` — Bradley-Terry MLE with Rao-Kupper
  draw extension, L-BFGS-B with bounds + soft prior on `log(theta)`. Public
  API: `fit_ratings(games, fixed_ratings=...)`, `bootstrap_ci`,
  `predicted_score`.
- `scripts/benchmark/calibrate_ladder.py` — engine self-gauntlet on adjacent
  rungs + a bottom-top anchor; writes `data/benchmark/<game>_ladder_elo.json`.
- `scripts/benchmark/run_benchmark.py` — CLI orchestrator: ladder load or
  recompute → `--games-per-rung` LLM games per rung → Bradley-Terry fit with
  rungs pinned → bootstrap 95 % CI → matplotlib win-rate plot.
- `scripts/benchmark/smoke.py` — 1 LLM-vs-engine game per game at 50 ms.

### Updated

- `pyproject.toml`: added `chess>=1.10,<2` (python-chess); added
  `scripts/**/*.py` to the ruff `E402` per-file-ignores so the `sys.path`
  prepend pattern continues to lint cleanly.

### Not changed

- `LLM_RL_agent_FSDP_v2.py` — bench reads its prompt strings only; v2 stays
  bit-for-bit identical.
- `pikafish_eval.py`, `xiangqi_board.py`, `xiangqi_labels.py` — reused as
  shared helpers.
- The running `build_xiangqi_sft_dataset.py` job — none of the files it
  imports are touched by this change.

## 3. Run Command

```bash
# 0) Once the SFT build finishes, sync the new python-chess dep + install
#    Stockfish if needed.
uv sync
sudo apt-get install -y stockfish   # or download from stockfishchess.org

export STOCKFISH_BIN=$(which stockfish)
export PIKAFISH_BIN=$(which pikafish)

# 1) Smoke test (1 game per game at 50 ms).
uv run python -m scripts.benchmark.smoke --game both --movetime-ms 50

# 2) Full benchmark (default 5 rungs, 20 games/rung, 30 calibration games/pair).
uv run python -m scripts.benchmark.run_benchmark --game both \
  --rungs 10,50,200,1000,5000 \
  --games-per-rung 20 \
  --calibration-games-per-pair 30 \
  --out data/benchmark
```

Outputs:

- `data/benchmark/<game>_ladder_elo.json` — rung Elos + theta + telemetry.
- `data/benchmark/<game>_results.json` — LLM Elo + CI + win rates by rung.
- `data/benchmark/<game>_winrate.png` — win-rate-vs-movetime plot.
- `data/benchmark/summary.json` — both games' Elos + headline chess-minus-xiangqi gap.
- `data/benchmark/games/*.jsonl` — per-game move logs.

## 4. Quantitative Results

Not run in this session (the GRPO training is occupying the GPU). Targets
to verify in the follow-up run:

* **Calibration sanity:** `sanity_low_vs_high_elo_gap` (rung 10 ms vs
  5000 ms) should be **~+500 Elo** for both games (well-studied doubling
  of time ≈ +50-70 Elo for Stockfish-family engines, expected to hold for
  Pikafish). If the gap is far smaller, debug the rung config — common
  causes: `Threads`/`Hash` not actually set per process, OS not honoring
  `movetime` for the very-fast 10 ms rung.
* **Headline numbers we are looking for:**
  - LLM chess Elo: likely **~1000-1700** (intermediate club player).
  - LLM Xiangqi Elo: likely **~400-900** (knows the rules, blunders often).
  - Chess-minus-xiangqi Elo gap: **≥ 500** (clean evidence for the
    pretraining-exposure hypothesis).
* **Format compliance:** `format_fail_rate` should be **< 0.10** for chess
  and **< 0.20** for xiangqi based on v2 baseline behaviour with the same
  prompt. Higher rates dilute the Elo number toward random-move strength
  (a separate failure mode to report).

## 5. Conclusion & Next Steps

Build complete; will run the calibration + benchmark once the GPU frees up.

## 6. Limitations & Interpretation

1. **Time-equals-skill assumption.** For Stockfish / Pikafish-family
   engines this is empirically rock-solid: doubling thinking time gives
   roughly +50-70 Elo, well-studied on Fishtest and reproduced across many
   engines. It is **not** true for humans or for LLMs without search. This
   benchmark relies on the **engine-side** scaling; the LLM is fixed at one
   inference setting (T=0.6, top_p=0.9, max_new_tokens=192) and is the
   thing being measured, not the ladder.

2. **Cross-game Elo is pool-relative, not absolute.** Each game's rung Elos
   are anchored top-rung = 3500 **within their own engine pool**. There is
   no proof Pikafish@5s and Stockfish@5s play at the same absolute strength
   on some universal scale (no such scale exists). So:
   - "LLM chess Elo 1500" and "LLM xiangqi Elo 800" are each well-defined
     **inside** their pool.
   - "1500 - 800 = 700 Elo gap" is **suggestive evidence**, not a
     calibrated cross-game claim.
   - The **anchor-free** evidence is the shared win-rate-vs-movetime
     curve; it uses the same compute axis on both games and does not
     depend on any Elo anchor.

3. **Exposure vs intrinsic difficulty is confounded.** A weak xiangqi
   number could mean (a) the model saw less xiangqi in pretraining (the
   hypothesis), **or** (b) xiangqi is intrinsically harder for
   transformer-class models (bigger board, more piece types, blocked-horse
   / elephant-eye / palace rules, no familiar Western chess priors to
   transfer). A single base-model measurement **cannot** distinguish
   (a) from (b). The clean follow-up is to re-run the same bench on the
   xiangqi SFT adapter (`checkpoints/xiangqi_sft_strategy`) and compare
   the gap delta — added to backlog (see `docs/AGENT_TODO.md`).

4. **Color bias.** LLM always plays uppercase (White / Red). The v2 prompt
   is hard-coded for the uppercase side; supporting LLM-as-black would
   require either a separate prompt or perspective-flipping the board.
   This costs ~30 Elo of first-mover advantage. It applies **symmetrically
   to both games**, so the cross-game gap is unaffected.

5. **One LLM seed, one decoder setting.** We use sampling (T=0.6,
   top_p=0.9) with a fixed seed. Running the same benchmark with
   `do_sample=False` (greedy) would give a different Elo, often lower
   for instruction-tuned models on long-chain reasoning tasks. Report
   the decoder setting alongside the Elo.
