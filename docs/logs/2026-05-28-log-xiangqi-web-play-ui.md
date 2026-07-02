# Experiment Log: Xiangqi Web Play UI

**Date:** 2026-05-28
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Provide a locally hosted interactive board to play full games as Red against the ep_40 default LoRA policy on Black, with Pikafish legality and training-aligned greedy move selection.

## 2. Configuration Changes

- New `web/` package: FastAPI server, static board UI, `EnginePlayer` (CPU + ep_40 adapter).
- `pyproject.toml`: `[dependency-groups] web` with `fastapi`, `uvicorn`.
- `scripts/serve_xiangqi_play.py`, `scripts/test_play_api.py`.

## 3. Run Command

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv sync --group web
uv run python scripts/serve_xiangqi_play.py
# open http://127.0.0.1:8765
```

## 4. Quantitative Results

- Smoke test: `XIANGQI_PLAY_SKIP_ENGINE=1 uv run python scripts/test_play_api.py` (API + Pikafish legals, no 7B).
- Full engine latency on CPU not benchmarked in this log (expect minutes per Black move).

## 5. Conclusion & Next Steps

- Full game loop: human/engine alternate until `gameOver`; not a single ply demo.
- Optional: `--device cuda` when training GPU is free; 4-bit load for faster CPU.
