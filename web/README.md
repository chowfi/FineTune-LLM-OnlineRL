# Xiangqi Play UI

Local web app: **you (Red)** vs **ep_40 LoRA policy (Black)**.

## Requirements

- `PIKAFISH_BIN` pointing at Pikafish (same as training)
- NVIDIA GPU with ~16GB+ VRAM recommended (default). CPU works but is very slow.
- Stop other GPU jobs (e.g. training) before starting the play server.
- `uv sync --group web`

## Run

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish   # adjust path
uv sync --group web
uv run python scripts/serve_xiangqi_play.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

First startup loads the base model + `checkpoints/xiangqi_grpo_v2/ep_40` on GPU (default) and takes ~30s–2min.

## Options

```bash
uv run --group web python scripts/serve_xiangqi_play.py --device cuda
uv run --group web python scripts/serve_xiangqi_play.py --device cpu   # slow; avoids GPU
uv run --group web python scripts/serve_xiangqi_play.py --skip-engine   # board + Pikafish only (no 7B)
```

Environment:

- `XIANGQI_PLAY_ADAPTER` — LoRA directory
- `XIANGQI_PLAY_DEVICE` — `cuda` (default) or `cpu`
- `XIANGQI_PLAY_SKIP_ENGINE=1` — skip model load

## Game flow

**Ally mode (choose before New game):**

- **Human** — click Red piece → legal circles on intersections → click destination; your move appears, then the engine move after a short pause.
- **Greedy agent (ε=0)** — capture-greedy Red (no highlights); ally move is shown, then engine thinking, then Black.

Full game until terminal. Board uses **line intersections** (not square centers) and traditional Red/Black piece characters (e.g. 帥/將, 炮/砲, 傌/馬).

Engine moves use **greedy logprob over Pikafish-legal moves** on a flipped board. Legality is checked with Pikafish before each ply.

## Smoke test (no 7B)

```bash
export PIKAFISH_BIN=...
XIANGQI_PLAY_SKIP_ENGINE=1 uv run python scripts/test_play_api.py
```
