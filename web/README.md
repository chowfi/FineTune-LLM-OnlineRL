# Xiangqi Play UI

Local web app with two opponents:

- **LLM engine** (default): you (Red) vs the ep_40 LoRA policy (Black).
- **MuZero engine** (`--engine muzero`): you as **either color** vs the
  canonical MuZero checkpoint — CPU-friendly, so training can keep running.

## Requirements

- `PIKAFISH_BIN` pointing at Pikafish (same as training)
- LLM mode: NVIDIA GPU with ~16GB+ VRAM recommended; stop other GPU jobs first.
- MuZero mode: any CPU is fine (no GPU needed, no 7B load).
- `uv sync --group web`

## Play vs MuZero (either color, CPU-friendly)

```bash
export PIKAFISH_BIN=/path/to/pikafish
uv run --group web python scripts/serve_xiangqi_play.py --engine muzero
```

- Uses `checkpoints/muzero_xiangqi/latest.pt` by default (`--ckpt` to
  override); runs on CPU by default so training can keep going. Old
  pre-canonicalization (115-plane) checkpoints are rejected at startup.
- Pick **Red or Black** next to "New game" — as Black, the model moves first
  automatically.
- Full training strength (800 simulations): expect ~5–20 s of thinking per
  move on a laptop CPU. Startup takes seconds (no 7B load).
- The training-time hopeless-position auto-adjudication is disabled — you
  always get to finish (or attempt to save) a game. Repetition draws and the
  300-ply cap still apply.
- On a Mac: install a macOS Pikafish build, set `PIKAFISH_BIN`, and copy the
  checkpoint over:

  ```bash
  scp <box>:~/Documents/FineTune-LLM-OnlineRL/checkpoints/muzero_xiangqi/latest.pt checkpoints/muzero_xiangqi/
  ```

## Run (LLM engine)

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
