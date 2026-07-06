#!/usr/bin/env python3
"""Run the local Xiangqi play web server (LLM or MuZero opponent)."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Xiangqi play UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--adapter",
        default=os.path.join(_ROOT, "checkpoints/xiangqi_grpo_v2/ep_40"),
        help="LoRA adapter directory (default: ep_40)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda/cpu; default: cuda for llm, cpu for muzero",
    )
    ap.add_argument(
        "--skip-engine",
        action="store_true",
        help="Start without loading the 7B model (human moves only)",
    )
    ap.add_argument(
        "--engine",
        choices=["llm", "muzero"],
        default="llm",
        help="Opponent: llm (7B LoRA, default) or muzero (canonical checkpoint)",
    )
    ap.add_argument(
        "--ckpt",
        default=os.path.join(_ROOT, "checkpoints/muzero_xiangqi/latest.pt"),
        help="MuZero checkpoint path (muzero mode only)",
    )
    args = ap.parse_args()

    os.environ["XIANGQI_PLAY_ADAPTER"] = args.adapter
    os.environ["XIANGQI_PLAY_ENGINE"] = args.engine
    os.environ["XIANGQI_MUZERO_CKPT"] = args.ckpt
    os.environ["XIANGQI_PLAY_DEVICE"] = args.device or (
        "cpu" if args.engine == "muzero" else "cuda"
    )
    if args.skip_engine:
        os.environ["XIANGQI_PLAY_SKIP_ENGINE"] = "1"

    import uvicorn

    uvicorn.run(
        "web.server.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
