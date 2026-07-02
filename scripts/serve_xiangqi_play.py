#!/usr/bin/env python3
"""Run the local Xiangqi play web server (human Red vs LoRA engine)."""

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
        default="cuda",
        help="Engine device: cuda (default), cpu, or cuda:0",
    )
    ap.add_argument(
        "--skip-engine",
        action="store_true",
        help="Start without loading the 7B model (human moves only)",
    )
    args = ap.parse_args()

    os.environ["XIANGQI_PLAY_ADAPTER"] = args.adapter
    os.environ["XIANGQI_PLAY_DEVICE"] = args.device
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
