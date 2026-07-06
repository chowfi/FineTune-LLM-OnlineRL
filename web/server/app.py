"""FastAPI app for local human vs engine Xiangqi play."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: `build_pikafish` is imported from the light `pikafish_setup` module
# (not `game_session`) so that this module-level import never pulls in the
# heavy transformers/peft/torch stack that `game_session -> engine_player`
# depends on. `GameSession`/`EnginePlayer` (LLM mode) and
# `MuZeroGameSession`/`MuZeroPlayer` (MuZero mode) are imported lazily inside
# `lifespan`, gated on `XIANGQI_PLAY_ENGINE`, so only the selected engine's
# dependencies are ever loaded.
from web.server.pikafish_setup import build_pikafish

if TYPE_CHECKING:
    from web.server.engine_player import EnginePlayer
    from web.server.game_session import GameSession
    from web.server.muzero_session import MuZeroGameSession

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

_session: Optional[GameSession | MuZeroGameSession] = None
_engine: Optional[EnginePlayer] = None


class MoveRequest(BaseModel):
    move: str = Field(..., description="Algebraic move e.g. b7b4")


class NewGameRequest(BaseModel):
    allyMode: Literal["human", "greedy"] = Field(
        "human",
        description="LLM mode only. human: click to move; greedy: capture-greedy ally",
    )
    humanSide: Literal["red", "black"] = Field(
        "red", description="MuZero mode only: which color the human plays"
    )


# CONCURRENCY INVARIANT: all route handlers below are `async def` with fully
# synchronous bodies (no awaits around session calls), so requests serialize
# on the event loop and the single global _session is never re-entered.
# Do not convert handlers to `def` (threadpool) or add awaits mid-handler
# without adding a lock around the session's mutating methods.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _engine
    pikafish = build_pikafish()
    engine_kind = os.environ.get("XIANGQI_PLAY_ENGINE", "llm").strip().lower()
    if engine_kind == "muzero":
        from web.server.muzero_player import MuZeroPlayer
        from web.server.muzero_session import MuZeroGameSession

        ckpt = os.environ.get(
            "XIANGQI_MUZERO_CKPT",
            str(_REPO_ROOT / "checkpoints/muzero_xiangqi/latest.pt"),
        )
        device = os.environ.get("XIANGQI_PLAY_DEVICE", "cpu")
        print(f"[xiangqi-play] Loading MuZero ckpt={ckpt} device={device}", flush=True)
        player = MuZeroPlayer(ckpt, device=device)
        _session = MuZeroGameSession(pikafish=pikafish, player=player)
        print("[xiangqi-play] MuZero engine ready.", flush=True)
    else:
        from web.server.engine_player import EnginePlayer
        from web.server.game_session import GameSession

        skip_engine = os.environ.get(
            "XIANGQI_PLAY_SKIP_ENGINE", ""
        ).strip().lower() in {"1", "true", "yes"}
        if not skip_engine:
            adapter = os.environ.get(
                "XIANGQI_PLAY_ADAPTER",
                str(_REPO_ROOT / "checkpoints/xiangqi_grpo_v2/ep_40"),
            )
            device = os.environ.get("XIANGQI_PLAY_DEVICE", "cuda")
            print(
                f"[xiangqi-play] Loading engine adapter={adapter} device={device}",
                flush=True,
            )
            _engine = EnginePlayer(adapter_path=adapter, device=device)
            print("[xiangqi-play] Engine ready.", flush=True)
        else:
            print(
                "[xiangqi-play] Engine skipped (XIANGQI_PLAY_SKIP_ENGINE).",
                flush=True,
            )
        _session = GameSession(pikafish=pikafish, engine=_engine)
    yield


app = FastAPI(title="Xiangqi Play", lifespan=lifespan)


def _require_session() -> GameSession | MuZeroGameSession:
    if _session is None:
        raise HTTPException(503, "Game session not initialized")
    return _session


def _play_error(
    status: int, message: str, session: GameSession | MuZeroGameSession
) -> None:
    snap = session.snapshot()
    detail = {
        "message": message,
        "turn": snap.get("turn"),
        "sideToMove": snap.get("sideToMove"),
        "allyMode": snap.get("allyMode"),
        "gameOver": snap.get("gameOver"),
        "lastAllyMove": snap.get("lastAllyMove"),
        "lastEngineMove": snap.get("lastEngineMove"),
    }
    print(f"[xiangqi-play] HTTP {status}: {detail}", flush=True)
    raise HTTPException(status_code=status, detail=detail)


@app.get("/")
async def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(404, "index.html missing")
    return FileResponse(index_path)


@app.post("/api/game/new")
async def new_game(body: NewGameRequest | None = None) -> Dict[str, Any]:
    sess = _require_session()
    if getattr(sess, "engine_kind", "llm") == "muzero":
        return sess.reset(human_side=body.humanSide if body else "red")
    return sess.reset(ally_mode=body.allyMode if body else "human")


@app.get("/api/state")
async def state() -> Dict[str, Any]:
    return _require_session().snapshot()


@app.get("/api/legal")
async def legal(from_sq: str = Query(..., alias="from")) -> Dict[str, Any]:
    targets = _require_session().legal_targets_from(from_sq)
    return {"from": from_sq.lower(), "targets": targets}


@app.post("/api/move")
async def move(body: MoveRequest) -> Dict[str, Any]:
    """Apply human ally move only (engine is a separate call)."""
    sess = _require_session()
    snap, err = sess.apply_human_move(body.move)
    if err:
        _play_error(400, err, sess)
    return snap


@app.post("/api/ally/greedy")
async def ally_greedy() -> Dict[str, Any]:
    """Apply one capture-greedy ally move (ε=0)."""
    sess = _require_session()
    snap, err = sess.apply_greedy_ally()
    if err:
        _play_error(400, err, sess)
    return snap


@app.post("/api/engine/move")
async def engine_move() -> Dict[str, Any]:
    """Apply one engine (Black) move."""
    sess = _require_session()
    snap, err = sess.apply_engine_move()
    if err:
        _play_error(400, err, sess)
    return snap


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
