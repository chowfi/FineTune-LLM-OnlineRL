"""FastAPI app for local human vs engine Xiangqi play."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web.server.engine_player import EnginePlayer
from web.server.game_session import GameSession, build_pikafish

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

_session: Optional[GameSession] = None
_engine: Optional[EnginePlayer] = None


class MoveRequest(BaseModel):
    move: str = Field(..., description="Algebraic move e.g. b7b4")


class NewGameRequest(BaseModel):
    allyMode: Literal["human", "greedy"] = Field(
        "human",
        description="human: click to move; greedy: capture-greedy ally (ε=0)",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _engine
    pikafish = build_pikafish()
    skip_engine = os.environ.get("XIANGQI_PLAY_SKIP_ENGINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
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
        print("[xiangqi-play] Engine skipped (XIANGQI_PLAY_SKIP_ENGINE).", flush=True)
    _session = GameSession(pikafish=pikafish, engine=_engine)
    yield


app = FastAPI(title="Xiangqi Play", lifespan=lifespan)


def _require_session() -> GameSession:
    if _session is None:
        raise HTTPException(503, "Game session not initialized")
    return _session


def _play_error(status: int, message: str, session: GameSession) -> None:
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
    mode = body.allyMode if body else "human"
    return _require_session().reset(ally_mode=mode)


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
