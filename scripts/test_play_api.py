#!/usr/bin/env python3
"""Smoke test for the Xiangqi play API (no 7B load by default)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("XIANGQI_PLAY_SKIP_ENGINE", "1")


def main() -> None:
    from fastapi.testclient import TestClient

    from web.server.app import app

    with TestClient(app) as client:
        r = client.post("/api/game/new", json={"allyMode": "human"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["gameOver"] is False
        assert data["turn"] == "human"
        assert data["allyMode"] == "human"
        assert len(data["board"]) == 10

        r2 = client.get("/api/legal", params={"from": "b7"})
        assert r2.status_code == 200, r2.text
        targets = r2.json().get("targets") or []
        assert isinstance(targets, list)

        if targets:
            move = f"b7{targets[0]}"
            r3 = client.post("/api/move", json={"move": move})
            assert r3.status_code == 200, r3.text
            assert r3.json().get("lastAllyMove") == move

        r_g = client.post("/api/game/new", json={"allyMode": "greedy"})
        assert r_g.status_code == 200
        rg = client.post("/api/ally/greedy")
        assert rg.status_code == 200, rg.text
        assert rg.json().get("lastAllyMove")

        r4 = client.get("/api/state")
        assert r4.status_code == 200

    print("test_play_api: OK")


if __name__ == "__main__":
    main()
