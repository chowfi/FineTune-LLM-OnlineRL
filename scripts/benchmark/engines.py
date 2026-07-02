"""UCI engine clients for Stockfish + Pikafish, with movetime-rung support.

Both Stockfish and Pikafish speak UCI; only their initialization differs
slightly (Pikafish wants ``EvalFile pikafish.nnue`` co-located with the
binary). This module provides a single ``UciEngine`` class that handles
both, and a tiny pool that keeps one subprocess alive per
``(engine, movetime)`` rung so we don't pay startup cost per move.

Returned moves are lowercase UCI strings (engine-native: Pikafish uses
bottom-origin xiangqi coords; Stockfish uses standard chess UCI).

Design note: we intentionally do not reuse :class:`PikafishEvaluator` here
because that class is tuned for the GRPO training loop (eval caching,
restart-on-error). The benchmark only needs ``bestmove`` and benefits from
a simpler, deterministic client. The handshake/I/O pattern is the same.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple


def _resolve_binary(env_var: Optional[str], default_name: str) -> Optional[str]:
    """``env_var`` -> ``PATH`` -> None. ``env_var`` may be a path or a name."""
    if env_var:
        path = os.environ.get(env_var, "").strip()
        if path:
            resolved = shutil.which(path) or (path if os.path.isfile(path) else None)
            if resolved:
                return resolved
    return shutil.which(default_name)


class UciEngine:
    """Minimal blocking UCI client.

    Lifecycle:

        eng = UciEngine.stockfish(movetime_ms=200)
        try:
            uci = eng.bestmove(fen)
        finally:
            eng.close()
    """

    def __init__(
        self,
        binary_path: str,
        movetime_ms: int,
        *,
        eval_file: Optional[str] = None,
        engine_name: str = "uci",
        startup_timeout: float = 5.0,
        bestmove_timeout: Optional[float] = None,
        threads: int = 1,
        hash_mb: int = 64,
        extra_options: Optional[Dict[str, str]] = None,
    ):
        self.binary_path = binary_path
        self.movetime_ms = max(1, int(movetime_ms))
        self.engine_name = engine_name
        self.startup_timeout = float(startup_timeout)
        self.bestmove_timeout = (
            float(bestmove_timeout)
            if bestmove_timeout is not None
            else (self.movetime_ms / 1000.0 + 5.0)
        )
        self.engine_dir = os.path.dirname(binary_path) or None
        self.eval_file = eval_file
        self.threads = max(1, int(threads))
        self.hash_mb = max(1, int(hash_mb))
        self.extra_options = dict(extra_options or {})

        self._buf = b""
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._launch()

    # ----- Factories -----

    @classmethod
    def stockfish(
        cls,
        *,
        movetime_ms: int,
        binary_path: Optional[str] = None,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> "UciEngine":
        path = binary_path or _resolve_binary("STOCKFISH_BIN", "stockfish")
        if not path:
            raise RuntimeError(
                "Stockfish binary not found. Set STOCKFISH_BIN or install "
                "stockfish on PATH (apt-get install stockfish, or download "
                "from https://stockfishchess.org/download/)."
            )
        return cls(
            binary_path=path,
            movetime_ms=movetime_ms,
            engine_name="stockfish",
            threads=threads,
            hash_mb=hash_mb,
        )

    @classmethod
    def pikafish(
        cls,
        *,
        movetime_ms: int,
        binary_path: Optional[str] = None,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> "UciEngine":
        path = binary_path or _resolve_binary("PIKAFISH_BIN", "pikafish")
        if not path:
            raise RuntimeError(
                "Pikafish binary not found. Set PIKAFISH_BIN or install "
                "pikafish on PATH (https://github.com/official-pikafish/Pikafish)."
            )
        engine_dir = os.path.dirname(path) or None
        eval_candidate = (
            os.path.join(engine_dir, "pikafish.nnue") if engine_dir else None
        )
        eval_file = (
            eval_candidate
            if eval_candidate and os.path.isfile(eval_candidate)
            else None
        )
        return cls(
            binary_path=path,
            movetime_ms=movetime_ms,
            eval_file=eval_file,
            engine_name="pikafish",
            threads=threads,
            hash_mb=hash_mb,
        )

    # ----- Subprocess management -----

    def _launch(self) -> None:
        self._proc = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            cwd=self.engine_dir,
        )
        self._buf = b""
        self._send("uci")
        if not self._read_until(lambda line: "uciok" in line.lower()):
            self.close()
            raise RuntimeError(f"{self.engine_name}: no uciok after startup")
        # Common options
        self._set_option("Threads", str(self.threads))
        self._set_option("Hash", str(self.hash_mb))
        if self.eval_file:
            ev_arg = (
                os.path.basename(self.eval_file)
                if self.engine_dir
                and os.path.dirname(os.path.abspath(self.eval_file))
                == os.path.abspath(self.engine_dir)
                else self.eval_file
            )
            self._set_option("EvalFile", ev_arg)
        for k, v in self.extra_options.items():
            self._set_option(k, v)
        if not self._isready():
            self.close()
            raise RuntimeError(f"{self.engine_name}: readyok timeout after options")

    def _set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def _send(self, line: str) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError(f"{self.engine_name}: process not available")
        self._proc.stdin.write((line + "\n").encode())
        self._proc.stdin.flush()

    def _read_lines(self, timeout: float) -> List[str]:
        deadline = time.time() + timeout
        lines: List[str] = []
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0)
            ready, _, _ = select.select(
                [self._proc.stdout], [], [], min(remaining, 0.05)
            )
            if ready:
                chunk = os.read(self._proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                self._buf += chunk
            while b"\n" in self._buf:
                line_bytes, self._buf = self._buf.split(b"\n", 1)
                lines.append(line_bytes.decode(errors="replace").strip())
            if lines and lines[-1].lower().startswith("bestmove"):
                break
        return lines

    def _read_until(self, predicate) -> bool:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0)
            ready, _, _ = select.select(
                [self._proc.stdout], [], [], min(remaining, 0.05)
            )
            if ready:
                chunk = os.read(self._proc.stdout.fileno(), 65536)
                if not chunk:
                    return False
                self._buf += chunk
            while b"\n" in self._buf:
                line_bytes, self._buf = self._buf.split(b"\n", 1)
                if predicate(line_bytes.decode(errors="replace").strip()):
                    return True
        return False

    def _isready(self) -> bool:
        self._send("isready")
        return self._read_until(lambda line: "readyok" in line.lower())

    # ----- Public API -----

    def newgame(self) -> None:
        self._send("ucinewgame")
        self._isready()

    def bestmove(self, fen: str) -> Optional[str]:
        if not self._proc or self._proc.poll() is not None:
            return None
        self._send(f"position fen {fen}")
        self._send(f"go movetime {self.movetime_ms}")
        lines = self._read_lines(self.bestmove_timeout)
        for text in lines:
            if text.lower().startswith("bestmove"):
                parts = text.split()
                if len(parts) >= 2 and parts[1].lower() not in {"(none)", "0000"}:
                    return parts[1].lower()
        return None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._proc.stdin.write(b"quit\n")
                self._proc.stdin.flush()
            self._proc.terminate()
        except Exception:
            pass
        self._proc = None


# -----------------------------------------------------------------------------
# Engine pool: one subprocess per (game, movetime_ms) rung
# -----------------------------------------------------------------------------


class EnginePool:
    """Keeps one UciEngine alive per (game, movetime_ms) so per-game startup
    cost is paid once across the whole benchmark."""

    def __init__(self, *, threads: int = 1, hash_mb: int = 64):
        self._engines: Dict[Tuple[str, int], UciEngine] = {}
        self._threads = threads
        self._hash_mb = hash_mb

    def get(self, game: str, movetime_ms: int) -> UciEngine:
        key = (game, int(movetime_ms))
        eng = self._engines.get(key)
        if eng is not None:
            return eng
        if game == "chess":
            eng = UciEngine.stockfish(
                movetime_ms=movetime_ms,
                threads=self._threads,
                hash_mb=self._hash_mb,
            )
        elif game in {"xiangqi", "xq"}:
            eng = UciEngine.pikafish(
                movetime_ms=movetime_ms,
                threads=self._threads,
                hash_mb=self._hash_mb,
            )
        else:
            raise ValueError(f"Unknown game: {game!r}")
        self._engines[key] = eng
        return eng

    def close_all(self) -> None:
        for eng in self._engines.values():
            try:
                eng.close()
            except Exception:
                pass
        self._engines.clear()

    def __enter__(self) -> "EnginePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_all()
