"""Pikafish UCI subprocess client (stdlib only). Used by RL training and offline SFT tools."""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

ENGINE_CP_RE = re.compile(r"\bscore\s+cp\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_MATE_RE = re.compile(r"\bscore\s+mate\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_PERFT_MOVE_RE = re.compile(
    r"^([a-i][0-9][a-i][0-9]):\s+\d+\b", flags=re.IGNORECASE
)


class PikafishEvaluator:
    """Communicate with Pikafish using raw (unbuffered) I/O so that
    ``select()`` accurately reflects available data.

    Results are cached by ``(fen, moves_tuple)`` for ``evaluate_cp`` and by
    ``fen`` for ``list_legal_moves``.
    """

    def __init__(
        self,
        binary_path: str,
        depth: int,
        timeout_sec: float = 2.0,
        eval_cache_size: int = 20000,
        legal_cache_size: int = 20000,
        movetime_ms: Optional[int] = None,
        eval_timeout_sec: Optional[float] = None,
        negative_cache_ttl_sec: float = 30.0,
        negative_cache_max: int = 4096,
        poison_fen_ttl_sec: float = 90.0,
        poison_fen_max: int = 1024,
        threads: int = 2,
        hash_mb: int = 128,
        verbose: bool = True,
    ):
        self.binary_path = binary_path
        self.depth = max(1, int(depth))
        self.timeout_sec = max(0.2, float(timeout_sec))
        if movetime_ms is None:
            movetime_ms = max(200, 80 * self.depth)
        self.movetime_ms = max(50, int(movetime_ms))
        self.eval_timeout_sec = (
            float(eval_timeout_sec)
            if eval_timeout_sec is not None
            else (self.movetime_ms / 1000.0 + 2.0)
        )
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.enabled = False
        self.eval_file: Optional[str] = None
        self._buf = b""
        self.engine_dir: Optional[str] = None
        self._eval_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], float]" = (
            OrderedDict()
        )
        self._legal_cache: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict()
        self._root_bm_cache: "OrderedDict[str, Tuple[Optional[str], Optional[float]]]" = OrderedDict()
        self._root_bm_cache_max = 512
        self._eval_cache_max = max(0, int(eval_cache_size))
        self._legal_cache_max = max(0, int(legal_cache_size))
        self._negative_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], float]" = (
            OrderedDict()
        )
        self._negative_cache_ttl = max(0.0, float(negative_cache_ttl_sec))
        self._negative_cache_max = max(0, int(negative_cache_max))
        self._poison_fen_cache: "OrderedDict[str, float]" = OrderedDict()
        self._poison_fen_ttl = max(0.0, float(poison_fen_ttl_sec))
        self._poison_fen_max = max(0, int(poison_fen_max))
        self.threads = max(1, int(threads))
        self.hash_mb = max(1, int(hash_mb))
        self._verbose = bool(verbose)
        self._cache_stats = {
            "eval_hits": 0,
            "eval_misses": 0,
            "legal_hits": 0,
            "legal_misses": 0,
            "negative_hits": 0,
            "restarts": 0,
            "health_restarts": 0,
        }

        resolved = shutil.which(binary_path) if binary_path else None
        if not resolved:
            return
        self.binary_path = resolved
        engine_dir = os.path.dirname(self.binary_path)
        self.engine_dir = engine_dir or None
        eval_candidate = os.path.join(engine_dir, "pikafish.nnue") if engine_dir else ""
        self.eval_file = (
            eval_candidate
            if eval_candidate and os.path.isfile(eval_candidate)
            else None
        )
        self._launch()

    def _launch(self) -> None:
        try:
            self.proc = subprocess.Popen(
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
                return
            if self.eval_file:
                ev_arg = (
                    os.path.basename(self.eval_file)
                    if self.engine_dir
                    and os.path.dirname(os.path.abspath(self.eval_file))
                    == os.path.abspath(self.engine_dir)
                    else self.eval_file
                )
                self._send(f"setoption name EvalFile value {ev_arg}")
            # Keep Pikafish from competing with the LLM training loop for every
            # CPU core. Explicit Hash also avoids engine-default surprises.
            self._send(f"setoption name Threads value {self.threads}")
            self._send(f"setoption name Hash value {self.hash_mb}")
            if not self._sync_ready():
                self.close()
                return
            self.enabled = True
        except Exception:
            self.close()

    def _send(self, command: str) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Pikafish process is not available")
        self.proc.stdin.write((command + "\n").encode())
        self.proc.stdin.flush()

    def _read_lines(self, timeout: float) -> List[str]:
        deadline = time.time() + timeout
        lines: List[str] = []
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0)
            ready, _, _ = select.select(
                [self.proc.stdout], [], [], min(remaining, 0.05)
            )
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                self._buf += chunk
            while b"\n" in self._buf:
                line_bytes, self._buf = self._buf.split(b"\n", 1)
                lines.append(line_bytes.decode(errors="replace").strip())
            if lines and lines[-1].lower().startswith("bestmove"):
                break
        return lines

    def _read_until(self, done_predicate) -> bool:
        if not self.proc or not self.proc.stdout:
            return False
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0)
            ready, _, _ = select.select(
                [self.proc.stdout], [], [], min(remaining, 0.05)
            )
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    return False
                self._buf += chunk
            while b"\n" in self._buf:
                line_bytes, self._buf = self._buf.split(b"\n", 1)
                if done_predicate(line_bytes.decode(errors="replace").strip()):
                    return True
        return False

    def _sync_ready(self) -> bool:
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            self._send("isready")
            return self._read_until(lambda line: "readyok" in line.lower())
        except Exception:
            return False

    def _stop_search(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self._send("stop")
            self._read_until(lambda line: line.lower().startswith("bestmove"))
        except Exception:
            pass

    def _restart(self, reason: str = "") -> bool:
        self._cache_stats["restarts"] += 1
        if self._verbose:
            msg = "[pikafish] restarting engine"
            if reason:
                msg += f" ({reason})"
            print(msg, flush=True)
        self.close()
        self._launch()
        return bool(self.enabled and self.proc and self.proc.poll() is None)

    def _ensure_alive(self, reason: str = "") -> bool:
        if self.proc and self.proc.poll() is None:
            return self.enabled
        return self._restart(reason or "process exited")

    def _health_check_or_restart(self, reason: str = "") -> bool:
        if not self.proc or self.proc.poll() is not None:
            return self._restart(reason or "proc dead before health check")
        if self._sync_ready():
            return True
        self._cache_stats["health_restarts"] += 1
        return self._restart(reason or "isready timeout")

    def _negative_cache_hit(self, key: Tuple[str, Tuple[str, ...]]) -> bool:
        if self._negative_cache_ttl <= 0 or self._negative_cache_max <= 0:
            return False
        ts = self._negative_cache.get(key)
        if ts is None:
            return False
        if time.time() - ts > self._negative_cache_ttl:
            try:
                del self._negative_cache[key]
            except KeyError:
                pass
            return False
        self._negative_cache.move_to_end(key)
        self._cache_stats["negative_hits"] += 1
        return True

    def _negative_cache_put(self, key: Tuple[str, Tuple[str, ...]]) -> None:
        if self._negative_cache_ttl <= 0 or self._negative_cache_max <= 0:
            return
        self._negative_cache[key] = time.time()
        self._negative_cache.move_to_end(key)
        while len(self._negative_cache) > self._negative_cache_max:
            self._negative_cache.popitem(last=False)

    def _poison_fen_hit(self, fen: str) -> bool:
        if self._poison_fen_ttl <= 0 or self._poison_fen_max <= 0:
            return False
        ts = self._poison_fen_cache.get(fen)
        if ts is None:
            return False
        if time.time() - ts > self._poison_fen_ttl:
            try:
                del self._poison_fen_cache[fen]
            except KeyError:
                pass
            return False
        self._poison_fen_cache.move_to_end(fen)
        self._cache_stats["negative_hits"] += 1
        return True

    def _poison_fen_put(self, fen: str) -> None:
        if self._poison_fen_ttl <= 0 or self._poison_fen_max <= 0:
            return
        self._poison_fen_cache[fen] = time.time()
        self._poison_fen_cache.move_to_end(fen)
        while len(self._poison_fen_cache) > self._poison_fen_max:
            self._poison_fen_cache.popitem(last=False)

    def _cache_put_eval(self, key: Tuple[str, Tuple[str, ...]], value: float) -> None:
        if self._eval_cache_max <= 0:
            return
        self._eval_cache[key] = value
        self._eval_cache.move_to_end(key)
        while len(self._eval_cache) > self._eval_cache_max:
            self._eval_cache.popitem(last=False)

    def _cache_put_legal(self, fen: str, value: Tuple[str, ...]) -> None:
        if self._legal_cache_max <= 0:
            return
        self._legal_cache[fen] = value
        self._legal_cache.move_to_end(fen)
        while len(self._legal_cache) > self._legal_cache_max:
            self._legal_cache.popitem(last=False)

    def cache_stats(self) -> Dict[str, int]:
        return dict(self._cache_stats)

    def list_legal_moves(self, fen: str) -> Optional[List[str]]:
        if fen in self._legal_cache:
            self._legal_cache.move_to_end(fen)
            self._cache_stats["legal_hits"] += 1
            return list(self._legal_cache[fen])
        if self._poison_fen_hit(fen):
            return None
        neg_key = (fen, ("__legal__",))
        if self._negative_cache_hit(neg_key):
            return None
        if not self._ensure_alive("list_legal_moves"):
            return None
        self._cache_stats["legal_misses"] += 1
        for attempt in range(2):
            try:
                if attempt > 0 and not self._restart("legal_moves retry"):
                    self._negative_cache_put(neg_key)
                    return None
                if not self._sync_ready():
                    continue
                self._send(f"position fen {fen}")
                self._send("go perft 1")
                lines = self._read_lines(self.timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._poison_fen_put(fen)
                    self._restart("critical error on perft")
                    continue
                legal_moves: List[str] = []
                for text in lines:
                    match = ENGINE_PERFT_MOVE_RE.match(text)
                    if match:
                        legal_moves.append(match.group(1).lower())
                if legal_moves:
                    self._cache_put_legal(fen, tuple(legal_moves))
                    return legal_moves
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
        self._health_check_or_restart("legal_moves exhausted")
        self._negative_cache_put(neg_key)
        return None

    def _parse_score_from_lines(self, lines: List[str]) -> Optional[float]:
        latest_score: Optional[float] = None
        for text in lines:
            text_lower = text.lower()
            cp_match = ENGINE_CP_RE.search(text_lower)
            if cp_match:
                latest_score = float(cp_match.group(1))
            mate_match = ENGINE_MATE_RE.search(text_lower)
            if mate_match:
                mate_dist = int(mate_match.group(1))
                sign = 1.0 if mate_dist >= 0 else -1.0
                latest_score = sign * max(
                    9000.0, 10000.0 - 100.0 * min(abs(mate_dist), 90)
                )
        return latest_score

    def bestmove_and_root_cp(self, fen: str) -> Tuple[Optional[str], Optional[float]]:
        """One search from *fen*: ``(bestmove_uci_lower, root_cp_for_side_to_move)``."""
        if self._poison_fen_hit(fen):
            return None, None
        if not self._ensure_alive("bestmove_and_root_cp"):
            return None, None
        for attempt in range(2):
            try:
                if attempt > 0 and not self._restart("bestmove_and_root_cp retry"):
                    return None, None
                if not self._sync_ready():
                    continue
                self._send(f"position fen {fen}")
                self._send(f"go movetime {self.movetime_ms}")
                lines = self._read_lines(self.eval_timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._poison_fen_put(fen)
                    self._restart("critical error on bestmove search")
                    continue
                best: Optional[str] = None
                for text in lines:
                    tl = text.lower()
                    if tl.startswith("bestmove"):
                        parts = text.split()
                        if len(parts) >= 2 and parts[1].lower() != "(none)":
                            best = parts[1].lower()
                score = self._parse_score_from_lines(lines)
                if score is not None:
                    self._cache_put_eval((fen, ()), score)
                return best, score
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
        self._health_check_or_restart("bestmove_and_root_cp exhausted")
        return None, None

    def bestmove_root_cached(self, fen: str) -> Tuple[Optional[str], Optional[float]]:
        """Cache one ``(bestmove, root_cp)`` search per *fen* (many GRPO candidates)."""
        if fen in self._root_bm_cache:
            self._root_bm_cache.move_to_end(fen)
            return self._root_bm_cache[fen]
        bm, sc = self.bestmove_and_root_cp(fen)
        if self._root_bm_cache_max > 0 and (bm is not None or sc is not None):
            self._root_bm_cache[fen] = (bm, sc)
            self._root_bm_cache.move_to_end(fen)
            while len(self._root_bm_cache) > self._root_bm_cache_max:
                self._root_bm_cache.popitem(last=False)
        return bm, sc

    def evaluate_cp(
        self, fen: str, moves: Optional[List[str]] = None
    ) -> Optional[float]:
        moves_tuple: Tuple[str, ...] = tuple(moves or ())
        cache_key = (fen, moves_tuple)
        if self._poison_fen_hit(fen):
            return None
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            self._eval_cache.move_to_end(cache_key)
            self._cache_stats["eval_hits"] += 1
            return cached
        if self._negative_cache_hit(cache_key):
            return None
        if not self._ensure_alive("evaluate_cp"):
            return None
        self._cache_stats["eval_misses"] += 1
        for attempt in range(2):
            try:
                if attempt > 0 and not self._restart("evaluate_cp retry"):
                    self._negative_cache_put(cache_key)
                    return None
                if not self._sync_ready():
                    continue

                position_cmd = f"position fen {fen}"
                if moves_tuple:
                    position_cmd += " moves " + " ".join(moves_tuple)
                self._send(position_cmd)
                self._send(f"go movetime {self.movetime_ms}")

                lines = self._read_lines(self.eval_timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._poison_fen_put(fen)
                    self._restart("critical error on eval")
                    continue
                latest_score = self._parse_score_from_lines(lines)
                if latest_score is not None:
                    self._cache_put_eval(cache_key, latest_score)
                    return latest_score
                self._stop_search()
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
        self._health_check_or_restart("evaluate_cp exhausted")
        self._negative_cache_put(cache_key)
        return None

    def close(self) -> None:
        if self.proc is None:
            self.enabled = False
            return
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write(b"quit\n")
                self.proc.stdin.flush()
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        self.enabled = False
