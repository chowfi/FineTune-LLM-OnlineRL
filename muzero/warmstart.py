"""Cold-start buffer fill: Pikafish-vs-Pikafish games with MultiPV soft
policy targets. Uses a small dedicated UCI wrapper so the shared
PikafishEvaluator (and the LLM pipeline that depends on it) stays untouched."""

from __future__ import annotations

import re
import subprocess

import numpy as np

from muzero.config import MuZeroConfig
from muzero.encoding import move_to_index
from muzero.env import XiangqiEnv
from muzero.replay_buffer import GameHistory, ReplayBuffer
from src.xiangqi_board import engine_uci_to_algebraic

_INFO_RE = re.compile(r"multipv (\d+) score (cp|mate) (-?\d+).* pv ([a-i]\d[a-i]\d)")


def _multipv_probs(lines) -> np.ndarray:
    """Softmax over MultiPV centipawn scores (temperature 200 cp)."""
    cps = np.array([cp for _, cp in lines], dtype=np.float64)
    probs = np.exp((cps - cps.max()) / 200.0)
    return (probs / probs.sum()).astype(np.float32)


def _pick_move_index(lines, ply: int, temperature_moves: int, rng) -> int:
    """Index of the move to PLAY: sampled from the MultiPV softmax for the
    first temperature_moves plies (variety — a deterministic engine would
    otherwise replay one game per opening line forever), best line after."""
    if ply < temperature_moves and len(lines) > 1:
        return int(rng.choice(len(lines), p=_multipv_probs(lines)))
    return 0


class SimpleUciEngine:
    def __init__(
        self, binary_path: str, movetime_ms: int, multipv: int, nodes: int | None = None
    ):
        self.movetime_ms = movetime_ms
        self.nodes = nodes
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._cmd("uci")
            self._wait("uciok")
            self._cmd(f"setoption name MultiPV value {multipv}")
            self._cmd("isready")
            self._wait("readyok")
        except Exception:
            self.proc.kill()
            raise

    def _cmd(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str) -> list:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def _go_command(self) -> str:
        if self.nodes is not None:
            return f"go nodes {self.nodes}"
        return f"go movetime {self.movetime_ms}"

    def search(self, fen: str) -> list:
        """Returns [(engine_uci, cp_side_to_move)] best-first, one per multipv."""
        self._cmd(f"position fen {fen}")
        self._cmd(self._go_command())
        lines = self._wait("bestmove")
        best: dict = {}
        for line in lines:
            m = _INFO_RE.search(line)
            if m:
                rank = int(m.group(1))
                cp = (
                    float(m.group(3))
                    if m.group(2) == "cp"
                    else float(np.sign(int(m.group(3))) * 30000)
                )
                best[rank] = (m.group(4), cp)
        return [best[r] for r in sorted(best)]

    def close(self):
        try:
            self._cmd("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_engine_game(cfg: MuZeroConfig, engine, evaluator, rng) -> GameHistory:
    """One engine-vs-engine game through the standard env pipeline (same
    referee, rewards, and targets as self-play games). Used by warmstart
    (cold start) and by per-iteration seeding (experiment #3)."""
    env = XiangqiEnv(cfg, evaluator)
    env.reset(ally_side="w")
    history = GameHistory()
    opening = cfg.opening_book[int(rng.integers(len(cfg.opening_book)))]
    done = _play_move(env, history, engine_uci_to_algebraic(opening), None)
    while not done:
        lines = engine.search(env.fen())
        if not lines:
            break
        pick = _pick_move_index(lines, len(history), cfg.temperature_moves, rng)
        move = engine_uci_to_algebraic(lines[pick][0])
        done = _play_move(env, history, move, lines)
    history.boards = [b.copy() for b in env.boards]
    history.to_play_history = list(env.to_play_history)
    history.rep_history = list(env.rep_history)
    history.no_progress_history = list(env.no_progress_history)
    history.truncated = env.truncated
    history.ally_side = env.ally_side
    history.result = env.result or "engine_aborted"
    return history


def generate_warmstart_games(
    cfg: MuZeroConfig, buffer: ReplayBuffer, evaluator
) -> dict:
    """Play engine-vs-engine games until >= cfg.warmstart_plies plies are stored."""
    engine = SimpleUciEngine(
        cfg.pikafish_bin, cfg.warmstart_movetime_ms, cfg.warmstart_multipv
    )
    rng = np.random.default_rng(cfg.seed)
    total_plies = games = 0
    try:
        while total_plies < cfg.warmstart_plies:
            history = play_engine_game(cfg, engine, evaluator, rng)
            buffer.add(history)
            total_plies += len(history)
            games += 1
            print(
                f"[warmstart] game {games}: {len(history)} plies ({history.result}) "
                f"— {total_plies}/{cfg.warmstart_plies} plies",
                flush=True,
            )
    finally:
        engine.close()
    return {"plies": total_plies, "games": games}


def generate_seed_games(
    cfg: MuZeroConfig, buffer: ReplayBuffer, evaluator, n_games: int, rng
) -> dict:
    """Experiment #3 (2026-07-14): per-iteration expert-demonstration
    trickle — n_games engine-vs-engine games into the buffer. Unlike
    warmstart this runs every training loop, so the buffer permanently
    holds ~seed_games_per_loop/84 expert data instead of washing it out."""
    if n_games <= 0:
        return {"games": 0, "plies": 0}
    engine = SimpleUciEngine(
        cfg.pikafish_bin, cfg.warmstart_movetime_ms, cfg.warmstart_multipv
    )
    total_plies = games = 0
    try:
        for _ in range(n_games):
            history = play_engine_game(cfg, engine, evaluator, rng)
            buffer.add(history)
            total_plies += len(history)
            games += 1
    finally:
        engine.close()
    return {"games": games, "plies": total_plies}


def _play_move(env: XiangqiEnv, history: GameHistory, move: str, multipv_lines) -> bool:
    if multipv_lines:
        idx = np.array(
            [move_to_index(engine_uci_to_algebraic(u)) for u, _ in multipv_lines],
            dtype=np.int64,
        )
        probs = _multipv_probs(multipv_lines)
        root_value = float(np.tanh(multipv_lines[0][1] / 600.0))  # mover perspective
    else:  # forced opening ply: one-hot
        idx = np.array([move_to_index(move)], dtype=np.int64)
        probs = np.array([1.0], dtype=np.float32)
        root_value = 0.0
    history.actions.append(move_to_index(move))
    history.policy_indices.append(idx)
    history.policy_probs.append(probs)
    history.root_values.append(root_value)
    _, reward, done, _ = env.step(move)
    history.rewards.append(reward)
    return done
