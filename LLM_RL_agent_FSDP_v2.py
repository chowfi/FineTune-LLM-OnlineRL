"""
export PIKAFISH_BIN=/absolute/path/to/pikafish
7B / single 5090:
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
14B / two 5090:
torchrun --nproc_per_node 2 LLM_RL_agent_FSDP_v2.py --model-size 14b --mixed-precision
"""

import argparse
import csv
import json
import math
import os
import random
import re
import select
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

import gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from peft import PeftModel
import peft.tuners.tuners_utils as _peft_tuner_utils

try:
    from unsloth import FastLanguageModel
except ImportError as err:
    raise ImportError(
        "unsloth is required for LLM_RL_agent_FSDP_v2.py. "
        "From the repo root: uv sync (see pyproject.toml; Unsloth needs transformers 4.57.x)."
    ) from err

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from gym_xiangqi.constants import ALLY, PIECE_ID_TO_NAME, PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move, move_to_action_space

# PEFT's BaseTuner.forward() uses self.model.forward() directly, which bypasses
# __call__ and can skip distributed pre-forward hooks.
_peft_tuner_utils.BaseTuner.forward = lambda self, *args, **kwargs: self.model(*args, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--mixed-precision", action="store_true")
parser.add_argument("--use-ddp", action="store_true")
parser.add_argument("--model-size", choices=["7b", "14b"], default="7b")
parser.add_argument("--episodes", type=int, default=None)
args = parser.parse_args()


def _dist_env() -> Tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank_val = int(os.environ["RANK"])
        local_rank_val = int(os.environ.get("LOCAL_RANK", 0))
        world_size_val = int(os.environ["WORLD_SIZE"])
        return rank_val, local_rank_val, world_size_val, True
    return 0, 0, 1, False


rank, local_rank, world_size, distributed = _dist_env()

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

if distributed and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
    )


def dist_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_int(value: int, src: int = 0) -> int:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.long, device=device)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def broadcast_bool(value: bool, src: int = 0) -> bool:
    return bool(broadcast_int(int(value), src=src))


def broadcast_generation_inputs(
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    num_generations: int,
    src: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if not dist.is_initialized():
        ids = input_ids.long().to(device).repeat(num_generations, 1)
        mask = attention_mask.long().to(device).repeat(num_generations, 1)
        return ids, mask, int(attention_mask.size(1))

    if rank == src:
        seq_len = int(input_ids.size(1))
        batch = int(num_generations)
    else:
        seq_len = 0
        batch = 0

    seq_len = broadcast_int(seq_len, src=src)
    batch = broadcast_int(batch, src=src)

    if rank == src:
        ids = input_ids.long().to(device).repeat(batch, 1)
        mask = attention_mask.long().to(device).repeat(batch, 1)
    else:
        ids = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask = torch.zeros(batch, seq_len, dtype=torch.long, device=device)

    dist.broadcast(ids, src=src)
    dist.broadcast(mask, src=src)
    return ids, mask, seq_len


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


# -----------------------------------------------------------------------------
# Board conversion utilities
# -----------------------------------------------------------------------------

COLS = "abcdefghi"
COL_TO_IDX = {c: i for i, c in enumerate(COLS)}

_PIECE_TO_FEN = {
    1: "k",
    2: "a",
    3: "a",
    4: "b",
    5: "b",
    6: "n",
    7: "n",
    8: "r",
    9: "r",
    10: "c",
    11: "c",
    12: "p",
    13: "p",
    14: "p",
    15: "p",
    16: "p",
}

MOVE_RE = re.compile(r"Move:\s*([a-i][0-9][a-i][0-9])", flags=re.IGNORECASE)
ALGEBRAIC_RE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])$")
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
ENGINE_CP_RE = re.compile(r"\bscore\s+cp\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_MATE_RE = re.compile(r"\bscore\s+mate\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_PERFT_MOVE_RE = re.compile(r"^([a-i][0-9][a-i][0-9]):\s+\d+\b", flags=re.IGNORECASE)
MOVE_IN_TEXT_RE = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", flags=re.IGNORECASE)

THREAT_KEYWORDS = (
    "threat",
    "attack",
    "pressure",
    "fork",
    "pin",
    "check",
    "mate",
    "expose",
    "vulnerable",
    "risk",
)
ENEMY_KEYWORDS = (
    "enemy",
    "opponent",
    "their",
    "response",
    "counter",
    "counterplay",
    "reply",
)


def board_to_fen(state: np.ndarray) -> str:
    fen_rows: List[str] = []
    for row in state:
        empties = 0
        tokens: List[str] = []
        for cell in row:
            val = int(cell)
            if val == 0:
                empties += 1
                continue
            if empties > 0:
                tokens.append(str(empties))
                empties = 0
            base = _PIECE_TO_FEN.get(abs(val), "?")
            tokens.append(base.upper() if val > 0 else base)
        if empties > 0:
            tokens.append(str(empties))
        fen_rows.append("".join(tokens))
    return "/".join(fen_rows)


def board_to_uci_fen(state: np.ndarray, side_to_move: str = "w") -> str:
    stm = side_to_move if side_to_move in {"w", "b"} else "w"
    return f"{board_to_fen(state)} {stm} - - 0 1"


def board_to_graphic(state: np.ndarray) -> str:
    lines = ["  " + " ".join(COLS)]
    for row_idx in range(state.shape[0]):
        row_tokens: List[str] = []
        for col_idx in range(state.shape[1]):
            val = int(state[row_idx][col_idx])
            if val == 0:
                row_tokens.append(".")
                continue
            base = _PIECE_TO_FEN.get(abs(val), "?")
            row_tokens.append(base.upper() if val > 0 else base.lower())
        lines.append(f"{row_idx} " + " ".join(row_tokens))
        if row_idx == 4:
            lines.append("  ~~~~~~~~~~~~~~~~~")
    return "\n".join(lines)


def board_coords_to_algebraic(from_row: int, from_col: int, to_row: int, to_col: int) -> str:
    return f"{COLS[from_col]}{from_row}{COLS[to_col]}{to_row}"


def algebraic_to_board_coords(move_str: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if not move_str:
        return None
    match = ALGEBRAIC_RE.match(move_str.strip().lower())
    if not match:
        return None
    from_col = COL_TO_IDX[match.group(1)]
    from_row = int(match.group(2))
    to_col = COL_TO_IDX[match.group(3)]
    to_row = int(match.group(4))
    return (from_row, from_col), (to_row, to_col)


def algebraic_to_engine_move(move_str: str) -> Optional[str]:
    """
    Convert internal board-index algebraic into Pikafish UCI move notation.

    Internal rows: 0 = top (Black/enemy), 9 = bottom (Red/ally).
    Pikafish ranks: 0 = bottom (Red/ally), 9 = top (Black/enemy).
    So we flip: pikafish_rank = 9 - internal_row.
    """
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return None
    (from_row, from_col), (to_row, to_col) = parsed
    return f"{COLS[from_col]}{9 - from_row}{COLS[to_col]}{9 - to_row}"


def algebraic_to_action(move_str: str, board_state: np.ndarray, env: gym.Env) -> Optional[int]:
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return None
    (from_row, from_col), (to_row, to_col) = parsed
    piece_id = int(board_state[from_row][from_col])
    if piece_id <= 0:
        return None
    action = int(move_to_action_space(piece_id, (from_row, from_col), (to_row, to_col)))
    legal = np.where(env.ally_actions == 1)[0]
    if action in legal:
        return action
    return None


def engine_move_to_action(move_str: str, board_state: np.ndarray) -> Optional[int]:
    """Convert a Pikafish UCI move string to an internal action ID.

    Pikafish ranks: 0 = bottom (Red/ally), 9 = top (Black/enemy).
    Internal rows: 0 = top (enemy), 9 = bottom (ally).
    So we flip: numpy_row = 9 - pikafish_rank.
    """
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return None
    (pf_from_row, from_col), (pf_to_row, to_col) = parsed
    from_row = 9 - pf_from_row
    to_row = 9 - pf_to_row
    piece_id = int(board_state[from_row][from_col])
    if piece_id <= 0:
        return None
    return int(move_to_action_space(piece_id, (from_row, from_col), (to_row, to_col)))


def apply_pikafish_legal_mask(
    board_state: np.ndarray,
    env: gym.Env,
    pikafish_evaluator: Optional["PikafishEvaluator"],
) -> Tuple[np.ndarray, bool, int]:
    env_legal_actions = np.where(env.ally_actions == 1)[0]
    if (
        pikafish_evaluator is None
        or not pikafish_evaluator.enabled
        or len(env_legal_actions) == 0
    ):
        return env_legal_actions, False, 0

    fen_before = board_to_uci_fen(board_state, side_to_move="w")
    engine_legal_moves = pikafish_evaluator.list_legal_moves(fen_before)
    if not engine_legal_moves:
        return env_legal_actions, False, 0

    engine_actions: List[int] = []
    for move_str in engine_legal_moves:
        action = engine_move_to_action(move_str, board_state)
        if action is not None:
            engine_actions.append(int(action))

    if not engine_actions:
        return env_legal_actions, False, len(engine_legal_moves)

    unique_actions = np.array(sorted(set(engine_actions)), dtype=int)
    env.ally_actions.fill(0)
    env.ally_actions[unique_actions] = 1
    return unique_actions, True, len(engine_legal_moves)


def action_to_algebraic(action: int) -> str:
    _, start, end = action_space_to_move(int(action))
    return board_coords_to_algebraic(start[0], start[1], end[0], end[1])


def describe_action(action: int) -> str:
    piece_id, start, end = action_space_to_move(int(action))
    piece_name = PIECE_ID_TO_NAME[piece_id] if piece_id < len(PIECE_ID_TO_NAME) else f"piece_{piece_id}"
    alg = action_to_algebraic(action)
    return f"{piece_name} {alg} ({start[0]},{start[1]})->({end[0]},{end[1]})"


# -----------------------------------------------------------------------------
# Reward and candidate evaluation
# -----------------------------------------------------------------------------


@dataclass
class CandidateEval:
    response: str
    move_str: Optional[str]
    action: Optional[int]
    legal: bool
    has_reasoning: bool
    has_format: bool
    capture_value: float
    engine_reward: float
    format_reward: float
    reasoning_quality: float
    reward: float
    cp_before: Optional[float]
    cp_after_raw: Optional[float]
    cp_delta: Optional[float]
    engine_eval_success: bool
    query_ids: torch.Tensor
    response_ids: torch.Tensor


def _extract_move(response: str) -> Optional[str]:
    match = MOVE_RE.search(response or "")
    if not match:
        return None
    return match.group(1).lower()


def _extract_reasoning(response: str) -> bool:
    return THINK_RE.search(response or "") is not None


def _extract_think_text(response: str) -> str:
    match = THINK_CAPTURE_RE.search(response or "")
    if not match:
        return ""
    return match.group(1).strip()


def _reasoning_quality_score(response: str, enemy_move_desc: Optional[str]) -> float:
    think = _extract_think_text(response).lower()
    if not think:
        return 0.0

    score = 0.0
    if len(think.split()) >= 8:
        score += 0.25
    if any(token in think for token in THREAT_KEYWORDS):
        score += 0.25

    enemy_move_str = ""
    if enemy_move_desc:
        m = MOVE_IN_TEXT_RE.search(enemy_move_desc)
        if m:
            enemy_move_str = m.group(1).lower()
    if any(token in think for token in ENEMY_KEYWORDS) or (enemy_move_str and enemy_move_str in think):
        score += 0.25

    if MOVE_IN_TEXT_RE.search(think):
        score += 0.25
    return min(score, 1.0)


def _capture_value_for_move(board_before: np.ndarray, move_str: str) -> float:
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return 0.0
    _, (to_row, to_col) = parsed
    target = int(board_before[to_row][to_col])
    if target < 0:
        return float(PIECE_POINTS[abs(target)])
    return 0.0


def normalize_cp_delta_to_reward(cp_delta: float, cp_scale: float) -> float:
    scale = max(10.0, float(cp_scale))
    reward = 5.5 + 4.5 * math.tanh(cp_delta / scale)
    return float(np.clip(reward, 1.0, 10.0))


class PikafishEvaluator:
    """Communicate with Pikafish using raw (unbuffered) I/O so that
    ``select()`` accurately reflects available data."""

    def __init__(
        self,
        binary_path: str,
        depth: int,
        timeout_sec: float = 2.0,
    ):
        self.binary_path = binary_path
        self.depth = max(1, int(depth))
        self.timeout_sec = max(0.2, float(timeout_sec))
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.enabled = False
        self.eval_file: Optional[str] = None
        self._buf = b""
        self.engine_dir: Optional[str] = None

        resolved = shutil.which(binary_path) if binary_path else None
        if not resolved:
            return
        self.binary_path = resolved
        engine_dir = os.path.dirname(self.binary_path)
        self.engine_dir = engine_dir or None
        eval_candidate = os.path.join(engine_dir, "pikafish.nnue") if engine_dir else ""
        self.eval_file = eval_candidate if eval_candidate and os.path.isfile(eval_candidate) else None
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
            if self.eval_file:
                self._send(f"setoption name EvalFile value {self.eval_file}")
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
        """Read all available lines within *timeout* seconds using raw I/O."""
        deadline = time.time() + timeout
        lines: List[str] = []
        while time.time() < deadline:
            remaining = max(deadline - time.time(), 0)
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 0.05))
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
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 0.05))
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

    def _restart(self) -> bool:
        self.close()
        self._launch()
        return bool(self.enabled and self.proc and self.proc.poll() is None)

    def list_legal_moves(self, fen: str) -> Optional[List[str]]:
        if not self.enabled or not self.proc or self.proc.poll() is not None:
            return None
        for attempt in range(2):
            try:
                if attempt > 0 and not self._restart():
                    return None
                if not self._sync_ready():
                    continue
                self._send(f"position fen {fen}")
                self._send("go perft 1")
                lines = self._read_lines(self.timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._restart()
                    continue
                legal_moves: List[str] = []
                for text in lines:
                    match = ENGINE_PERFT_MOVE_RE.match(text)
                    if match:
                        legal_moves.append(match.group(1).lower())
                if legal_moves:
                    return legal_moves
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
        return None

    def evaluate_cp(self, fen: str, moves: Optional[List[str]] = None) -> Optional[float]:
        if not self.enabled or not self.proc or self.proc.poll() is not None:
            return None
        moves = moves or []
        for attempt in range(2):
            try:
                if attempt > 0 and not self._restart():
                    return None
                if not self._sync_ready():
                    continue

                position_cmd = f"position fen {fen}"
                if moves:
                    position_cmd += " moves " + " ".join(moves)
                self._send(position_cmd)
                self._send(f"go depth {self.depth}")

                latest_score: Optional[float] = None
                lines = self._read_lines(self.timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._restart()
                    continue
                for text in lines:
                    text_lower = text.lower()
                    cp_match = ENGINE_CP_RE.search(text_lower)
                    if cp_match:
                        latest_score = float(cp_match.group(1))
                    mate_match = ENGINE_MATE_RE.search(text_lower)
                    if mate_match:
                        mate_dist = int(mate_match.group(1))
                        sign = 1.0 if mate_dist >= 0 else -1.0
                        latest_score = sign * max(9000.0, 10000.0 - 100.0 * min(abs(mate_dist), 90))
                if latest_score is not None:
                    return latest_score
                self._stop_search()
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
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


def evaluate_candidate_response(
    response: str,
    board_before: np.ndarray,
    env: gym.Env,
    query_ids: torch.Tensor,
    tokenizer,
    enemy_move_desc: Optional[str],
    pikafish_evaluator: Optional[PikafishEvaluator],
    cp_scale: float,
    format_weight: float,
    forced_action: Optional[int] = None,
) -> CandidateEval:
    has_reasoning = _extract_reasoning(response)
    parsed_move_str = _extract_move(response)
    move_str = parsed_move_str
    # Format compliance now focuses on structured <think> output; the move can be
    # selected from legal actions by the harness.
    has_format = has_reasoning

    legal = False
    action = None
    capture_value = 0.0
    engine_reward = 0.0
    format_reward = 0.0
    reasoning_quality = _reasoning_quality_score(response, enemy_move_desc)
    reward = 0.0
    cp_before: Optional[float] = None
    cp_after_raw: Optional[float] = None
    cp_delta: Optional[float] = None
    engine_eval_success = False

    if forced_action is not None:
        action = int(forced_action)
        move_str = action_to_algebraic(action)
    elif move_str is not None:
        action = algebraic_to_action(move_str, board_before, env)

    if action is not None and move_str is not None:
        legal = True
        capture_value = _capture_value_for_move(board_before, move_str)

        # Illegal moves always score 0. Legal moves map to [1, 10].
        engine_reward = 5.5
        if pikafish_evaluator and pikafish_evaluator.enabled:
            fen_before = board_to_uci_fen(board_before, side_to_move="w")
            engine_move = algebraic_to_engine_move(move_str)
            cp_before = pikafish_evaluator.evaluate_cp(fen_before, moves=None)
            cp_after_raw = (
                pikafish_evaluator.evaluate_cp(fen_before, moves=[engine_move]) if engine_move else None
            )
            if cp_before is not None and cp_after_raw is not None:
                cp_after_from_ally = -float(cp_after_raw)
                cp_delta = cp_after_from_ally - float(cp_before)
                engine_reward = normalize_cp_delta_to_reward(cp_delta, cp_scale=cp_scale)
                engine_eval_success = True
        format_subscore = 0.0
        if has_format:
            format_subscore += 0.25
        if has_reasoning:
            format_subscore += 0.25
        format_subscore += 0.5 * reasoning_quality
        format_reward = 1.0 + 9.0 * min(format_subscore, 1.0)
        mix = float(np.clip(format_weight, 0.0, 0.8))
        reward = (1.0 - mix) * engine_reward + mix * format_reward

    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if response_ids.numel() == 0:
        response_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)

    return CandidateEval(
        response=response,
        move_str=move_str,
        action=action,
        legal=legal,
        has_reasoning=has_reasoning,
        has_format=has_format,
        capture_value=capture_value,
        engine_reward=float(engine_reward),
        format_reward=float(format_reward),
        reasoning_quality=float(reasoning_quality),
        reward=reward,
        cp_before=None if cp_before is None else float(cp_before),
        cp_after_raw=None if cp_after_raw is None else float(cp_after_raw),
        cp_delta=None if cp_delta is None else float(cp_delta),
        engine_eval_success=bool(engine_eval_success),
        query_ids=query_ids.cpu(),
        response_ids=response_ids.cpu(),
    )


# -----------------------------------------------------------------------------
# Logging and episode metrics
# -----------------------------------------------------------------------------

SYNC_LOG_FILE = "xiangqi_v2_board_sync.log"
EPISODE_METRICS_CSV = "chinese_chess_episode_metrics_v2.csv"

_EPISODE_METRICS_FIELDNAMES = [
    "episode",
    "rounds",
    "ally_return",
    "enemy_return",
    "total_return",
    "outcome",
    "ally_turns_episode",
    "random_fallback_episode",
    "random_move_rate_episode",
    "game_legal_move_rate",
    "game_format_compliance_rate",
    "game_reasoning_rate",
    "game_mean_capture_value",
    "game_mean_best_candidate_reward",
    "game_move_diversity",
    "grpo_loss",
    "grpo_mean_kl",
    "grpo_mean_reward",
    "mfu",
    "hfu",
    "mfu_step_time_sec",
]


def _fmt_metric(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return repr(val)
    return str(val)


def _fmt_optional_float(val: Optional[float], precision: int = 3) -> str:
    if val is None:
        return "None"
    return f"{float(val):.{precision}f}"


def reset_episode_metrics_csv(filepath: str) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_EPISODE_METRICS_FIELDNAMES).writeheader()


def append_episode_metrics_csv(filepath: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_METRICS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_board_sync(lines: List[str], log_file: str = SYNC_LOG_FILE) -> None:
    payload = "\n".join(lines)
    print(payload)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(payload + "\n\n")


# -----------------------------------------------------------------------------
# MFU helpers
# -----------------------------------------------------------------------------

_GPU_PEAK_TFLOPS: Dict[str, Dict[str, float]] = {
    "5090": {"bf16": 209.5, "fp32": 104.8, "fp16": 209.5},
    "5080": {"bf16": 112.0, "fp32": 56.0, "fp16": 112.0},
    "4090": {"bf16": 330.0, "fp32": 82.6, "fp16": 330.0},
    "h100": {"bf16": 989.0, "fp32": 67.0, "fp16": 989.0},
    "a100": {"bf16": 312.0, "fp32": 19.5, "fp16": 312.0},
}


def _get_gpu_peak_tflops(device_index: int = 0, dtype_str: str = "bf16") -> float:
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(device_index).lower()
    for key, peaks in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return peaks.get(dtype_str, peaks.get("bf16", 50.0))
    return 50.0


def _resolve_compute_dtype_str(mp_policy: Optional[MixedPrecisionPolicy]) -> str:
    if mp_policy is None:
        return "bf16"
    dt = getattr(mp_policy, "param_dtype", None)
    if dt is None:
        return "bf16"
    mapper = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return mapper.get(dt, "bf16")


class MFUTracker:
    def __init__(
        self,
        model,
        device_index: int = 0,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        gradient_checkpointing: bool = True,
    ):
        unwrapped = unwrap_model(model)
        self.total_params = sum(p.numel() for p in unwrapped.parameters())
        self.trainable_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype_str = _resolve_compute_dtype_str(mp_policy)
        self.gpu_peak_tflops = _get_gpu_peak_tflops(device_index, self.dtype_str)
        self.gpu_peak_flops = self.gpu_peak_tflops * 1e12
        self.history: List[Dict[str, float]] = []

    def compute(self, total_tokens: int, elapsed_sec: float, num_fwd_per_sample: int = 2) -> Dict[str, float]:
        if elapsed_sec <= 0 or total_tokens <= 0:
            return {}

        p = self.total_params
        p_t = self.trainable_params
        t = total_tokens
        fwd_flops = num_fwd_per_sample * 2 * p * t
        bwd_act_flops = 2 * p * t
        bwd_w_flops = 2 * p_t * t
        mfu_flops = fwd_flops + bwd_act_flops + bwd_w_flops
        recompute_flops = (2 * p * t) if self.gradient_checkpointing else 0
        hfu_flops = mfu_flops + recompute_flops

        mfu = (mfu_flops / elapsed_sec) / self.gpu_peak_flops if self.gpu_peak_flops > 0 else 0.0
        hfu = (hfu_flops / elapsed_sec) / self.gpu_peak_flops if self.gpu_peak_flops > 0 else 0.0

        stats = {
            "mfu/mfu": float(mfu),
            "mfu/hfu": float(hfu),
            "mfu/mfu_achieved_tflops": float((mfu_flops / elapsed_sec) / 1e12),
            "mfu/hfu_achieved_tflops": float((hfu_flops / elapsed_sec) / 1e12),
            "mfu/peak_tflops": float(self.gpu_peak_tflops),
            "mfu/total_tokens_step": float(total_tokens),
            "mfu/step_time_sec": float(elapsed_sec),
        }
        self.history.append(stats)
        return stats


# -----------------------------------------------------------------------------
# Environment opponent
# -----------------------------------------------------------------------------


class GreedyEnemyAgent:
    def move(self, env: gym.Env) -> int:
        actions = np.where(env.enemy_actions == 1)[0]
        if len(actions) == 0:
            raise RuntimeError("GreedyEnemyAgent: no legal enemy moves")
        board = env.state
        best_score = -1.0
        best_actions: List[int] = []
        for action in actions:
            _, _, end = action_space_to_move(int(action))
            target = int(board[end[0]][end[1]])
            score = float(PIECE_POINTS[target]) if target > 0 else 0.0
            if score > best_score:
                best_score = score
                best_actions = [int(action)]
            elif score == best_score:
                best_actions.append(int(action))
        return int(random.choice(best_actions))


# -----------------------------------------------------------------------------
# GRPO trainer - per turn group updates
# -----------------------------------------------------------------------------


def _build_optimizer(model, lr: float, optimizer_name: str):
    params = (p for p in model.parameters() if p.requires_grad)
    if optimizer_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:
            if rank == 0:
                print("[optimizer] bitsandbytes unavailable, falling back to torch AdamW.")
    return torch.optim.AdamW(params, lr=lr)


class GRPOTrainerOnline:
    def __init__(
        self,
        model,
        tokenizer,
        device_obj: torch.device,
        lr: float,
        beta: float,
        kl_penalty_min: float,
        kl_penalty_max: float,
        max_grad_norm: float,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        optimizer_name: str = "adamw_8bit",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device_obj
        self.beta = beta
        self.kl_penalty_min = kl_penalty_min
        self.kl_penalty_max = kl_penalty_max
        self.max_grad_norm = max_grad_norm
        self.optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name)

        unwrapped = unwrap_model(model)
        if hasattr(unwrapped, "gradient_checkpointing_enable"):
            unwrapped.gradient_checkpointing_enable()

        self.mfu_tracker = MFUTracker(
            model,
            device_index=device_obj.index or 0,
            mp_policy=mp_policy,
            gradient_checkpointing=getattr(unwrapped, "is_gradient_checkpointing", False),
        )

    def _compute_response_log_probs(self, query_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        input_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        logits = self.model(input_ids=input_ids).logits
        response_start = query_ids.size(0)
        response_logits = logits[0, response_start - 1 : -1, :]
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
        return token_log_probs.sum()

    def _toggle_adapters(self, enable: bool) -> None:
        unwrapped = unwrap_model(self.model)
        fn_name = "enable_adapter_layers" if enable else "disable_adapter_layers"
        if hasattr(unwrapped, fn_name):
            getattr(unwrapped, fn_name)()

    def _broadcast_group(
        self,
        query_ids_batch: Optional[List[torch.Tensor]],
        response_ids_batch: Optional[List[torch.Tensor]],
        rewards: Optional[List[float]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        if not dist.is_initialized():
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            return query_ids_batch, response_ids_batch, rewards_t

        if rank == 0:
            n = len(query_ids_batch)
        else:
            n = 0
        n = broadcast_int(n)

        synced_q: List[torch.Tensor] = []
        synced_r: List[torch.Tensor] = []
        for i in range(n):
            if rank == 0:
                q_len = int(query_ids_batch[i].numel())
                r_len = int(response_ids_batch[i].numel())
            else:
                q_len, r_len = 0, 0
            q_len = broadcast_int(q_len)
            r_len = broadcast_int(r_len)

            if rank == 0:
                q = query_ids_batch[i].long().to(device)
                r = response_ids_batch[i].long().to(device)
            else:
                q = torch.zeros(q_len, dtype=torch.long, device=device)
                r = torch.zeros(r_len, dtype=torch.long, device=device)
            dist.broadcast(q, src=0)
            dist.broadcast(r, src=0)
            synced_q.append(q.cpu())
            synced_r.append(r.cpu())

        if rank == 0:
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        else:
            rewards_t = torch.zeros(n, dtype=torch.float32, device=device)
        dist.broadcast(rewards_t, src=0)
        return synced_q, synced_r, rewards_t.cpu()

    def train_group(
        self,
        query_ids_batch: Optional[List[torch.Tensor]],
        response_ids_batch: Optional[List[torch.Tensor]],
        rewards: Optional[List[float]],
    ) -> Dict[str, float]:
        if rank == 0:
            should_train_local = bool(
                query_ids_batch and response_ids_batch and rewards and len(rewards) > 0
            )
        else:
            should_train_local = False

        should_train = broadcast_bool(should_train_local) if dist.is_initialized() else should_train_local
        if not should_train:
            return {}

        if dist.is_initialized():
            query_ids_batch, response_ids_batch, rewards_t = self._broadcast_group(
                query_ids_batch, response_ids_batch, rewards
            )
        else:
            rewards_t = torch.tensor(rewards, dtype=torch.float32)

        reward_std = float(rewards_t.std().item())
        if reward_std > 1e-6:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)
        else:
            advantages = rewards_t - rewards_t.mean()

        total_tokens = sum(q.numel() + r.numel() for q, r in zip(query_ids_batch, response_ids_batch))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_kl = 0.0
        total_kl_penalty = 0.0
        n = len(query_ids_batch)
        samples_ok = 0

        for idx in range(n):
            q = query_ids_batch[idx]
            r = response_ids_batch[idx]
            adv = advantages[idx].to(self.device)
            try:
                with torch.no_grad():
                    self._toggle_adapters(enable=False)
                    ref_log_prob = self._compute_response_log_probs(q, r)
                    self._toggle_adapters(enable=True)
                cur_log_prob = self._compute_response_log_probs(q, r)

                kl_raw = cur_log_prob - ref_log_prob
                # Clamp per-sample KL contribution to avoid runaway policy drift.
                kl_penalty = torch.clamp(kl_raw, min=self.kl_penalty_min, max=self.kl_penalty_max)
                sample_loss = (-adv * cur_log_prob + self.beta * kl_penalty) / n
                sample_loss.backward()

                total_loss += float(sample_loss.item())
                total_kl += float(kl_raw.item())
                total_kl_penalty += float(kl_penalty.item())
                samples_ok += 1
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"[GRPO] CUDA OOM on sample {idx + 1}/{n}, skipping sample.")
                self._toggle_adapters(enable=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if samples_ok == 0:
            self.optimizer.zero_grad()
            return {}

        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.max_grad_norm,
        )
        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        mfu_stats = self.mfu_tracker.compute(
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            num_fwd_per_sample=2,
        )
        mem_alloc = (torch.cuda.memory_allocated(device) / 1e9) if torch.cuda.is_available() else 0.0
        mem_res = (torch.cuda.memory_reserved(device) / 1e9) if torch.cuda.is_available() else 0.0

        stats = {
            "grpo/loss": total_loss,
            "grpo/mean_advantage": float(advantages.mean().item()),
            "grpo/mean_kl": total_kl / max(1, samples_ok),
            "grpo/mean_kl_penalty": total_kl_penalty / max(1, samples_ok),
            "grpo/mean_reward": float(rewards_t.mean().item()),
            "grpo/batch_reward_std": reward_std,
            "grpo/samples_ok": float(samples_ok),
            "grpo/samples_total": float(n),
            "grpo/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "system/gpu_memory_allocated_gb": float(mem_alloc),
            "system/gpu_memory_reserved_gb": float(mem_res),
            "system/step_time_sec": float(elapsed),
        }
        stats.update(mfu_stats)
        return stats


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


@dataclass
class TurnResult:
    action: int
    move_algebraic: str
    used_random_fallback: bool
    chosen_capture_value: float
    best_candidate_reward: float
    chosen_engine_reward: float
    chosen_format_reward: float
    chosen_cp_before: Optional[float]
    chosen_cp_after_raw: Optional[float]
    chosen_cp_delta: Optional[float]
    chosen_engine_eval_success: bool
    candidate_metrics: Dict[str, float]
    train_stats: Dict[str, float]
    chosen_response: str


class XiangqiAgent:
    def __init__(
        self,
        model,
        tokenizer,
        grpo_trainer: GRPOTrainerOnline,
        max_seq_length: int,
        max_prompt_length: int,
        max_train_query_ctx: int,
        generate_config: Dict[str, Any],
        num_generations: int,
        pikafish_evaluator: Optional[PikafishEvaluator],
        reward_cp_scale: float,
        reward_format_weight: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grpo_trainer = grpo_trainer
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.max_train_query_ctx = max_train_query_ctx
        self.generate_config = generate_config
        self.num_generations = num_generations
        self.pikafish_evaluator = pikafish_evaluator
        self.reward_cp_scale = float(reward_cp_scale)
        self.reward_format_weight = float(reward_format_weight)

    def _set_generation_checkpointing(self, enable: bool) -> None:
        model = unwrap_model(self.model)
        if enable:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        else:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

    def _sync_success_all_ranks(self, local_success: bool) -> bool:
        if not dist.is_initialized():
            return local_success
        success_t = torch.tensor([1 if local_success else 0], dtype=torch.long, device=device)
        dist.all_reduce(success_t, op=dist.ReduceOp.MIN)
        return bool(success_t.item())

    def _build_generation_retry_schedule(self) -> List[Tuple[int, int]]:
        max_new_tokens = int(self.generate_config["max_new_tokens"])
        schedules = [
            (self.num_generations, max_new_tokens),
            (max(1, self.num_generations // 2), max_new_tokens),
            (1, max_new_tokens),
            (1, min(max_new_tokens, 128)),
            (1, min(max_new_tokens, 96)),
            (1, min(max_new_tokens, 64)),
        ]
        deduped: List[Tuple[int, int]] = []
        for schedule in schedules:
            if schedule not in deduped:
                deduped.append(schedule)
        return deduped

    def system_prompt(self) -> str:
        return (
            "You are a Xiangqi (Chinese Chess) player.\n"
            "Your pieces are uppercase, enemy pieces are lowercase.\n"
            "K=General A=Advisor B=Elephant N=Horse R=Chariot C=Cannon P=Soldier.\n\n"
            "In <think>, include: enemy's last move impact, your tactical threat, and enemy counterplay risk.\n"
            "Do not output a move; only output tactical reasoning.\n\n"
            "Respond exactly as:\n"
            "<think>brief tactical reasoning</think>"
        )

    def format_turn_prompt(
        self,
        board_state: np.ndarray,
        enemy_move_desc: Optional[str],
    ) -> List[Dict[str, str]]:
        fen = board_to_fen(board_state)
        graphic = board_to_graphic(board_state)
        prefix = f"Enemy previous move: {enemy_move_desc}\n" if enemy_move_desc else "Enemy previous move: none\n"
        user_msg = (
            f"{prefix}"
            f"Current board FEN: {fen}\n"
            f"Current board graphic:\n{graphic}\n"
            "Provide tactical reasoning for the best plan."
        )
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def _generate_candidates(
        self,
        board_state: Optional[np.ndarray],
        enemy_move_desc: Optional[str],
        episode: int,
        round_idx: int,
    ) -> Tuple[Optional[torch.Tensor], List[str], str, str]:
        if rank == 0:
            messages = self.format_turn_prompt(board_state, enemy_move_desc)
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            if encoded.input_ids.size(1) > self.max_prompt_length:
                encoded.input_ids = encoded.input_ids[:, -self.max_prompt_length :]
                encoded.attention_mask = encoded.attention_mask[:, -self.max_prompt_length :]
            query_ids = encoded.input_ids[0].cpu()
            fen = board_to_fen(board_state)
            graphic = board_to_graphic(board_state)
        else:
            encoded = None
            query_ids = None
            fen = ""
            graphic = ""

        generate_model = unwrap_model(self.model)
        generate_model.eval()

        retry_schedule = self._build_generation_retry_schedule()
        decoded: List[str] = []
        context_len = 0
        attempt_used: Optional[Tuple[int, int]] = None

        self._set_generation_checkpointing(enable=False)
        try:
            for attempt_idx, (num_generations, max_new_tokens) in enumerate(retry_schedule, start=1):
                ids_batch, mask_batch, context_len = broadcast_generation_inputs(
                    encoded.input_ids if rank == 0 else None,
                    encoded.attention_mask if rank == 0 else None,
                    num_generations=num_generations,
                )

                local_success = True
                outputs = None
                try:
                    with torch.no_grad():
                        outputs = generate_model.generate(
                            inputs=ids_batch,
                            attention_mask=mask_batch,
                            **{
                                **self.generate_config,
                                "max_new_tokens": max_new_tokens,
                            },
                        )
                except torch.cuda.OutOfMemoryError:
                    local_success = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                success = self._sync_success_all_ranks(local_success)
                if not success:
                    if rank == 0:
                        print(
                            f"[Ep {episode} Rd {round_idx}] Generate OOM on attempt "
                            f"{attempt_idx}/{len(retry_schedule)} with "
                            f"num_generations={num_generations}, max_new_tokens={max_new_tokens}. "
                            "Retrying with a smaller generation load."
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                out_tokens = outputs[:, context_len:]
                decoded = self.tokenizer.batch_decode(
                    out_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                attempt_used = (num_generations, max_new_tokens)
                break
        finally:
            self._set_generation_checkpointing(enable=True)

        if rank == 0:
            if attempt_used is not None:
                print(
                    f"[Ep {episode} Rd {round_idx}] Generated {len(decoded)} candidates "
                    f"(prompt_tokens={context_len}, num_generations={attempt_used[0]}, "
                    f"max_new_tokens={attempt_used[1]})"
                )
            else:
                print(
                    f"[Ep {episode} Rd {round_idx}] Generate failed after all retries; "
                    "falling back to a random legal move."
                )
            return query_ids, decoded, fen, graphic
        return None, [], "", ""

    def act_and_train(
        self,
        board_state: Optional[np.ndarray],
        env: Optional[gym.Env],
        enemy_move_desc: Optional[str],
        episode: int,
        round_idx: int,
    ) -> TurnResult:
        query_ids, responses, fen, graphic = self._generate_candidates(
            board_state=board_state,
            enemy_move_desc=enemy_move_desc,
            episode=episode,
            round_idx=round_idx,
        )

        if rank == 0:
            evals: List[CandidateEval] = []
            legal_count = 0
            format_count = 0
            reasoning_count = 0
            move_strings: List[str] = []
            legal_actions, using_pikafish_legality, engine_legal_count = apply_pikafish_legal_mask(
                board_state=board_state,
                env=env,
                pikafish_evaluator=self.pikafish_evaluator,
            )
            if len(legal_actions) == 0:
                raise RuntimeError("XiangqiAgent: no legal ally moves available")

            for response in responses:
                clipped_query = query_ids
                if clipped_query.numel() > self.max_train_query_ctx:
                    clipped_query = clipped_query[-self.max_train_query_ctx :]
                forced_action = int(np.random.choice(legal_actions))
                ev = evaluate_candidate_response(
                    response=response,
                    board_before=board_state,
                    env=env,
                    query_ids=clipped_query,
                    tokenizer=self.tokenizer,
                    enemy_move_desc=enemy_move_desc,
                    pikafish_evaluator=self.pikafish_evaluator,
                    cp_scale=self.reward_cp_scale,
                    format_weight=self.reward_format_weight,
                    forced_action=forced_action,
                )
                evals.append(ev)
                if ev.legal:
                    legal_count += 1
                if ev.has_format:
                    format_count += 1
                if ev.has_reasoning:
                    reasoning_count += 1
                if ev.move_str is not None:
                    move_strings.append(ev.move_str)

            legal_evals = [ev for ev in evals if ev.legal and ev.action is not None]
            used_random_fallback = False
            chosen_capture = 0.0
            chosen_response = ""
            chosen_eval: Optional[CandidateEval] = None

            if legal_evals:
                best_reward = max(ev.reward for ev in legal_evals)
                best_candidates = [ev for ev in legal_evals if ev.reward == best_reward]
                chosen = random.choice(best_candidates)
                chosen_eval = chosen
                chosen_action = int(chosen.action)
                chosen_move = str(chosen.move_str)
                chosen_capture = float(chosen.capture_value)
                chosen_response = chosen.response
            else:
                chosen_action = int(np.random.choice(legal_actions))
                chosen_move = action_to_algebraic(chosen_action)
                if evals:
                    chosen_response = (
                        "<think>Unable to evaluate generated reasoning, using fallback legal move.</think>"
                    )
                else:
                    chosen_response = (
                        "<think>Generation failed due to CUDA memory pressure, using fallback legal move.</think>"
                    )
                used_random_fallback = True
                best_reward = 0.0

            query_batch = [ev.query_ids for ev in evals]
            response_batch = [ev.response_ids for ev in evals]
            reward_batch = [float(ev.reward) for ev in evals]
            train_stats = self.grpo_trainer.train_group(query_batch, response_batch, reward_batch)
            successful_cp_deltas = [float(ev.cp_delta) for ev in legal_evals if ev.cp_delta is not None]
            mean_cp_delta_success = (
                float(np.mean(successful_cp_deltas)) if successful_cp_deltas else None
            )

            candidate_metrics = {
                "game/legal_move_rate": legal_count / max(1, len(evals)),
                "game/format_compliance_rate": format_count / max(1, len(evals)),
                "game/reasoning_rate": reasoning_count / max(1, len(evals)),
                "game/reasoning_quality_rate": float(np.mean([ev.reasoning_quality for ev in evals]))
                if evals
                else 0.0,
                "game/mean_engine_reward": float(np.mean([ev.engine_reward for ev in legal_evals]))
                if legal_evals
                else 0.0,
                "game/mean_format_reward": float(np.mean([ev.format_reward for ev in legal_evals]))
                if legal_evals
                else 0.0,
                "game/engine_eval_success_rate": float(
                    np.mean([1.0 if ev.engine_eval_success else 0.0 for ev in legal_evals])
                )
                if legal_evals
                else 0.0,
                "game/mean_cp_delta_success": float(mean_cp_delta_success)
                if mean_cp_delta_success is not None
                else 0.0,
                "game/move_diversity": len(set(move_strings)) / max(1, len(evals)),
                "game/mean_best_candidate_reward": float(best_reward),
                "game/using_pikafish_legality": 1.0 if using_pikafish_legality else 0.0,
                "game/engine_legal_action_count": float(len(legal_actions)),
                "game/engine_legal_move_count_raw": float(engine_legal_count),
            }

            log_board_sync(
                [
                    f"[Ep {episode} Rd {round_idx}] === BOARD SYNC CHECK ===",
                    f"FEN: {fen}",
                    "Graphic:",
                    graphic,
                    "API Board (numpy):",
                    np.array2string(board_state),
                    (
                        f"Candidates: {len(evals)} generated, {legal_count} legal, "
                        f"best_reward={best_reward:.4f}"
                    ),
                    (
                        "Legality source: "
                        f"{'pikafish' if using_pikafish_legality else 'env'} "
                        f"(usable_actions={len(legal_actions)}, engine_list_count={engine_legal_count})"
                    ),
                    (
                        "Candidate scoring: "
                        f"mean_engine_reward={candidate_metrics['game/mean_engine_reward']:.4f} "
                        f"mean_format_reward={candidate_metrics['game/mean_format_reward']:.4f} "
                        f"engine_eval_success_rate={candidate_metrics['game/engine_eval_success_rate']:.3f} "
                        f"mean_cp_delta_success={_fmt_optional_float(mean_cp_delta_success)}"
                    ),
                    (
                        f"Chosen move: {chosen_move} "
                        f"= {describe_action(chosen_action)}"
                    ),
                    f"Chosen response: {chosen_response}",
                    (
                        "Chosen scoring: "
                        f"combined_reward={chosen_eval.reward:.4f} "
                        f"engine_reward={chosen_eval.engine_reward:.4f} "
                        f"format_reward={chosen_eval.format_reward:.4f} "
                        f"engine_eval_success={int(chosen_eval.engine_eval_success)} "
                        f"cp_before={_fmt_optional_float(chosen_eval.cp_before)} "
                        f"cp_after_raw={_fmt_optional_float(chosen_eval.cp_after_raw)} "
                        f"cp_delta={_fmt_optional_float(chosen_eval.cp_delta)}"
                    )
                    if chosen_eval is not None
                    else "Chosen scoring: fallback legal move selected, no evaluated candidate chosen.",
                ]
            )

            return TurnResult(
                action=chosen_action,
                move_algebraic=chosen_move,
                used_random_fallback=used_random_fallback,
                chosen_capture_value=chosen_capture,
                best_candidate_reward=float(best_reward),
                chosen_engine_reward=float(chosen_eval.engine_reward) if chosen_eval is not None else 0.0,
                chosen_format_reward=float(chosen_eval.format_reward) if chosen_eval is not None else 0.0,
                chosen_cp_before=None if chosen_eval is None else chosen_eval.cp_before,
                chosen_cp_after_raw=None if chosen_eval is None else chosen_eval.cp_after_raw,
                chosen_cp_delta=None if chosen_eval is None else chosen_eval.cp_delta,
                chosen_engine_eval_success=bool(chosen_eval.engine_eval_success)
                if chosen_eval is not None
                else False,
                candidate_metrics=candidate_metrics,
                train_stats=train_stats,
                chosen_response=chosen_response,
            )

        train_stats = self.grpo_trainer.train_group(None, None, None)
        return TurnResult(
            action=-1,
            move_algebraic="",
            used_random_fallback=False,
            chosen_capture_value=0.0,
            best_candidate_reward=0.0,
            chosen_engine_reward=0.0,
            chosen_format_reward=0.0,
            chosen_cp_before=None,
            chosen_cp_after_raw=None,
            chosen_cp_delta=None,
            chosen_engine_eval_success=False,
            candidate_metrics={},
            train_stats=train_stats,
            chosen_response="",
        )


# -----------------------------------------------------------------------------
# Setup and training
# -----------------------------------------------------------------------------

MODEL_REGISTRY = {
    "7b": "unsloth/Qwen2.5-7B-Instruct",
    "14b": "unsloth/Qwen2.5-14B-Instruct",
}

hyperparams = {
    "model_name": MODEL_REGISTRY[args.model_size],
    "env": "gym_xiangqi:xiangqi-v0",
    "max_seq_length": 512,
    "max_prompt_length": 256,
    "max_train_query_ctx": 256,
    "lora/r": 32,
    "lora/lora_alpha": 32,
    "lora/lora_dropout": 0.0,
    "lora/target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "grpo/num_generations": 6,
    "grpo/lr": 2e-6,
    "grpo/beta": 0.1,
    "grpo/kl_penalty_min": 0.0,
    "grpo/kl_penalty_max": 10.0,
    "grpo/max_grad_norm": 0.1,
    "grpo/optim": "adamw_8bit",
    "generate/max_new_tokens": 200,
    "generate/do_sample": True,
    "generate/temperature": 0.8,
    "generate/top_p": 0.95,
    "episodes": 500,
    "max_rounds_per_episode": 200,
    "seed": 42069,
    "metrics/clear_csv_on_start": True,
    "checkpoint/dir": "./checkpoints/xiangqi_grpo_v2",
    "checkpoint/every_n_episodes": 25,
    "checkpoint/load_adapter_path": "",
    "pikafish/bin": os.environ.get("PIKAFISH_BIN", "/home/fchow/bin/pikafish"),
    "pikafish/depth": 8,
    "pikafish/timeout_sec": 2.0,
    "reward/engine_cp_scale": 250.0,
    "reward/format_weight": 0.2,
}

if args.episodes is not None:
    hyperparams["episodes"] = int(args.episodes)

random.seed(int(hyperparams["seed"]) + rank)
np.random.seed(int(hyperparams["seed"]) + rank)
torch.manual_seed(int(hyperparams["seed"]) + rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(hyperparams["seed"]) + rank)

if rank == 0:
    wandb.init(
        project=os.environ.get("WANDB_PROJECT") or "xiangqi-grpo-v2",
        config=hyperparams,
    )
else:
    wandb.init(mode="disabled")

if rank == 0:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hyperparams["model_name"],
        max_seq_length=hyperparams["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        fast_inference=False,
    )
dist_barrier()
if rank != 0:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hyperparams["model_name"],
        max_seq_length=hyperparams["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        fast_inference=False,
    )
dist_barrier()

adapter_dir = (hyperparams.get("checkpoint/load_adapter_path") or "").strip()
if adapter_dir:
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter path is not a directory: {adapter_dir!r}")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    if rank == 0:
        print(f"[checkpoint] Loaded adapter from {adapter_dir!r}")
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(hyperparams["lora/r"]),
        lora_alpha=int(hyperparams["lora/lora_alpha"]),
        lora_dropout=float(hyperparams["lora/lora_dropout"]),
        target_modules=list(hyperparams["lora/target_modules"]),
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

mesh = None
mp_policy = None
fsdp_kwargs = {}
if args.mixed_precision:
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fsdp_kwargs["mp_policy"] = mp_policy

if dist.is_initialized() and not args.use_ddp and world_size > 1:
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
    # Handle model layout differences by searching for the transformer block list.
    layer_list = None
    candidate_paths = [
        ["base_model", "model", "model", "layers"],
        ["model", "layers"],
        ["layers"],
    ]
    for path in candidate_paths:
        cur = model
        ok = True
        for key in path:
            if not hasattr(cur, key):
                ok = False
                break
            cur = getattr(cur, key)
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            layer_list = cur
            break
    if layer_list is None:
        raise RuntimeError("Could not find transformer layers for fully_shard on unsloth model.")
    for layer in layer_list:
        fully_shard(layer, mesh=mesh, **fsdp_kwargs)
    fully_shard(model, mesh=mesh, **fsdp_kwargs)
elif dist.is_initialized() and not args.use_ddp and world_size == 1 and rank == 0:
    print("[setup] Skipping FSDP because WORLD_SIZE=1; using plain single-GPU model.")
elif args.use_ddp and dist.is_initialized():
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


def count_trainable_params(model_obj):
    total = sum(p.numel() for p in model_obj.parameters())
    trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


trainable_params, total_params, pct_trainable = count_trainable_params(model)
if rank == 0:
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({pct_trainable:.2f}%)")
    print(f"Model: {hyperparams['model_name']}")


def save_lora_checkpoint(
    model_obj,
    tokenizer_obj,
    checkpoint_path: str,
    episode: int,
    label: str = "",
) -> None:
    unwrapped = unwrap_model(model_obj)
    if dist.is_initialized():
        full_sd = get_model_state_dict(
            unwrapped,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        full_sd = unwrapped.state_dict()

    if rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)
        unwrapped.save_pretrained(checkpoint_path, state_dict=full_sd)
        tokenizer_obj.save_pretrained(checkpoint_path)
        meta = {
            "episode": int(episode),
            "label": label or None,
            "base_model": hyperparams["model_name"],
            "lora": {
                "r": hyperparams["lora/r"],
                "lora_alpha": hyperparams["lora/lora_alpha"],
                "lora_dropout": hyperparams["lora/lora_dropout"],
                "target_modules": hyperparams["lora/target_modules"],
            },
        }
        with open(os.path.join(checkpoint_path, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[checkpoint] Saved LoRA checkpoint to {checkpoint_path!r}")
    dist_barrier()


grpo_trainer = GRPOTrainerOnline(
    model=model,
    tokenizer=tokenizer,
    device_obj=device,
    lr=float(hyperparams["grpo/lr"]),
    beta=float(hyperparams["grpo/beta"]),
    kl_penalty_min=float(hyperparams["grpo/kl_penalty_min"]),
    kl_penalty_max=float(hyperparams["grpo/kl_penalty_max"]),
    max_grad_norm=float(hyperparams["grpo/max_grad_norm"]),
    mp_policy=mp_policy,
    optimizer_name=str(hyperparams["grpo/optim"]),
)

generate_config = {
    "max_new_tokens": int(hyperparams["generate/max_new_tokens"]),
    "do_sample": bool(hyperparams["generate/do_sample"]),
    "temperature": float(hyperparams["generate/temperature"]),
    "top_p": float(hyperparams["generate/top_p"]),
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

pikafish_evaluator: Optional[PikafishEvaluator] = None
if rank == 0:
    pikafish_evaluator = PikafishEvaluator(
        binary_path=str(hyperparams["pikafish/bin"]),
        depth=int(hyperparams["pikafish/depth"]),
        timeout_sec=float(hyperparams["pikafish/timeout_sec"]),
    )
    if pikafish_evaluator.enabled:
        print(
            f"[reward] Pikafish enabled at {pikafish_evaluator.binary_path!r} "
            f"(depth={hyperparams['pikafish/depth']})."
        )
    else:
        print(
            "[reward] Pikafish unavailable; using neutral legal-move reward baseline only. "
            "Set PIKAFISH_BIN to enable engine-scored rewards."
        )

ally_agent = XiangqiAgent(
    model=model,
    tokenizer=tokenizer,
    grpo_trainer=grpo_trainer,
    max_seq_length=int(hyperparams["max_seq_length"]),
    max_prompt_length=int(hyperparams["max_prompt_length"]),
    max_train_query_ctx=int(hyperparams["max_train_query_ctx"]),
    generate_config=generate_config,
    num_generations=int(hyperparams["grpo/num_generations"]),
    pikafish_evaluator=pikafish_evaluator if rank == 0 else None,
    reward_cp_scale=float(hyperparams["reward/engine_cp_scale"]),
    reward_format_weight=float(hyperparams["reward/format_weight"]),
)
enemy_agent = GreedyEnemyAgent()


env = None
if rank == 0:
    env = gym.make(hyperparams["env"])
    if hyperparams.get("metrics/clear_csv_on_start", True):
        reset_episode_metrics_csv(EPISODE_METRICS_CSV)
        with open(SYNC_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")

_SIGNAL_ALLY_TURN = 1
_SIGNAL_EPISODE_DONE = 2

episodes = int(hyperparams["episodes"])
max_rounds = int(hyperparams["max_rounds_per_episode"])
ckpt_root = hyperparams["checkpoint/dir"]
ckpt_every = int(hyperparams["checkpoint/every_n_episodes"])

ally_wins = 0
enemy_wins = 0
truncated_games = 0
lifetime_ally_turns = 0
lifetime_random_fallback = 0
global_train_steps = 0

try:
    for episode in range(1, episodes + 1):
        if rank == 0:
            observation = env.reset()
            done = False
            round_idx = 1
            ally_reward_terminal = 0.0
            enemy_reward_terminal = 0.0
            ally_return = 0.0
            enemy_return = 0.0
            enemy_move_desc_for_prompt: Optional[str] = None
            ally_turns_episode = 0
            random_fallback_episode = 0
            legal_rate_series: List[float] = []
            format_rate_series: List[float] = []
            reasoning_rate_series: List[float] = []
            capture_series: List[float] = []
            best_reward_series: List[float] = []
            diversity_series: List[float] = []
            engine_eval_success_series: List[float] = []
            chosen_engine_reward_series: List[float] = []
            chosen_format_reward_series: List[float] = []
            chosen_cp_delta_series: List[float] = []
            train_stats_last: Dict[str, float] = {}

            print(f"\n[Ep {episode}] Opponent: GreedyEnemy")

        while True:
            if rank == 0:
                while not done and env.turn != ALLY:
                    board_before_enemy = env.state.copy()
                    enemy_action = enemy_agent.move(env)
                    observation, enemy_reward, done, _ = env.step(enemy_action)
                    enemy_return += float(enemy_reward)
                    enemy_reward_terminal = float(enemy_reward)

                    enemy_move_desc_for_prompt = describe_action(enemy_action)
                    log_board_sync(
                        [
                            f"[Ep {episode} Rd {round_idx}] Enemy move: {enemy_move_desc_for_prompt}",
                            f"Enemy board_before FEN: {board_to_fen(board_before_enemy)}",
                            "Enemy board_before graphic:",
                            board_to_graphic(board_before_enemy),
                            "Enemy board_after numpy:",
                            np.array2string(env.state),
                        ]
                    )

                    round_idx += 1
                    if round_idx >= max_rounds and not done:
                        done = True
                        truncated_games += 1
                        break

                signal = _SIGNAL_EPISODE_DONE if done else _SIGNAL_ALLY_TURN
            else:
                signal = 0

            signal = broadcast_int(signal)
            if signal == _SIGNAL_EPISODE_DONE:
                break

            if rank == 0:
                board_before_ally = env.state.copy()
                turn_result = ally_agent.act_and_train(
                    board_state=board_before_ally,
                    env=env,
                    enemy_move_desc=enemy_move_desc_for_prompt,
                    episode=episode,
                    round_idx=round_idx,
                )
                enemy_move_desc_for_prompt = None
            else:
                turn_result = ally_agent.act_and_train(
                    board_state=None,
                    env=None,
                    enemy_move_desc=None,
                    episode=episode,
                    round_idx=round_idx,
                )

            if rank == 0:
                global_train_steps += 1
                ally_turns_episode += 1
                if turn_result.used_random_fallback:
                    random_fallback_episode += 1

                observation, ally_reward, done, _ = env.step(turn_result.action)
                ally_return += float(ally_reward)
                ally_reward_terminal = float(ally_reward)

                legal_rate_series.append(float(turn_result.candidate_metrics["game/legal_move_rate"]))
                format_rate_series.append(float(turn_result.candidate_metrics["game/format_compliance_rate"]))
                reasoning_rate_series.append(float(turn_result.candidate_metrics["game/reasoning_rate"]))
                diversity_series.append(float(turn_result.candidate_metrics["game/move_diversity"]))
                best_reward_series.append(float(turn_result.best_candidate_reward))
                capture_series.append(float(turn_result.chosen_capture_value))
                engine_eval_success_series.append(
                    float(turn_result.candidate_metrics["game/engine_eval_success_rate"])
                )
                chosen_engine_reward_series.append(float(turn_result.chosen_engine_reward))
                chosen_format_reward_series.append(float(turn_result.chosen_format_reward))
                if turn_result.chosen_cp_delta is not None:
                    chosen_cp_delta_series.append(float(turn_result.chosen_cp_delta))

                train_stats_last = turn_result.train_stats
                if turn_result.train_stats:
                    step_payload = dict(turn_result.train_stats)
                    step_payload.update(
                        {
                            "train/global_step": global_train_steps,
                            "episode": episode,
                            "round": round_idx,
                            "train/candidate_legal_rate": turn_result.candidate_metrics["game/legal_move_rate"],
                            "train/candidate_format_rate": turn_result.candidate_metrics["game/format_compliance_rate"],
                            "train/candidate_reasoning_rate": turn_result.candidate_metrics["game/reasoning_rate"],
                            "train/candidate_reasoning_quality_rate": turn_result.candidate_metrics[
                                "game/reasoning_quality_rate"
                            ],
                            "train/candidate_move_diversity": turn_result.candidate_metrics["game/move_diversity"],
                            "train/candidate_mean_engine_reward": turn_result.candidate_metrics[
                                "game/mean_engine_reward"
                            ],
                            "train/candidate_mean_format_reward": turn_result.candidate_metrics[
                                "game/mean_format_reward"
                            ],
                            "train/candidate_engine_eval_success_rate": turn_result.candidate_metrics[
                                "game/engine_eval_success_rate"
                            ],
                            "train/candidate_mean_cp_delta_success": turn_result.candidate_metrics[
                                "game/mean_cp_delta_success"
                            ],
                            "train/using_pikafish_legality": turn_result.candidate_metrics[
                                "game/using_pikafish_legality"
                            ],
                            "train/engine_legal_action_count": turn_result.candidate_metrics[
                                "game/engine_legal_action_count"
                            ],
                            "train/engine_legal_move_count_raw": turn_result.candidate_metrics[
                                "game/engine_legal_move_count_raw"
                            ],
                            "train/best_candidate_reward": turn_result.best_candidate_reward,
                            "train/chosen_capture_value": turn_result.chosen_capture_value,
                            "train/chosen_engine_reward": turn_result.chosen_engine_reward,
                            "train/chosen_format_reward": turn_result.chosen_format_reward,
                            "train/chosen_engine_eval_success": float(
                                turn_result.chosen_engine_eval_success
                            ),
                        }
                    )
                    if turn_result.chosen_cp_before is not None:
                        step_payload["train/chosen_cp_before"] = float(turn_result.chosen_cp_before)
                    if turn_result.chosen_cp_after_raw is not None:
                        step_payload["train/chosen_cp_after_raw"] = float(
                            turn_result.chosen_cp_after_raw
                        )
                    if turn_result.chosen_cp_delta is not None:
                        step_payload["train/chosen_cp_delta"] = float(turn_result.chosen_cp_delta)
                    wandb.log(step_payload)

                round_idx += 1
                if round_idx >= max_rounds and not done:
                    done = True
                    truncated_games += 1

        if rank == 0:
            lifetime_ally_turns += ally_turns_episode
            lifetime_random_fallback += random_fallback_episode

            if enemy_reward_terminal == 100:
                enemy_wins += 1
            elif ally_reward_terminal == 100:
                ally_wins += 1

            legal_move_rate = float(np.mean(legal_rate_series)) if legal_rate_series else 0.0
            format_rate = float(np.mean(format_rate_series)) if format_rate_series else 0.0
            reasoning_rate = float(np.mean(reasoning_rate_series)) if reasoning_rate_series else 0.0
            move_diversity = float(np.mean(diversity_series)) if diversity_series else 0.0
            mean_capture = float(np.mean(capture_series)) if capture_series else 0.0
            mean_best_reward = float(np.mean(best_reward_series)) if best_reward_series else 0.0
            mean_engine_eval_success_rate = (
                float(np.mean(engine_eval_success_series)) if engine_eval_success_series else 0.0
            )
            mean_chosen_engine_reward = (
                float(np.mean(chosen_engine_reward_series)) if chosen_engine_reward_series else 0.0
            )
            mean_chosen_format_reward = (
                float(np.mean(chosen_format_reward_series)) if chosen_format_reward_series else 0.0
            )
            mean_chosen_cp_delta = (
                float(np.mean(chosen_cp_delta_series)) if chosen_cp_delta_series else None
            )
            random_rate_episode = (
                100.0 * random_fallback_episode / ally_turns_episode if ally_turns_episode else 0.0
            )

            outcome = "other"
            if enemy_reward_terminal == 100:
                outcome = "enemy_win"
            elif ally_reward_terminal == 100:
                outcome = "ally_win"
            elif round_idx >= max_rounds:
                outcome = "truncated_cap"

            episode_stats = {
                "episode": episode,
                "game/episode_length": round_idx,
                "game/ally_return": ally_return,
                "game/enemy_return": enemy_return,
                "game/total_return": ally_return + enemy_return,
                "game/ally_win_rate": (100.0 * ally_wins / episode),
                "game/enemy_win_rate": (100.0 * enemy_wins / episode),
                "game/truncated_rate": (100.0 * truncated_games / episode),
                "game/ally_turns_episode": ally_turns_episode,
                "game/random_fallback_episode": random_fallback_episode,
                "game/random_move_rate_episode": random_rate_episode,
                "game/random_move_rate_lifetime": (
                    100.0 * lifetime_random_fallback / lifetime_ally_turns
                    if lifetime_ally_turns
                    else 0.0
                ),
                "game/legal_move_rate": legal_move_rate,
                "game/format_compliance_rate": format_rate,
                "game/reasoning_rate": reasoning_rate,
                "game/mean_capture_value": mean_capture,
                "game/mean_best_candidate_reward": mean_best_reward,
                "game/move_diversity": move_diversity,
                "game/engine_eval_success_rate": mean_engine_eval_success_rate,
                "game/mean_chosen_engine_reward": mean_chosen_engine_reward,
                "game/mean_chosen_format_reward": mean_chosen_format_reward,
            }
            if mean_chosen_cp_delta is not None:
                episode_stats["game/mean_chosen_cp_delta"] = mean_chosen_cp_delta
            wandb.log(episode_stats)

            csv_row = {
                "episode": episode,
                "rounds": round_idx,
                "ally_return": round(ally_return, 6),
                "enemy_return": round(enemy_return, 6),
                "total_return": round(ally_return + enemy_return, 6),
                "outcome": outcome,
                "ally_turns_episode": ally_turns_episode,
                "random_fallback_episode": random_fallback_episode,
                "random_move_rate_episode": round(random_rate_episode, 6),
                "game_legal_move_rate": round(legal_move_rate, 6),
                "game_format_compliance_rate": round(format_rate, 6),
                "game_reasoning_rate": round(reasoning_rate, 6),
                "game_mean_capture_value": round(mean_capture, 6),
                "game_mean_best_candidate_reward": round(mean_best_reward, 6),
                "game_move_diversity": round(move_diversity, 6),
                "grpo_loss": _fmt_metric(train_stats_last.get("grpo/loss")),
                "grpo_mean_kl": _fmt_metric(train_stats_last.get("grpo/mean_kl")),
                "grpo_mean_reward": _fmt_metric(train_stats_last.get("grpo/mean_reward")),
                "mfu": _fmt_metric(train_stats_last.get("mfu/mfu")),
                "hfu": _fmt_metric(train_stats_last.get("mfu/hfu")),
                "mfu_step_time_sec": _fmt_metric(train_stats_last.get("mfu/step_time_sec")),
            }
            append_episode_metrics_csv(EPISODE_METRICS_CSV, csv_row)

            print(
                f"[Ep {episode}] ally_return={ally_return:.2f} enemy_return={enemy_return:.2f} "
                f"ally_win_rate={episode_stats['game/ally_win_rate']:.1f}% "
                f"enemy_win_rate={episode_stats['game/enemy_win_rate']:.1f}% "
                f"legal_rate={legal_move_rate:.3f} format_rate={format_rate:.3f} "
                f"reasoning_rate={reasoning_rate:.3f} mean_capture={mean_capture:.3f} "
                f"engine_eval_success_rate={mean_engine_eval_success_rate:.3f} "
                f"mean_chosen_engine_reward={mean_chosen_engine_reward:.3f} "
                f"mean_chosen_format_reward={mean_chosen_format_reward:.3f} "
                f"mean_chosen_cp_delta={_fmt_optional_float(mean_chosen_cp_delta)}"
            )

        if ckpt_every > 0 and episode % ckpt_every == 0:
            save_lora_checkpoint(
                model_obj=model,
                tokenizer_obj=tokenizer,
                checkpoint_path=os.path.join(ckpt_root, f"ep_{episode}"),
                episode=episode,
                label=f"every_{ckpt_every}",
            )

    if rank == 0:
        save_lora_checkpoint(
            model_obj=model,
            tokenizer_obj=tokenizer,
            checkpoint_path=os.path.join(ckpt_root, "final"),
            episode=episodes,
            label="normal_completion",
        )
except Exception:
    if rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING CRASHED")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
    try:
        crash_ep = episode  # noqa: F821
    except Exception:
        crash_ep = None
    if crash_ep is not None and crash_ep >= 1:
        try:
            save_lora_checkpoint(
                model_obj=model,
                tokenizer_obj=tokenizer,
                checkpoint_path=os.path.join(ckpt_root, f"interrupted_ep{crash_ep}"),
                episode=crash_ep,
                label="interrupted",
            )
        except Exception as save_err:
            if rank == 0:
                print(f"[checkpoint] interrupted save failed: {save_err}")
    raise
finally:
    if rank == 0 and pikafish_evaluator is not None:
        pikafish_evaluator.close()
    if env is not None:
        env.close()
    wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
