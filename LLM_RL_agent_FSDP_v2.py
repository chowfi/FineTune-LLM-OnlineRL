"""
export PIKAFISH_BIN=/home/fchow/bin/pikafish
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
from collections import OrderedDict
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
MOVE_TAG_RE = re.compile(r"Move:", flags=re.IGNORECASE)
ALGEBRAIC_RE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])$")
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
ENGINE_CP_RE = re.compile(r"\bscore\s+cp\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_MATE_RE = re.compile(r"\bscore\s+mate\s+(-?\d+)\b", flags=re.IGNORECASE)
ENGINE_PERFT_MOVE_RE = re.compile(r"^([a-i][0-9][a-i][0-9]):\s+\d+\b", flags=re.IGNORECASE)
MOVE_IN_TEXT_RE = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", flags=re.IGNORECASE)


def _find_region_token_indices(
    tokenizer,
    response_ids: torch.Tensor,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Locate the `<think>`/`</think>` and last `Move:` spans in a response.

    Returns (think_start, think_end, move_start) as token indices within
    ``response_ids``. ``think_end`` is exclusive (token index just past the
    closing tag). Returns ``None`` for any span that could not be located.
    """
    r_list = response_ids.tolist()
    r_len = len(r_list)
    if r_len == 0:
        return None, None, None
    try:
        text = tokenizer.decode(r_list, skip_special_tokens=False)
    except Exception:
        return None, None, None

    def _tok_idx(char_pos: int) -> Optional[int]:
        if char_pos < 0:
            return None
        prefix = text[:char_pos]
        if not prefix:
            return 0
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        except Exception:
            return None
        return max(0, min(len(prefix_ids), r_len))

    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    move_matches = list(MOVE_TAG_RE.finditer(text))
    move_char = move_matches[-1].start() if move_matches else -1
    think_start = _tok_idx(think_open) if think_open >= 0 else None
    think_end = (
        _tok_idx(think_close + len(THINK_CLOSE_TAG)) if think_close >= 0 else None
    )
    move_start = _tok_idx(move_char) if move_char >= 0 else None
    return think_start, think_end, move_start


def _build_region_masks_padded(
    region_indices: List[Tuple[Optional[int], Optional[int], Optional[int]]],
    resp_lens: torch.Tensor,
    max_total: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align per-response region indices to the shifted log-prob tensor layout.

    The shifted tensor has ``T = max_total - 1`` columns and response token
    ``k`` (0-indexed) of sample ``i`` lives at shifted column
    ``max_total - 1 - r_len_i + k``. Returns ``(think_mask, move_mask)`` of
    shape ``(G, T)``, both ``dtype``-valued floats.
    """
    G = len(region_indices)
    T = max(0, max_total - 1)
    think_mask = torch.zeros((G, T), device=device, dtype=dtype)
    move_mask = torch.zeros((G, T), device=device, dtype=dtype)
    if T == 0:
        return think_mask, move_mask
    for i, (ts, te, ms) in enumerate(region_indices):
        r_len_i = int(resp_lens[i].item())
        if r_len_i <= 0:
            continue
        base = max_total - 1 - r_len_i
        if ts is not None and te is not None and te > ts:
            a = base + max(0, ts)
            b = base + min(r_len_i, te)
            if b > a:
                think_mask[i, a:b] = 1.0
        if ms is not None and ms < r_len_i:
            a = base + max(0, ms)
            b = base + r_len_i
            if b > a:
                move_mask[i, a:b] = 1.0
    return think_mask, move_mask

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
    parsed_move_ok: bool
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


def _reasoning_quality_score(
    response: str,
    enemy_move_desc: Optional[str],
    chosen_move: Optional[str] = None,
    chosen_piece_name: Optional[str] = None,
) -> float:
    """Score how well the ``<think>`` block justifies the move that will be played.

    The old implementation awarded points for generic keyword presence and any
    move-like token, which the model easily gamed without actually describing
    its own move. The new version only gives full credit when the reasoning
    references the move (UCI string or source/target square or the piece type
    name) that the candidate will actually execute.
    """
    think = _extract_think_text(response).lower()
    if not think:
        return 0.0

    score = 0.0
    if len(think.split()) >= 12:
        score += 0.2
    if any(token in think for token in THREAT_KEYWORDS):
        score += 0.15

    enemy_move_str = ""
    if enemy_move_desc:
        m = MOVE_IN_TEXT_RE.search(enemy_move_desc)
        if m:
            enemy_move_str = m.group(1).lower()
    if any(token in think for token in ENEMY_KEYWORDS) or (
        enemy_move_str and enemy_move_str in think
    ):
        score += 0.15

    # Big bucket: does the reasoning actually describe the move being played?
    if chosen_move:
        cm = chosen_move.lower()
        from_sq, to_sq = cm[:2], cm[2:]
        if cm in think:
            score += 0.35
        elif from_sq in think and to_sq in think:
            score += 0.25
        elif from_sq in think or to_sq in think:
            score += 0.1
    else:
        # No move parsed from the response - no credit for referring to moves.
        score += 0.0

    if chosen_piece_name and chosen_piece_name.lower() in think:
        score += 0.15

    return float(min(score, 1.0))


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


def adaptive_cp_scale(base_scale: float, cp_before: Optional[float]) -> float:
    """Stretch the tanh normalisation by how extreme the current position is.

    In a dead-lost or dead-won position the absolute |cp_before| can be several
    thousand centipawns (including mate-inflated values near 9000-10000). With a
    fixed scale of 250 the tanh saturates and every candidate move receives the
    same reward, destroying the advantage signal. We widen the scale
    proportionally to ``|cp_before|`` but cap it at ``2 * base_scale`` so it
    never becomes so wide that every move in a lost position looks identical to
    the engine reward.
    """
    base = max(10.0, float(base_scale))
    if cp_before is None:
        return base
    widened = max(base, 0.2 * abs(float(cp_before)))
    return min(widened, 2.0 * base)


class PikafishEvaluator:
    """Communicate with Pikafish using raw (unbuffered) I/O so that
    ``select()`` accurately reflects available data.

    Results are cached by ``(fen, moves_tuple)`` for ``evaluate_cp`` and by
    ``fen`` for ``list_legal_moves`` so that the 16 candidates in a round that
    share the same ``cp_before`` position only trigger one engine search.
    The ``legal_moves`` call uses ``go perft 1`` (no depth search) while
    ``evaluate_cp`` uses ``eval_depth`` for a higher-quality positional score.
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
        verbose: bool = True,
    ):
        self.binary_path = binary_path
        self.depth = max(1, int(depth))
        # Short timeout for handshake-style commands: uciok, readyok, stop.
        self.timeout_sec = max(0.2, float(timeout_sec))
        # Use ``go movetime`` instead of ``go depth`` so per-candidate eval cost
        # is bounded by wall-clock. The old ``go depth N`` path could block
        # far longer than the read timeout on hard mid-game positions, which
        # caused every candidate of a round to time out in cascade.
        if movetime_ms is None:
            movetime_ms = max(200, 80 * self.depth)
        self.movetime_ms = max(50, int(movetime_ms))
        # Read deadline for search output. Must be comfortably larger than
        # ``movetime_ms`` so we always see ``bestmove`` on clean shutdown.
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
        self._eval_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], float]" = OrderedDict()
        self._legal_cache: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict()
        self._eval_cache_max = max(0, int(eval_cache_size))
        self._legal_cache_max = max(0, int(legal_cache_size))
        # Negative cache prevents the 24 candidates of a round from each
        # hammering the engine when one query fails: after any failure the
        # key is remembered for ``negative_cache_ttl_sec`` seconds.
        self._negative_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], float]" = OrderedDict()
        self._negative_cache_ttl = max(0.0, float(negative_cache_ttl_sec))
        self._negative_cache_max = max(0, int(negative_cache_max))
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
            # UCI: must send `uci` and wait for `uciok` before `setoption`. With EvalFile,
            # setoption-before-uci leaves Pikafish not responding to `isready` (timeouts → disabled).
            self._send("uci")
            if not self._read_until(lambda line: "uciok" in line.lower()):
                self.close()
                return
            if self.eval_file:
                # Pikafish can hang on `isready` after EvalFile is set to an absolute path; cwd is
                # engine_dir so a basename (e.g. pikafish.nnue) is enough and responds quickly.
                ev_arg = (
                    os.path.basename(self.eval_file)
                    if self.engine_dir
                    and os.path.dirname(os.path.abspath(self.eval_file))
                    == os.path.abspath(self.engine_dir)
                    else self.eval_file
                )
                self._send(f"setoption name EvalFile value {ev_arg}")
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
        """Top-guard auto-heal: if the subprocess has died, try to restart it
        once before giving up. Prevents a single engine crash from silently
        disabling engine rewards for the rest of training.
        """
        if self.proc and self.proc.poll() is None:
            return self.enabled
        # Process is absent or exited: attempt one restart.
        return self._restart(reason or "process exited")

    def _health_check_or_restart(self, reason: str = "") -> bool:
        """Probe the engine with ``isready``. If unresponsive, restart. Call
        this after any failed query so the *next* candidate in the round
        starts from a healthy engine rather than a stalled one.
        """
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
            # Expired entry: drop it and let the caller retry.
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
                # perft 1 is a move enumeration; depth search is not required for legality.
                self._send("go perft 1")
                lines = self._read_lines(self.timeout_sec)
                if any("critical error" in text.lower() for text in lines):
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
        # Last-ditch health restart so the next call isn't against a stalled engine.
        self._health_check_or_restart("legal_moves exhausted")
        self._negative_cache_put(neg_key)
        return None

    def evaluate_cp(self, fen: str, moves: Optional[List[str]] = None) -> Optional[float]:
        moves_tuple: Tuple[str, ...] = tuple(moves or ())
        cache_key = (fen, moves_tuple)
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
                # Bounded wall-clock search: Pikafish emits ``bestmove`` after
                # ~movetime_ms. Reading up to ``eval_timeout_sec`` (> movetime)
                # guarantees we capture a final ``info ... score cp`` line.
                self._send(f"go movetime {self.movetime_ms}")

                latest_score: Optional[float] = None
                lines = self._read_lines(self.eval_timeout_sec)
                if any("critical error" in text.lower() for text in lines):
                    self._restart("critical error on eval")
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
                    self._cache_put_eval(cache_key, latest_score)
                    return latest_score
                # No score captured: make sure a stale ``go`` isn't still
                # running, then fall through to the next attempt.
                self._stop_search()
            except Exception:
                self.enabled = False
            if attempt == 0:
                self.enabled = True
        # Prevent the remaining 23 candidates of the round from each paying
        # the same timeout cost, and heal the engine for the next round.
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
    """Evaluate a single LLM candidate.

    The model now emits both ``<think>`` reasoning and ``Move: <uci>`` on its own.
    ``forced_action`` is only honoured when the LLM failed to produce a parseable
    legal move; in that case we still evaluate the fallback move but heavily
    discount the reward so GRPO continues to push the policy toward self-picked
    moves. Illegal/unparseable candidates receive a zero reward which gives
    strong learning signal away from malformed outputs.
    """
    has_reasoning = _extract_reasoning(response)
    parsed_move_str = _extract_move(response)
    has_format = has_reasoning and parsed_move_str is not None

    legal = False
    parsed_move_ok = False
    action: Optional[int] = None
    move_str: Optional[str] = parsed_move_str
    used_forced = False
    capture_value = 0.0
    engine_reward = 0.0
    format_reward = 0.0
    reward = 0.0
    cp_before: Optional[float] = None
    cp_after_raw: Optional[float] = None
    cp_delta: Optional[float] = None
    engine_eval_success = False

    if parsed_move_str is not None:
        candidate_action = algebraic_to_action(parsed_move_str, board_before, env)
        if candidate_action is not None:
            action = int(candidate_action)
            move_str = parsed_move_str
            parsed_move_ok = True
    if action is None and forced_action is not None:
        action = int(forced_action)
        move_str = action_to_algebraic(action)
        used_forced = True

    chosen_piece_name: Optional[str] = None
    if action is not None:
        piece_id, _, _ = action_space_to_move(int(action))
        if 0 <= piece_id < len(PIECE_ID_TO_NAME):
            chosen_piece_name = PIECE_ID_TO_NAME[piece_id].split("_")[0]

    reasoning_quality = _reasoning_quality_score(
        response,
        enemy_move_desc,
        chosen_move=move_str,
        chosen_piece_name=chosen_piece_name,
    )

    if action is not None and move_str is not None:
        legal = True
        capture_value = _capture_value_for_move(board_before, move_str)

        # Baseline neutral engine reward; tanh of cp_delta overrides when the
        # engine is available.
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
                effective_scale = adaptive_cp_scale(cp_scale, cp_before)
                engine_reward = normalize_cp_delta_to_reward(cp_delta, cp_scale=effective_scale)
                engine_eval_success = True

        # Format reward: mostly reasoning quality now that the move is tied into
        # the scorer. We still require the <think> tags and the Move: line for
        # maximum credit.
        format_subscore = 0.0
        if has_reasoning:
            format_subscore += 0.2
        if parsed_move_ok:
            format_subscore += 0.2
        format_subscore += 0.6 * reasoning_quality
        format_reward = 1.0 + 9.0 * float(min(format_subscore, 1.0))

        mix = float(np.clip(format_weight, 0.0, 0.8))
        reward = (1.0 - mix) * engine_reward + mix * format_reward

        # Discourage relying on the forced fallback: give only 30% of the
        # reward when the LLM didn't produce a parseable legal move of its own.
        if used_forced:
            reward *= 0.3

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
        parsed_move_ok=parsed_move_ok,
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
    "game_cp_saturation_truncated",
    "game_mean_chosen_cp_delta_raw",
    "game_mean_chosen_cp_delta_clipped",
    "grpo_loss",
    "grpo_mean_kl",
    "grpo_mean_kl_per_token",
    "grpo_mean_kl_think",
    "grpo_mean_kl_move",
    "grpo_policy_entropy_move",
    "grpo_pg_clip_frac",
    "grpo_ratio_mean",
    "grpo_ppo_epochs_completed",
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
            "mfu/mfu_flops_step": float(mfu_flops),
            "mfu/hfu_flops_step": float(hfu_flops),
        }
        self.history.append(stats)
        return stats

    def generation_flops(
        self,
        num_sequences: int,
        prompt_len: int,
        generated_len: int,
    ) -> float:
        """Rough FLOP estimate for an autoregressive generate() call.

        Uses the standard 2P-per-token approximation: prefill costs ~2P per
        prompt token and each decoded token costs ~2P (KV-cache amortizes the
        attention-to-past, we do not count it separately here). Returns a
        single scalar summed over all sequences in the batch.
        """
        if num_sequences <= 0 or (prompt_len + generated_len) <= 0:
            return 0.0
        return float(
            2.0 * float(self.total_params) * float(num_sequences) * float(prompt_len + generated_len)
        )


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
        max_grad_norm: float,
        ppo_epochs: int = 2,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.28,
        entropy_coef_move: float = 0.0,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        optimizer_name: str = "adamw_8bit",
        logprob_micro_batch: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device_obj
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.clip_eps_low = float(clip_eps_low)
        self.clip_eps_high = float(clip_eps_high)
        self.entropy_coef_move = float(entropy_coef_move)
        self.logprob_micro_batch = max(1, int(logprob_micro_batch))
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

    def _compute_response_log_probs(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Per-token response log-probs for a single sample.

        Returns ``(token_lp, resp_mask, r_len)`` where ``token_lp`` and
        ``resp_mask`` both have shape ``(r_len,)``. ``resp_mask`` is all-ones
        here (no padding for a single sequence) but is returned for API
        parity with the batched helper.
        """
        input_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        logits = self.model(input_ids=input_ids).logits
        response_start = query_ids.size(0)
        response_logits = logits[0, response_start - 1 : -1, :]
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
        r_len = int(response_ids.numel())
        resp_mask = torch.ones(r_len, dtype=token_log_probs.dtype, device=self.device)
        return token_log_probs, resp_mask, r_len

    def _compute_response_log_probs_batch(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        move_mask: Optional[torch.Tensor] = None,
        return_move_entropy: bool = False,
    ) -> Dict[str, Any]:
        """Per-token response log-probs (and optional move-region entropy) in one forward.

        Sequences are left-padded to the common length so every response ends
        at column ``max_total - 1`` and can be sliced uniformly. Padding is
        masked out via ``attention_mask`` so it never contributes gradients.

        Returns a dict with:
          - ``token_lp``: ``(G, max_total - 1)`` per-token log-probs of the
            target tokens (zero outside the response span).
          - ``resp_mask``: ``(G, max_total - 1)`` float mask, 1 on response
            token predictions.
          - ``resp_lens``: ``(G,)`` int tensor of response lengths.
          - ``max_total``: ``int`` common padded sequence length.
          - ``move_entropy_mean``: ``(G,)`` or ``None``. Per-sample mean
            entropy over ``move_mask`` positions if ``return_move_entropy``
            and ``move_mask`` is non-empty.
        """
        G = len(queries)
        q_lens = [int(q.numel()) for q in queries]
        r_lens = [int(r.numel()) for r in responses]
        max_total = max(q + r for q, r in zip(q_lens, r_lens))
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = torch.full((G, max_total), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((G, max_total), dtype=torch.long, device=self.device)
        resp_tokens_padded = torch.zeros((G, max_total), dtype=torch.long, device=self.device)
        resp_lens = torch.tensor(r_lens, dtype=torch.long, device=self.device)

        for i, (q, r) in enumerate(zip(queries, responses)):
            seq = torch.cat([q, r]).to(self.device, dtype=torch.long)
            L = seq.numel()
            input_ids[i, max_total - L:] = seq
            attention_mask[i, max_total - L:] = 1
            resp_tokens_padded[i, max_total - r.numel():] = r.to(self.device, dtype=torch.long)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Shift by one so that column ``t-1`` predicts token at position ``t``.
        shifted = logits[:, :-1, :]
        log_probs = F.log_softmax(shifted.float(), dim=-1)
        # Align target tokens with the shifted positions.
        target = resp_tokens_padded[:, 1:]
        token_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (G, max_total - 1)
        # ``shifted[p]`` predicts the token at absolute position ``p + 1``. The
        # response occupies positions ``[max_total - r_len, max_total - 1]`` so
        # the relevant shifted indices are ``[max_total - r_len - 1, max_total - 2]``.
        pos = torch.arange(max_total - 1, device=self.device).unsqueeze(0)  # (1, L-1)
        resp_pred_start = (max_total - resp_lens - 1).unsqueeze(1)  # (G, 1)
        response_mask = (pos >= resp_pred_start).to(token_lp.dtype)
        token_lp = token_lp * response_mask

        move_entropy_mean: Optional[torch.Tensor] = None
        if return_move_entropy and move_mask is not None and move_mask.numel() > 0:
            mm_bool = move_mask.to(dtype=torch.bool, device=self.device)
            # Only honor move tokens that are inside the response span.
            mm_bool = mm_bool & response_mask.bool()
            counts = mm_bool.sum(dim=1)  # (G,)
            if int(counts.sum().item()) > 0:
                selected_logits = shifted[mm_bool]  # (N_sel, V)
                sel_logp = F.log_softmax(selected_logits.float(), dim=-1)
                sel_p = sel_logp.exp()
                entropy_selected = -(sel_p * sel_logp).sum(dim=-1)  # (N_sel,)
                sample_idx = (
                    torch.arange(G, device=self.device)
                    .unsqueeze(1)
                    .expand_as(mm_bool)[mm_bool]
                )
                per_sample_sum = torch.zeros(
                    G, device=self.device, dtype=entropy_selected.dtype
                )
                per_sample_sum.scatter_add_(0, sample_idx, entropy_selected)
                move_entropy_mean = per_sample_sum / counts.clamp_min(1).to(
                    per_sample_sum.dtype
                )
            else:
                move_entropy_mean = torch.zeros(
                    G, device=self.device, dtype=token_lp.dtype
                )
        elif return_move_entropy:
            move_entropy_mean = torch.zeros(G, device=self.device, dtype=token_lp.dtype)

        return {
            "token_lp": token_lp,
            "resp_mask": response_mask,
            "resp_lens": resp_lens,
            "max_total": max_total,
            "move_entropy_mean": move_entropy_mean,
        }

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

        n = len(query_ids_batch)
        advantages_dev = advantages.to(self.device)

        # Pre-compute region (think / move) token indices per sample once up
        # front: these depend only on the response text and are reused across
        # every PPO epoch.
        region_indices_all: List[Tuple[Optional[int], Optional[int], Optional[int]]] = [
            _find_region_token_indices(self.tokenizer, r) for r in response_ids_batch
        ]

        # -----------------------------------------------------------------
        # Phase 1: precompute (ref_tok_lp, cur_tok_lp_old) per micro-batch.
        # Both are detached — they anchor KL (to reference) and the PPO
        # importance ratio (to the pre-update policy).
        # -----------------------------------------------------------------
        micro = max(1, int(self.logprob_micro_batch))

        def _precompute_caches(micro_size: int) -> Optional[List[Dict[str, Any]]]:
            cache: List[Dict[str, Any]] = []
            try:
                for s in range(0, n, micro_size):
                    e = min(s + micro_size, n)
                    qs = [query_ids_batch[i] for i in range(s, e)]
                    rs = [response_ids_batch[i] for i in range(s, e)]
                    region_slice = region_indices_all[s:e]
                    with torch.no_grad():
                        self._toggle_adapters(enable=False)
                        ref_out = self._compute_response_log_probs_batch(qs, rs)
                        self._toggle_adapters(enable=True)
                        cur_old_out = self._compute_response_log_probs_batch(qs, rs)
                    max_total = ref_out["max_total"]
                    resp_lens = ref_out["resp_lens"]
                    think_mask, move_mask = _build_region_masks_padded(
                        region_slice,
                        resp_lens,
                        max_total,
                        self.device,
                        dtype=ref_out["token_lp"].dtype,
                    )
                    cache.append(
                        {
                            "start": s,
                            "end": e,
                            "qs": qs,
                            "rs": rs,
                            "ref_tok_lp": ref_out["token_lp"].detach(),
                            "cur_tok_lp_old": cur_old_out["token_lp"].detach(),
                            "resp_mask": ref_out["resp_mask"],
                            "resp_lens": resp_lens,
                            "think_mask": think_mask,
                            "move_mask": move_mask,
                            "max_total": max_total,
                        }
                    )
                return cache
            except torch.cuda.OutOfMemoryError:
                self._toggle_adapters(enable=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None

        caches: Optional[List[Dict[str, Any]]] = None
        oom_retries_left = 3
        while oom_retries_left >= 0:
            caches = _precompute_caches(micro)
            if caches is not None:
                break
            if rank == 0:
                print(
                    f"[GRPO] CUDA OOM during precompute at micro={micro}; halving and retrying."
                )
            if micro <= 1:
                break
            micro = max(1, micro // 2)
            oom_retries_left -= 1

        use_sequential_fallback = caches is None

        # -----------------------------------------------------------------
        # Phase 2: PPO epochs over cached micro-batches.
        # -----------------------------------------------------------------
        total_loss = 0.0
        total_kl_per_token_sum = 0.0
        total_kl_think_sum = 0.0
        total_kl_move_sum = 0.0
        total_entropy_move_sum = 0.0
        total_ratio_sum = 0.0
        total_clipped_tokens = 0.0
        total_response_tokens = 0.0
        total_think_tokens = 0.0
        total_move_tokens = 0.0
        total_entropy_samples = 0
        samples_ok = 0
        epochs_completed = 0

        eps_low = self.clip_eps_low
        eps_high = self.clip_eps_high
        beta = self.beta
        ent_coef = self.entropy_coef_move

        if not use_sequential_fallback and caches is not None:
            need_entropy = ent_coef != 0.0
            for epoch in range(self.ppo_epochs):
                self.optimizer.zero_grad()
                epoch_samples_ok = 0
                epoch_oom = False
                for entry in caches:
                    try:
                        fwd = self._compute_response_log_probs_batch(
                            entry["qs"],
                            entry["rs"],
                            move_mask=entry["move_mask"],
                            return_move_entropy=need_entropy,
                        )
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if rank == 0:
                            print(
                                "[GRPO] CUDA OOM during PPO epoch forward; falling back to sequential path."
                            )
                        epoch_oom = True
                        break
                    cur_tok_lp = fwd["token_lp"]
                    resp_mask = entry["resp_mask"]
                    resp_lens_fl = entry["resp_lens"].to(cur_tok_lp.dtype).clamp_min(1.0)
                    ref_tok_lp = entry["ref_tok_lp"]
                    cur_old = entry["cur_tok_lp_old"]
                    adv_slice = advantages_dev[entry["start"]:entry["end"]]

                    # Importance ratio against the pre-update policy (PPO).
                    ratio_raw = torch.exp(cur_tok_lp - cur_old)
                    ratio = ratio_raw * resp_mask  # zero outside response

                    adv_tok = adv_slice.unsqueeze(1).expand_as(ratio)
                    surr1 = ratio * adv_tok
                    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv_tok
                    pg_tok = -torch.min(surr1, surr2)
                    pg_per_sample = (pg_tok * resp_mask).sum(dim=1) / resp_lens_fl

                    # k3 KL estimator against the (frozen) reference policy.
                    diff = ref_tok_lp - cur_tok_lp  # log(pi_ref / pi_cur)
                    kl_tok = torch.exp(diff) - diff - 1.0
                    kl_tok = kl_tok * resp_mask
                    kl_per_sample = kl_tok.sum(dim=1) / resp_lens_fl

                    # Entropy bonus on the Move: region.
                    move_entropy_mean = fwd["move_entropy_mean"]
                    if move_entropy_mean is None:
                        entropy_bonus_per_sample = torch.zeros_like(pg_per_sample)
                        move_entropy_for_log = torch.zeros_like(pg_per_sample)
                    else:
                        move_entropy_for_log = move_entropy_mean
                        entropy_bonus_per_sample = -ent_coef * move_entropy_mean

                    loss_per = pg_per_sample + beta * kl_per_sample + entropy_bonus_per_sample
                    group_loss = loss_per.sum() / n
                    group_loss.backward()

                    with torch.no_grad():
                        total_loss += float(group_loss.item()) / max(1, self.ppo_epochs)
                        # Per-token mean KL across all response tokens.
                        total_kl_per_token_sum += float(kl_tok.sum().item())
                        total_response_tokens += float(resp_mask.sum().item())
                        # Per-region KL.
                        think_mask_e = entry["think_mask"]
                        move_mask_e = entry["move_mask"]
                        if think_mask_e.sum() > 0:
                            total_kl_think_sum += float((kl_tok * think_mask_e).sum().item())
                            total_think_tokens += float(think_mask_e.sum().item())
                        if move_mask_e.sum() > 0:
                            total_kl_move_sum += float((kl_tok * move_mask_e).sum().item())
                            total_move_tokens += float(move_mask_e.sum().item())
                        # Entropy accumulator (per-sample mean, averaged across batch).
                        if move_entropy_mean is not None:
                            total_entropy_move_sum += float(move_entropy_for_log.sum().item())
                            total_entropy_samples += int(move_entropy_mean.numel())
                        # Ratio + clip fractions (mask outside-response entries).
                        mask_bool = resp_mask.bool()
                        ratio_in_mask = ratio_raw[mask_bool]
                        if ratio_in_mask.numel() > 0:
                            total_ratio_sum += float(ratio_in_mask.sum().item())
                            clipped_mask = (ratio_in_mask < (1.0 - eps_low)) | (
                                ratio_in_mask > (1.0 + eps_high)
                            )
                            total_clipped_tokens += float(clipped_mask.sum().item())

                    epoch_samples_ok += (entry["end"] - entry["start"])

                if epoch_oom:
                    use_sequential_fallback = True
                    break

                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                epochs_completed += 1
                samples_ok = max(samples_ok, epoch_samples_ok)

        if use_sequential_fallback:
            # Fallback path: fully sequential, per-sample, single epoch, no
            # PPO clip (OOM-safe). Keeps the k3 KL estimator for consistency.
            if rank == 0:
                print("[GRPO] Falling back to fully sequential per-sample log-prob pass.")
            self.optimizer.zero_grad()
            total_loss = 0.0
            total_kl_per_token_sum = 0.0
            total_response_tokens = 0.0
            samples_ok = 0
            for idx in range(n):
                q = query_ids_batch[idx]
                r = response_ids_batch[idx]
                adv = advantages_dev[idx]
                try:
                    with torch.no_grad():
                        self._toggle_adapters(enable=False)
                        ref_tok_lp, _, r_len = self._compute_response_log_probs(q, r)
                        self._toggle_adapters(enable=True)
                    cur_tok_lp, _, _ = self._compute_response_log_probs(q, r)
                    r_len_fl = max(1.0, float(r_len))
                    diff = ref_tok_lp - cur_tok_lp
                    kl_tok = torch.exp(diff) - diff - 1.0
                    kl_per_sample = kl_tok.sum() / r_len_fl
                    cur_lp_mean = cur_tok_lp.sum() / r_len_fl
                    sample_loss = (-adv * cur_lp_mean + beta * kl_per_sample) / n
                    sample_loss.backward()
                    total_loss += float(sample_loss.item())
                    total_kl_per_token_sum += float(kl_tok.sum().item())
                    total_response_tokens += float(r_len)
                    samples_ok += 1
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"[GRPO] CUDA OOM on sample {idx + 1}/{n}; skipping.")
                    self._toggle_adapters(enable=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            if samples_ok > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                epochs_completed = 1

        if samples_ok == 0:
            self.optimizer.zero_grad()
            return {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # MFU accounting: ``epochs_completed`` training-forwards + 2 no-grad
        # forwards for the ref / cur_old caches when using the batched path.
        num_fwd_per_sample = (
            int(epochs_completed) + 2 if not use_sequential_fallback else 2
        )
        mfu_stats = self.mfu_tracker.compute(
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            num_fwd_per_sample=max(2, num_fwd_per_sample),
        )
        mem_alloc = (torch.cuda.memory_allocated(device) / 1e9) if torch.cuda.is_available() else 0.0
        mem_res = (torch.cuda.memory_reserved(device) / 1e9) if torch.cuda.is_available() else 0.0

        mean_kl_per_token = total_kl_per_token_sum / max(1.0, total_response_tokens)
        mean_kl_think = total_kl_think_sum / total_think_tokens if total_think_tokens > 0 else 0.0
        mean_kl_move = total_kl_move_sum / total_move_tokens if total_move_tokens > 0 else 0.0
        mean_entropy_move = (
            total_entropy_move_sum / total_entropy_samples
            if total_entropy_samples > 0
            else 0.0
        )
        ratio_tokens_seen = float(total_response_tokens)
        ratio_mean = total_ratio_sum / ratio_tokens_seen if ratio_tokens_seen > 0 else 0.0
        pg_clip_frac = (
            total_clipped_tokens / ratio_tokens_seen if ratio_tokens_seen > 0 else 0.0
        )

        stats = {
            "grpo/loss": total_loss,
            "grpo/mean_advantage": float(advantages.mean().item()),
            "grpo/mean_kl": mean_kl_per_token,
            "grpo/mean_kl_per_token": mean_kl_per_token,
            "grpo/mean_kl_think": mean_kl_think,
            "grpo/mean_kl_move": mean_kl_move,
            "grpo/policy_entropy_move": mean_entropy_move,
            "grpo/pg_clip_frac": pg_clip_frac,
            "grpo/ratio_mean": ratio_mean,
            "grpo/ppo_epochs_completed": float(epochs_completed),
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
    generation_stats: Dict[str, float]


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
        reward_format_weight_min: float = 0.05,
        reward_format_weight_anneal_start: float = 0.9,
        reward_format_weight_anneal_end: float = 0.98,
        reward_format_compliance_ema_alpha: float = 0.1,
        min_legal_candidates: int = 0,
        max_regeneration_rounds: int = 0,
        regeneration_batch_size: int = 0,
        min_distinct_legal_moves: int = 0,
        dedupe_legal_by_move: bool = True,
        regen_generate_overrides: Optional[Dict[str, Any]] = None,
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
        self.reward_format_weight_min = float(reward_format_weight_min)
        self.reward_format_weight_anneal_start = float(reward_format_weight_anneal_start)
        self.reward_format_weight_anneal_end = float(reward_format_weight_anneal_end)
        self.reward_format_compliance_ema_alpha = float(reward_format_compliance_ema_alpha)
        self.min_legal_candidates = max(0, int(min_legal_candidates))
        self.max_regeneration_rounds = max(0, int(max_regeneration_rounds))
        self.regeneration_batch_size = (
            max(1, int(regeneration_batch_size))
            if regeneration_batch_size and regeneration_batch_size > 0
            else max(1, int(num_generations) // 2)
        )
        self.min_distinct_legal_moves = max(0, int(min_distinct_legal_moves))
        self.dedupe_legal_by_move = bool(dedupe_legal_by_move)
        # Extra generate(...) kwargs applied only on regeneration passes
        # (pass_idx > 0). Typically widens temperature / top_p to boost
        # exploration once the initial pass lands in a low-diversity basin.
        self.regen_generate_overrides: Dict[str, Any] = dict(regen_generate_overrides or {})
        # Start the EMA at 0 so the full format_weight is used until the model
        # has demonstrated consistent compliance across multiple rounds.
        self._format_compliance_ema: float = 0.0
        self._format_compliance_ema_initialized: bool = False
        self._last_generation_stats: Dict[str, float] = {
            "num_sequences": 0.0,
            "prompt_len": 0.0,
            "generated_len": 0.0,
            "wall_sec": 0.0,
        }

    def _current_format_weight(self) -> float:
        """Linearly anneal format_weight toward ``reward_format_weight_min`` as
        the rolling (EMA) per-round format-compliance rate grows past
        ``reward_format_weight_anneal_start`` and down to
        ``reward_format_weight_min`` at ``reward_format_weight_anneal_end``.
        """
        start = self.reward_format_weight_anneal_start
        end = self.reward_format_weight_anneal_end
        high = self.reward_format_weight
        low = max(0.0, self.reward_format_weight_min)
        if end <= start or high <= low:
            return high
        compliance = self._format_compliance_ema
        if compliance <= start:
            return high
        if compliance >= end:
            return low
        # Linear interpolation between (start, high) and (end, low)
        t = (compliance - start) / (end - start)
        return float(high + t * (low - high))

    def _update_format_compliance_ema(self, round_format_rate: float) -> None:
        rate = float(np.clip(round_format_rate, 0.0, 1.0))
        alpha = float(np.clip(self.reward_format_compliance_ema_alpha, 1e-4, 1.0))
        if not self._format_compliance_ema_initialized:
            self._format_compliance_ema = rate
            self._format_compliance_ema_initialized = True
        else:
            self._format_compliance_ema = (
                (1.0 - alpha) * self._format_compliance_ema + alpha * rate
            )

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
            "You are a Xiangqi (Chinese Chess) player. You always play the uppercase side.\n"
            "Piece letters: K=General A=Advisor B=Elephant N=Horse R=Chariot C=Cannon P=Soldier.\n"
            "Coordinates: files a-i (left to right), ranks 0-9 (top to bottom as shown in the graphic).\n"
            "Uppercase (your) pieces start on the BOTTOM (ranks 7-9); lowercase enemy pieces start on the TOP\n"
            "(ranks 0-2). The river sits between ranks 4 and 5 (shown as '~~~' in the graphic).\n"
            "A move is written <from_file><from_rank><to_file><to_rank>, e.g. b7b4 moves the piece on b7 to b4.\n\n"
            "XIANGQI MOVEMENT RULES (these are NOT the same as Western chess):\n"
            "- K (General): moves 1 step orthogonally, must stay in the palace (files d-f, ranks 7-9 for you).\n"
            "  The two Generals may NEVER face each other on the same file with nothing between them\n"
            "  (flying-general rule), so never expose your K on an open file facing the enemy k.\n"
            "- A (Advisor): moves exactly 1 step diagonally, must stay in the palace (d-f, ranks 7-9).\n"
            "- B (Elephant): moves exactly 2 steps diagonally (e.g. c7 to a5 or e5). Cannot cross the river\n"
            "  (your B must stay on ranks 5-9). Blocked if the 1-step diagonal midpoint is occupied\n"
            "  (the 'elephant eye').\n"
            "- N (Horse): moves 1 step orthogonal + 1 step diagonal outward (L-shape, 8 possible targets).\n"
            "  BLOCKED if the orthogonal-adjacent square ('horse leg') is occupied. A horse does NOT\n"
            "  jump like a Western knight.\n"
            "- R (Chariot): slides any number of empty squares orthogonally, exactly like a Western rook.\n"
            "- C (Cannon): moves like R when NOT capturing, but to CAPTURE it must jump over exactly one\n"
            "  piece (of either side) between itself and the target ('screen'). Non-capture cannon moves\n"
            "  must be along empty squares with no piece in between.\n"
            "- P (Soldier): before crossing the river (your P on ranks 5-9) it moves 1 step forward only\n"
            "  (forward = decreasing rank number for you). After crossing the river (ranks 0-4) it may\n"
            "  also move 1 step sideways. A soldier NEVER moves backward and NEVER moves diagonally.\n\n"
            "IMPORTANT - DO NOT use Western chess concepts that do not apply here. In particular:\n"
            "- There is NO queen, NO bishop-pair, NO pawn promotion, NO castling, NO en-passant.\n"
            "- Do NOT talk about bishops (B here is an Elephant with very different movement), knights in\n"
            "  the Western sense (N is a Horse that can be leg-blocked), or rooks beyond the R=Chariot slide.\n"
            "- Only reason about the board using the Xiangqi rules above and the legal-move list provided.\n"
            "- Before committing to a move, mentally verify it matches the movement rule for that piece and\n"
            "  appears in the Legal moves list.\n\n"
            "In <think>, briefly state: (1) the impact of the enemy's last move, (2) the piece you will move\n"
            "and why (citing the Xiangqi rule it uses), (3) the enemy's most dangerous reply.\n"
            "Then output exactly one legal move on its own line.\n\n"
            "Respond exactly in this format (two lines, nothing else):\n"
            "<think>your tactical reasoning referring to the piece and squares of the move you will play</think>\n"
            "Move: <from><to>"
        )

    def format_turn_prompt(
        self,
        board_state: np.ndarray,
        enemy_move_desc: Optional[str],
        legal_moves_hint: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        fen = board_to_fen(board_state)
        graphic = board_to_graphic(board_state)
        prefix = f"Enemy previous move: {enemy_move_desc}\n" if enemy_move_desc else "Enemy previous move: none\n"
        # Hint only a trimmed list to avoid ballooning prompt tokens.
        hint_line = ""
        if legal_moves_hint:
            trimmed = legal_moves_hint[:48]
            more = "" if len(legal_moves_hint) <= 48 else f" (+{len(legal_moves_hint) - 48} more)"
            hint_line = f"Legal moves (subset): {' '.join(trimmed)}{more}\n"
        user_msg = (
            f"{prefix}"
            f"Current board FEN: {fen}\n"
            f"Current board graphic:\n{graphic}\n"
            f"{hint_line}"
            "Pick the single best legal move for the uppercase side and output reasoning + move."
        )
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def _generate_one_batch(
        self,
        encoded,
        num_generations: int,
        max_new_tokens_override: Optional[int],
        episode: int,
        round_idx: int,
        pass_label: str,
        generate_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[str]], int, int, int, float, Optional[Tuple[int, int]]]:
        """Run a single generation pass with OOM retry.

        Returns ``(decoded, context_len, num_sequences_used, generated_len, wall_sec,
        attempt_used)``. ``decoded`` is ``None`` if every OOM retry failed.
        """
        generate_model = unwrap_model(self.model)
        retry_schedule = self._build_generation_retry_schedule()
        base_max_new = (
            int(max_new_tokens_override)
            if max_new_tokens_override is not None
            else int(self.generate_config["max_new_tokens"])
        )
        # Rebuild schedule anchored on the requested generation budget.
        anchored_schedule: List[Tuple[int, int]] = [
            (num_generations, base_max_new),
        ]
        for _, tok_cap in retry_schedule:
            if tok_cap < base_max_new:
                anchored_schedule.append((max(1, num_generations), tok_cap))
            else:
                anchored_schedule.append((max(1, num_generations), base_max_new))
            anchored_schedule.append((max(1, num_generations // 2), tok_cap))
            anchored_schedule.append((1, tok_cap))

        deduped: List[Tuple[int, int]] = []
        for item in anchored_schedule:
            if item not in deduped:
                deduped.append(item)

        total_wall_sec = 0.0
        context_len_used = 0
        for attempt_idx, (ng, max_new) in enumerate(deduped, start=1):
            ids_batch, mask_batch, context_len = broadcast_generation_inputs(
                encoded.input_ids if rank == 0 else None,
                encoded.attention_mask if rank == 0 else None,
                num_generations=ng,
            )
            context_len_used = context_len

            local_success = True
            outputs = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            attempt_start = time.perf_counter()
            try:
                with torch.no_grad():
                    merged_cfg: Dict[str, Any] = {
                        **self.generate_config,
                        "max_new_tokens": max_new,
                    }
                    if generate_override:
                        merged_cfg.update(generate_override)
                    outputs = generate_model.generate(
                        inputs=ids_batch,
                        attention_mask=mask_batch,
                        **merged_cfg,
                    )
            except torch.cuda.OutOfMemoryError:
                local_success = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_wall_sec += time.perf_counter() - attempt_start

            success = self._sync_success_all_ranks(local_success)
            if not success:
                if rank == 0:
                    print(
                        f"[Ep {episode} Rd {round_idx}] ({pass_label}) Generate OOM on attempt "
                        f"{attempt_idx}/{len(deduped)} with num_generations={ng}, "
                        f"max_new_tokens={max_new}. Retrying with a smaller load."
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            out_tokens = outputs[:, context_len:]
            decoded = self.tokenizer.batch_decode(
                out_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return (
                decoded,
                context_len_used,
                int(outputs.shape[0]),
                int(out_tokens.shape[1]),
                total_wall_sec,
                (ng, max_new),
            )

        return None, context_len_used, 0, 0, total_wall_sec, None

    def _generate_candidates(
        self,
        board_state: Optional[np.ndarray],
        env: Optional[gym.Env],
        enemy_move_desc: Optional[str],
        episode: int,
        round_idx: int,
        legal_moves_hint: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[str], str, str, int]:
        if rank == 0:
            messages = self.format_turn_prompt(
                board_state, enemy_move_desc, legal_moves_hint=legal_moves_hint
            )
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

        all_decoded: List[str] = []
        total_num_sequences = 0
        # Accumulate num_seq * generated_len so that a weighted-average
        # generated_len keeps the FLOPs calc identical to summing each pass.
        total_gen_token_product = 0
        total_wall_sec = 0.0
        context_len = 0
        legal_so_far = 0
        regen_passes = 0
        distinct_legal_moves: set[str] = set()

        self._set_generation_checkpointing(enable=False)
        try:
            pass_idx = 0
            while True:
                if pass_idx == 0:
                    num_generations = self.num_generations
                    pass_label = "initial"
                    generate_override: Optional[Dict[str, Any]] = None
                else:
                    num_generations = self.regeneration_batch_size
                    pass_label = f"regen {pass_idx}"
                    generate_override = (
                        dict(self.regen_generate_overrides)
                        if self.regen_generate_overrides
                        else None
                    )

                decoded_batch, ctx_len_pass, n_seq, gen_len, wall_sec, attempt_used = (
                    self._generate_one_batch(
                        encoded=encoded,
                        num_generations=num_generations,
                        max_new_tokens_override=None,
                        episode=episode,
                        round_idx=round_idx,
                        pass_label=pass_label,
                        generate_override=generate_override,
                    )
                )
                context_len = ctx_len_pass
                total_wall_sec += wall_sec

                if decoded_batch is None:
                    if rank == 0:
                        print(
                            f"[Ep {episode} Rd {round_idx}] ({pass_label}) Generate failed "
                            "after all retries."
                        )
                    break

                all_decoded.extend(decoded_batch)
                total_num_sequences += n_seq
                total_gen_token_product += n_seq * gen_len
                if pass_idx > 0:
                    regen_passes += 1

                # Rank-0 cheap legality check so we can decide whether to regen.
                batch_legal = 0
                batch_distinct_new = 0
                if rank == 0:
                    for resp in decoded_batch:
                        move_str = _extract_move(resp)
                        if move_str is None or env is None or board_state is None:
                            continue
                        act = algebraic_to_action(move_str, board_state, env)
                        if act is not None:
                            batch_legal += 1
                            if move_str not in distinct_legal_moves:
                                distinct_legal_moves.add(move_str)
                                batch_distinct_new += 1
                    legal_so_far += batch_legal
                    print(
                        f"[Ep {episode} Rd {round_idx}] ({pass_label}) Generated "
                        f"{len(decoded_batch)} candidates ({batch_legal} legal, "
                        f"{batch_distinct_new} new distinct). "
                        f"Totals: candidates={len(all_decoded)}, "
                        f"legal={legal_so_far}, distinct_legal={len(distinct_legal_moves)}, "
                        f"regen_passes={regen_passes}"
                    )
                    legal_threshold_met = (
                        self.min_legal_candidates <= 0
                        or legal_so_far >= self.min_legal_candidates
                    )
                    distinct_threshold_met = (
                        self.min_distinct_legal_moves <= 0
                        or len(distinct_legal_moves) >= self.min_distinct_legal_moves
                    )
                    should_stop = (
                        (legal_threshold_met and distinct_threshold_met)
                        or regen_passes >= self.max_regeneration_rounds
                    )
                else:
                    should_stop = False
                should_stop = broadcast_bool(should_stop, src=0)

                pass_idx += 1
                if should_stop:
                    break
        finally:
            self._set_generation_checkpointing(enable=True)

        effective_gen_len = (
            float(total_gen_token_product) / float(total_num_sequences)
            if total_num_sequences > 0
            else 0.0
        )
        self._last_generation_stats = {
            "num_sequences": float(total_num_sequences),
            "prompt_len": float(context_len),
            "generated_len": effective_gen_len,
            "wall_sec": float(total_wall_sec),
        }

        if rank == 0:
            return query_ids, all_decoded, fen, graphic, legal_so_far
        return None, [], "", "", 0

    def act_and_train(
        self,
        board_state: Optional[np.ndarray],
        env: Optional[gym.Env],
        enemy_move_desc: Optional[str],
        episode: int,
        round_idx: int,
    ) -> TurnResult:
        legal_actions: np.ndarray = np.array([], dtype=int)
        using_pikafish_legality = False
        engine_legal_count = 0
        legal_moves_hint: Optional[List[str]] = None
        if rank == 0:
            legal_actions, using_pikafish_legality, engine_legal_count = apply_pikafish_legal_mask(
                board_state=board_state,
                env=env,
                pikafish_evaluator=self.pikafish_evaluator,
            )
            if len(legal_actions) == 0:
                raise RuntimeError("XiangqiAgent: no legal ally moves available")
            # Shuffle to avoid positional bias when we have to trim the hint list.
            hint_actions = list(legal_actions)
            random.shuffle(hint_actions)
            legal_moves_hint = [action_to_algebraic(int(a)) for a in hint_actions]

        query_ids, responses, fen, graphic, _prefilter_legal_count = self._generate_candidates(
            board_state=board_state,
            env=env,
            enemy_move_desc=enemy_move_desc,
            episode=episode,
            round_idx=round_idx,
            legal_moves_hint=legal_moves_hint,
        )

        if rank == 0:
            evals: List[CandidateEval] = []
            legal_count = 0
            format_count = 0
            reasoning_count = 0
            move_parsed_count = 0
            move_strings: List[str] = []

            effective_format_weight = self._current_format_weight()
            for response in responses:
                clipped_query = query_ids
                if clipped_query.numel() > self.max_train_query_ctx:
                    clipped_query = clipped_query[-self.max_train_query_ctx :]
                ev = evaluate_candidate_response(
                    response=response,
                    board_before=board_state,
                    env=env,
                    query_ids=clipped_query,
                    tokenizer=self.tokenizer,
                    enemy_move_desc=enemy_move_desc,
                    pikafish_evaluator=self.pikafish_evaluator,
                    cp_scale=self.reward_cp_scale,
                    format_weight=effective_format_weight,
                    forced_action=None,
                )
                evals.append(ev)
                if ev.legal:
                    legal_count += 1
                if ev.has_format:
                    format_count += 1
                if ev.has_reasoning:
                    reasoning_count += 1
                if ev.parsed_move_ok:
                    move_parsed_count += 1
                if ev.move_str is not None:
                    move_strings.append(ev.move_str)

            legal_evals = [ev for ev in evals if ev.legal and ev.action is not None]

            # Dedupe legal candidates by parsed move_str so GRPO's group isn't
            # dominated by many near-identical completions that all picked the
            # same move (which flattens the within-group advantage signal).
            # For each distinct legal move we keep the single candidate with
            # the highest combined reward; all illegal / unparseable
            # candidates are kept as-is (they still carry useful "don't do
            # this" signal).
            if self.dedupe_legal_by_move and legal_evals:
                best_idx_by_move: Dict[str, int] = {}
                for idx_ev, ev in enumerate(evals):
                    if not (ev.legal and ev.action is not None and ev.move_str is not None):
                        continue
                    prev = best_idx_by_move.get(ev.move_str)
                    if prev is None or ev.reward > evals[prev].reward:
                        best_idx_by_move[ev.move_str] = idx_ev
                keep_legal_indices = set(best_idx_by_move.values())
                train_evals: List[CandidateEval] = [
                    ev for i, ev in enumerate(evals)
                    if (not (ev.legal and ev.action is not None)) or i in keep_legal_indices
                ]
            else:
                train_evals = list(evals)

            deduped_legal_evals = [
                ev for ev in train_evals if ev.legal and ev.action is not None
            ]
            distinct_legal_move_count = len({
                ev.move_str for ev in deduped_legal_evals if ev.move_str is not None
            })

            used_random_fallback = False
            chosen_capture = 0.0
            chosen_response = ""
            chosen_eval: Optional[CandidateEval] = None

            if deduped_legal_evals:
                best_reward = max(ev.reward for ev in deduped_legal_evals)
                best_candidates = [ev for ev in deduped_legal_evals if ev.reward == best_reward]
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

            query_batch = [ev.query_ids for ev in train_evals]
            response_batch = [ev.response_ids for ev in train_evals]
            reward_batch = [float(ev.reward) for ev in train_evals]
            train_stats = self.grpo_trainer.train_group(query_batch, response_batch, reward_batch)
            successful_cp_deltas = [float(ev.cp_delta) for ev in legal_evals if ev.cp_delta is not None]
            mean_cp_delta_success = (
                float(np.mean(successful_cp_deltas)) if successful_cp_deltas else None
            )

            round_format_rate = format_count / max(1, len(evals))
            self._update_format_compliance_ema(round_format_rate)

            candidate_metrics = {
                "game/legal_move_rate": legal_count / max(1, len(evals)),
                "game/parsed_move_rate": move_parsed_count / max(1, len(evals)),
                "game/format_compliance_rate": round_format_rate,
                "game/format_compliance_ema": float(self._format_compliance_ema),
                "game/effective_format_weight": float(effective_format_weight),
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
                "game/distinct_legal_moves": float(distinct_legal_move_count),
                "game/deduped_train_group_size": float(len(train_evals)),
                "game/dedupe_dropped_count": float(len(evals) - len(train_evals)),
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
                        f"{len(deduped_legal_evals)} legal_after_dedupe "
                        f"({distinct_legal_move_count} distinct moves), "
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
                    "Legal candidates [deduped by move] (move | engine_reward | cp_before | cp_after_raw | cp_delta | format_reward | combined_reward):",
                    *[
                        (
                            f"  [{idx}] {ev.move_str} "
                            f"engine={ev.engine_reward:.4f} "
                            f"cp_before={_fmt_optional_float(ev.cp_before)} "
                            f"cp_after_raw={_fmt_optional_float(ev.cp_after_raw)} "
                            f"cp_delta={_fmt_optional_float(ev.cp_delta)} "
                            f"format={ev.format_reward:.4f} "
                            f"combined={ev.reward:.4f} "
                            f"engine_eval_success={int(ev.engine_eval_success)}"
                        )
                        for idx, ev in enumerate(deduped_legal_evals)
                    ],
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
                generation_stats=dict(self._last_generation_stats),
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
            generation_stats=dict(self._last_generation_stats),
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
    # The expanded Xiangqi system prompt (piece-movement rules + anti-Western-chess
    # warning) is noticeably longer than the old one, so bump the prompt budget
    # accordingly. Total seq length = max_prompt_length + generate/max_new_tokens
    # with a small safety margin.
    "max_seq_length": 1536,
    "max_prompt_length": 768,
    "max_train_query_ctx": 768,
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
    "grpo/num_generations": 32,
    # When fewer than ``grpo/min_legal_candidates`` of the ``grpo/num_generations``
    # candidates are legal, regenerate an additional batch (of size
    # ``grpo/regeneration_batch_size``) and append it to the training group. Up to
    # ``grpo/max_regeneration_rounds`` extra passes are made before we give up.
    # This prevents the GRPO signal from collapsing when the policy happens to
    # output mostly illegal moves for a given position, which starves the group
    # of comparable rewards and kills the advantage signal.
    "grpo/min_legal_candidates": 8,
    # Regenerate if the number of *distinct* legal move strings seen so far is
    # below this threshold. Stops the group collapsing onto a handful of
    # moves (e.g. 20 of 26 legal candidates all outputting g9e7), which kills
    # GRPO's group-relative advantage signal. Set to 0 to disable.
    "grpo/min_distinct_legal_moves": 8,
    # When true, collapse legal candidates down to one row per distinct
    # parsed move (keeping the highest-reward row per move) before the GRPO
    # update. This complements ``min_distinct_legal_moves``: regeneration
    # grows the variety of moves in the pool, and dedup ensures the GRPO
    # group isn't dominated by duplicate-move rows whose identical rewards
    # flatten the within-group advantage. Illegal/unparseable rows are kept.
    "grpo/dedupe_legal_by_move": True,
    "grpo/max_regeneration_rounds": 3,
    "grpo/regeneration_batch_size": 12,
    "grpo/lr": 3e-6,
    # Lowered from 0.3 to match the per-token k3 KL scale (previous value was
    # calibrated to per-sequence summed KL). Expect per-token KL to live near
    # 1e-2 so a small beta is already strong regularization.
    "grpo/beta": 0.05,
    # PPO-style ratio clipping over ``grpo/ppo_epochs`` inner optimizer steps.
    # ``clip_eps_high > clip_eps_low`` (DAPO "Clip Higher") gives exploration
    # tokens a little more headroom to increase their probability while still
    # bounding the step on the downside.
    "grpo/ppo_epochs": 2,
    "grpo/clip_eps_low": 0.2,
    "grpo/clip_eps_high": 0.28,
    # Small entropy bonus on just the ``Move:`` region tokens (3-5 tokens).
    # Directly counteracts move-level mode collapse without affecting <think>
    # reasoning. Set to 0 to disable.
    "grpo/entropy_coef_move": 0.01,
    "grpo/max_grad_norm": 0.1,
    "grpo/optim": "adamw_8bit",
    "generate/max_new_tokens": 384,
    "generate/do_sample": True,
    # Higher sampling temperature + mild top_p widening discourage the 32
    # candidates from collapsing onto a single move, which was previously
    # destroying GRPO's group-relative advantage signal and wasting the
    # regeneration budget. Paired with ``grpo/dedupe_legal_by_move`` the
    # effective group keeps distinct-move diversity without blowing up the
    # group size.
    "generate/temperature": 1.2,
    "generate/top_p": 0.98,
    "generate/repetition_penalty": 1.1,
    # Regeneration passes (pass_idx > 0) use a higher temperature + top_p to
    # inject move diversity once the initial pass lands in a low-entropy
    # basin. Leave unset (None) to reuse the base generate config.
    "generate/regen_temperature": 1.6,
    "generate/regen_top_p": 0.98,
    "episodes": 500,
    "max_rounds_per_episode": 200,
    "seed": 42069,
    "metrics/clear_csv_on_start": True,
    "checkpoint/dir": "./checkpoints/xiangqi_grpo_v2",
    "checkpoint/every_n_episodes": 25,
    "checkpoint/load_adapter_path": "",
    "pikafish/bin": os.environ.get("PIKAFISH_BIN", "/home/fchow/bin/pikafish"),
    # Positional cp evaluation uses ``go movetime`` instead of ``go depth`` so
    # each candidate eval is bounded in wall-clock. ``list_legal_moves`` uses
    # ``go perft 1`` internally and ignores depth/movetime. ``pikafish/depth``
    # is kept only as a signal for ``movetime_ms`` when it isn't set.
    "pikafish/depth": 12,
    # Short timeout for handshake commands: uciok, readyok, stop.
    "pikafish/timeout_sec": 120.0,
    # Wall-clock budget per cp evaluation. ~500ms at depth-12-class quality on
    # modern CPUs is plenty while staying well under the old 2.5s read window.
    "pikafish/movetime_ms": 500,
    "reward/engine_cp_scale": 250.0,
    "reward/format_weight": 0.2,
    # Once the rolling format-compliance rate crosses
    # ``reward/format_weight_anneal_start`` the effective format weight decays
    # linearly toward ``reward/format_weight_min`` (reached at
    # ``reward/format_weight_anneal_end``). This keeps the format bonus useful
    # while the model is still learning the <think>/Move: template, but lets
    # the engine signal dominate once compliance is a solved problem.
    "reward/format_weight_min": 0.05,
    "reward/format_weight_anneal_start": 0.9,
    "reward/format_weight_anneal_end": 0.98,
    "reward/format_compliance_ema_alpha": 0.1,
    "grpo/logprob_micro_batch": 4,
    # One catastrophic blunder is worth thousands of cp, which dominates the
    # ``mean_chosen_cp_delta`` metric (and anything derived from it) across an
    # otherwise normal ~90-round game. Clip the raw per-turn cp_delta before it
    # feeds into aggregates so a single -10000 outlier can't swing the mean by
    # -100+. The reward itself is already bounded by tanh and is not touched.
    "metrics/cp_delta_clip_abs": 400.0,
    # When the engine's score stays at mate saturation (|cp_before| >=
    # ``game/cp_saturation_threshold``) for ``game/cp_saturation_consecutive``
    # ally turns in a row, the position is effectively terminal and continuing
    # the episode just pads the logs with 0-delta rounds. Truncate early.
    # Set ``game/cp_saturation_consecutive`` to 0 to disable.
    "game/cp_saturation_threshold": 8000.0,
    "game/cp_saturation_consecutive": 0,
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
    max_grad_norm=float(hyperparams["grpo/max_grad_norm"]),
    ppo_epochs=int(hyperparams.get("grpo/ppo_epochs", 2)),
    clip_eps_low=float(hyperparams.get("grpo/clip_eps_low", 0.2)),
    clip_eps_high=float(hyperparams.get("grpo/clip_eps_high", 0.28)),
    entropy_coef_move=float(hyperparams.get("grpo/entropy_coef_move", 0.0)),
    mp_policy=mp_policy,
    optimizer_name=str(hyperparams["grpo/optim"]),
    logprob_micro_batch=int(hyperparams.get("grpo/logprob_micro_batch", 4)),
)

generate_config = {
    "max_new_tokens": int(hyperparams["generate/max_new_tokens"]),
    "do_sample": bool(hyperparams["generate/do_sample"]),
    "temperature": float(hyperparams["generate/temperature"]),
    "top_p": float(hyperparams["generate/top_p"]),
    "repetition_penalty": float(hyperparams.get("generate/repetition_penalty", 1.0)),
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

regen_generate_overrides: Dict[str, Any] = {}
_regen_temp_cfg = hyperparams.get("generate/regen_temperature")
if _regen_temp_cfg is not None:
    regen_generate_overrides["temperature"] = float(_regen_temp_cfg)
_regen_top_p_cfg = hyperparams.get("generate/regen_top_p")
if _regen_top_p_cfg is not None:
    regen_generate_overrides["top_p"] = float(_regen_top_p_cfg)

pikafish_evaluator: Optional[PikafishEvaluator] = None
if rank == 0:
    pikafish_evaluator = PikafishEvaluator(
        binary_path=str(hyperparams["pikafish/bin"]),
        depth=int(hyperparams["pikafish/depth"]),
        timeout_sec=float(hyperparams["pikafish/timeout_sec"]),
        movetime_ms=int(hyperparams.get("pikafish/movetime_ms", 0) or 0) or None,
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
    reward_format_weight_min=float(hyperparams["reward/format_weight_min"]),
    reward_format_weight_anneal_start=float(
        hyperparams["reward/format_weight_anneal_start"]
    ),
    reward_format_weight_anneal_end=float(
        hyperparams["reward/format_weight_anneal_end"]
    ),
    reward_format_compliance_ema_alpha=float(
        hyperparams["reward/format_compliance_ema_alpha"]
    ),
    min_legal_candidates=int(hyperparams["grpo/min_legal_candidates"]),
    max_regeneration_rounds=int(hyperparams["grpo/max_regeneration_rounds"]),
    regeneration_batch_size=int(hyperparams["grpo/regeneration_batch_size"]),
    min_distinct_legal_moves=int(hyperparams["grpo/min_distinct_legal_moves"]),
    dedupe_legal_by_move=bool(hyperparams["grpo/dedupe_legal_by_move"]),
    regen_generate_overrides=regen_generate_overrides,
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
cp_delta_clip_abs = float(hyperparams["metrics/cp_delta_clip_abs"])
cp_saturation_threshold = float(hyperparams["game/cp_saturation_threshold"])
cp_saturation_consecutive = int(hyperparams["game/cp_saturation_consecutive"])

ally_wins = 0
enemy_wins = 0
truncated_games = 0
cp_saturation_truncations = 0
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
            chosen_cp_delta_raw_series: List[float] = []
            cp_saturation_streak = 0
            cp_saturation_truncated_this_episode = False
            train_stats_last: Dict[str, float] = {}
            episode_train_mfu_flops = 0.0
            episode_train_hfu_flops = 0.0
            episode_gen_flops = 0.0
            episode_train_wall_sec = 0.0
            episode_gen_wall_sec = 0.0
            episode_wall_start = time.perf_counter()

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
                    raw_delta = float(turn_result.chosen_cp_delta)
                    chosen_cp_delta_raw_series.append(raw_delta)
                    if cp_delta_clip_abs > 0.0:
                        chosen_cp_delta_series.append(
                            float(np.clip(raw_delta, -cp_delta_clip_abs, cp_delta_clip_abs))
                        )
                    else:
                        chosen_cp_delta_series.append(raw_delta)

                if (
                    cp_saturation_consecutive > 0
                    and turn_result.chosen_cp_before is not None
                    and abs(float(turn_result.chosen_cp_before)) >= cp_saturation_threshold
                ):
                    cp_saturation_streak += 1
                else:
                    cp_saturation_streak = 0

                train_stats_last = turn_result.train_stats

                gen_stats_turn = turn_result.generation_stats or {}
                if gen_stats_turn.get("num_sequences", 0.0) > 0.0:
                    episode_gen_flops += ally_agent.grpo_trainer.mfu_tracker.generation_flops(
                        num_sequences=int(gen_stats_turn.get("num_sequences", 0.0)),
                        prompt_len=int(gen_stats_turn.get("prompt_len", 0.0)),
                        generated_len=int(gen_stats_turn.get("generated_len", 0.0)),
                    )
                    episode_gen_wall_sec += float(gen_stats_turn.get("wall_sec", 0.0))

                if turn_result.train_stats:
                    episode_train_mfu_flops += float(
                        turn_result.train_stats.get("mfu/mfu_flops_step", 0.0)
                    )
                    episode_train_hfu_flops += float(
                        turn_result.train_stats.get("mfu/hfu_flops_step", 0.0)
                    )
                    episode_train_wall_sec += float(
                        turn_result.train_stats.get("mfu/step_time_sec", 0.0)
                    )

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

                if (
                    not done
                    and cp_saturation_consecutive > 0
                    and cp_saturation_streak >= cp_saturation_consecutive
                ):
                    done = True
                    cp_saturation_truncated_this_episode = True
                    cp_saturation_truncations += 1
                    # cp-saturation is a form of early truncation. Count it as a
                    # truncated game so that ally_win_rate + enemy_win_rate +
                    # truncated_rate sums to 100% (cp_saturation_truncation_rate
                    # is kept as a separate sub-category metric).
                    truncated_games += 1
                    last_cp = turn_result.chosen_cp_before
                    print(
                        f"[Ep {episode} Rd {round_idx - 1}] cp-saturation truncation "
                        f"(|cp_before| >= {cp_saturation_threshold:.0f} for "
                        f"{cp_saturation_streak} consecutive ally turns; "
                        f"last cp_before={_fmt_optional_float(last_cp)})"
                    )

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
            mean_chosen_cp_delta_raw = (
                float(np.mean(chosen_cp_delta_raw_series))
                if chosen_cp_delta_raw_series
                else None
            )
            random_rate_episode = (
                100.0 * random_fallback_episode / ally_turns_episode if ally_turns_episode else 0.0
            )

            outcome = "other"
            if enemy_reward_terminal == 100:
                outcome = "enemy_win"
            elif ally_reward_terminal == 100:
                outcome = "ally_win"
            elif cp_saturation_truncated_this_episode:
                outcome = "truncated_cp_saturation"
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
            if mean_chosen_cp_delta_raw is not None:
                episode_stats["game/mean_chosen_cp_delta_raw"] = mean_chosen_cp_delta_raw
            episode_stats["game/cp_delta_clip_abs"] = cp_delta_clip_abs
            episode_stats["game/cp_saturation_truncated"] = (
                1.0 if cp_saturation_truncated_this_episode else 0.0
            )
            episode_stats["game/cp_saturation_truncation_rate"] = (
                100.0 * cp_saturation_truncations / episode
            )

            episode_wall_sec = max(1e-9, time.perf_counter() - episode_wall_start)
            peak_flops = ally_agent.grpo_trainer.mfu_tracker.gpu_peak_flops
            episode_total_flops_mfu = episode_train_mfu_flops + episode_gen_flops
            episode_total_flops_hfu = episode_train_hfu_flops + episode_gen_flops
            if peak_flops > 0:
                e2e_mfu = episode_total_flops_mfu / (episode_wall_sec * peak_flops)
                e2e_hfu = episode_total_flops_hfu / (episode_wall_sec * peak_flops)
            else:
                e2e_mfu = 0.0
                e2e_hfu = 0.0
            episode_stats["mfu/episode_e2e_mfu"] = float(e2e_mfu)
            episode_stats["mfu/episode_e2e_hfu"] = float(e2e_hfu)
            episode_stats["mfu/episode_e2e_achieved_tflops_mfu"] = float(
                (episode_total_flops_mfu / episode_wall_sec) / 1e12
            )
            episode_stats["mfu/episode_e2e_achieved_tflops_hfu"] = float(
                (episode_total_flops_hfu / episode_wall_sec) / 1e12
            )
            episode_stats["mfu/episode_wall_sec"] = float(episode_wall_sec)
            episode_stats["mfu/episode_train_wall_sec"] = float(episode_train_wall_sec)
            episode_stats["mfu/episode_gen_wall_sec"] = float(episode_gen_wall_sec)
            episode_stats["mfu/episode_train_flops_mfu"] = float(episode_train_mfu_flops)
            episode_stats["mfu/episode_train_flops_hfu"] = float(episode_train_hfu_flops)
            episode_stats["mfu/episode_gen_flops"] = float(episode_gen_flops)
            if episode_wall_sec > 0:
                episode_stats["mfu/episode_gen_time_fraction"] = float(
                    episode_gen_wall_sec / episode_wall_sec
                )
                episode_stats["mfu/episode_train_time_fraction"] = float(
                    episode_train_wall_sec / episode_wall_sec
                )

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
                "game_cp_saturation_truncated": int(cp_saturation_truncated_this_episode),
                "game_mean_chosen_cp_delta_raw": _fmt_metric(mean_chosen_cp_delta_raw),
                "game_mean_chosen_cp_delta_clipped": _fmt_metric(mean_chosen_cp_delta),
                "grpo_loss": _fmt_metric(train_stats_last.get("grpo/loss")),
                "grpo_mean_kl": _fmt_metric(train_stats_last.get("grpo/mean_kl")),
                "grpo_mean_kl_per_token": _fmt_metric(
                    train_stats_last.get("grpo/mean_kl_per_token")
                ),
                "grpo_mean_kl_think": _fmt_metric(train_stats_last.get("grpo/mean_kl_think")),
                "grpo_mean_kl_move": _fmt_metric(train_stats_last.get("grpo/mean_kl_move")),
                "grpo_policy_entropy_move": _fmt_metric(
                    train_stats_last.get("grpo/policy_entropy_move")
                ),
                "grpo_pg_clip_frac": _fmt_metric(train_stats_last.get("grpo/pg_clip_frac")),
                "grpo_ratio_mean": _fmt_metric(train_stats_last.get("grpo/ratio_mean")),
                "grpo_ppo_epochs_completed": _fmt_metric(
                    train_stats_last.get("grpo/ppo_epochs_completed")
                ),
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
                f"mean_chosen_cp_delta={_fmt_optional_float(mean_chosen_cp_delta)} "
                f"mean_chosen_cp_delta_raw={_fmt_optional_float(mean_chosen_cp_delta_raw)} "
                f"outcome={outcome} "
                f"cp_sat_trunc_rate={episode_stats['game/cp_saturation_truncation_rate']:.1f}%"
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
