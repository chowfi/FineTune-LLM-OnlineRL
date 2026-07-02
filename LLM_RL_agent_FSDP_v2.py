"""
export PIKAFISH_BIN=/home/fchow/bin/pikafish
7B / single 5090:
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
14B / two 5090:
torchrun --nproc_per_node 2 LLM_RL_agent_FSDP_v2.py --model-size 14b --mixed-precision
"""

import argparse
import atexit
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256"
)

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

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from gym_xiangqi.constants import ALLY, PIECE_ID_TO_NAME, PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move, move_to_action_space

from pikafish_eval import PikafishEvaluator
from xiangqi_board import (
    COLS,
    algebraic_to_board_coords,
    algebraic_to_engine_move,
    board_coords_to_algebraic,
    board_to_fen,
    board_to_graphic,
    board_to_uci_fen,
    engine_uci_to_algebraic,
)
from xiangqi_labels import (
    SIGMA_GOOD,
    is_good_move,
    parse_situation_from_response,
    red_value_after_uci_move,
    root_value_red_oriented,
    situation_3class,
)

# PEFT's BaseTuner.forward() uses self.model.forward() directly, which bypasses
# __call__ and can skip distributed pre-forward hooks.
_peft_tuner_utils.BaseTuner.forward = lambda self, *args, **kwargs: self.model(
    *args, **kwargs
)


parser = argparse.ArgumentParser()
parser.add_argument("--mixed-precision", action="store_true")
parser.add_argument("--use-ddp", action="store_true")
parser.add_argument("--model-size", choices=["7b", "14b"], default="7b")
parser.add_argument("--episodes", type=int, default=None)
# Resume support. ``--resume-from`` overrides ``checkpoint/load_adapter_path``
# and ``--start-episode`` overrides ``checkpoint/start_episode`` (1-indexed,
# next episode to run after the loaded checkpoint).
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="Adapter directory to resume from (overrides checkpoint/load_adapter_path).",
)
parser.add_argument(
    "--start-episode",
    type=int,
    default=None,
    help="1-indexed episode number to start from (overrides checkpoint/start_episode).",
)
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


def restore_policy_train_mode(model) -> None:
    """Return the policy to training mode after inference-only passes.

    Unsloth-patched forwards gate gradient checkpointing on ``self.training``;
    leaving ``PeftModel.eval()`` from the legal-move scorer can therefore yield
    logits without a grad_fn during GRPO. Always pair inference ``.eval()`` blocks
    with this helper (or an equivalent ``.train()``) before ``backward``.

    When the policy is wrapped (e.g. DDP), callers may have called ``.eval()`` on
    the unwrapped module only; we set ``.train()`` on both outer and inner.
    """

    model.train()
    unwrap_model(model).train()


# -----------------------------------------------------------------------------
# Move / reasoning parsing (board ↔ FEN in xiangqi_board; engine in pikafish_eval)
# -----------------------------------------------------------------------------

MOVE_RE = re.compile(r"Move:\s*([a-i][0-9][a-i][0-9])", flags=re.IGNORECASE)
MOVE_TAG_RE = re.compile(r"Move:", flags=re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
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


def _build_move_constraint_fn(
    legal_moves: List[str],
    tokenizer,
    prompt_len: int,
):
    """Return a ``prefix_allowed_tokens_fn`` that forces every sample to, once
    it has emitted the literal ``"Move: "`` label, continue only with tokens
    that extend at least one legal move string (e.g. ``e9e8``).

    If the legal move list is empty or the tokenizer boundary-merges a move
    into the trailing space of ``"Move: "`` differently than it encodes it in
    isolation, we return ``None`` and the caller should skip the constraint
    (falling back to unconstrained regen sampling).

    Because HF requires ``prefix_allowed_tokens_fn`` to always return a list,
    we cache a single full-vocab fallback list and return it when no
    constraint applies.
    """
    if not legal_moves:
        return None
    move_prefix_text = "Move: "
    try:
        prefix_ids = tokenizer.encode(move_prefix_text, add_special_tokens=False)
    except Exception:
        return None
    if not prefix_ids:
        return None

    per_move_only: List[Tuple[int, ...]] = []
    for m in legal_moves:
        try:
            full_ids = tokenizer.encode(move_prefix_text + m, add_special_tokens=False)
        except Exception:
            continue
        if len(full_ids) <= len(prefix_ids):
            continue
        if full_ids[: len(prefix_ids)] != prefix_ids:
            # Tokenizer merged the boundary; skip this move. As long as at
            # least one move survives the per-move list is still useful.
            continue
        per_move_only.append(tuple(full_ids[len(prefix_ids) :]))
    if not per_move_only:
        return None

    prefix_tuple = tuple(prefix_ids)
    plen = len(prefix_tuple)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    full_vocab: List[int] = list(range(vocab_size))

    def fn(batch_id: int, input_ids) -> List[int]:
        try:
            gen = input_ids.tolist()[prompt_len:]
        except Exception:
            return full_vocab
        if not gen:
            return full_vocab
        # Walk backwards to find the latest ``"Move: "`` marker whose move
        # has not yet been fully emitted. Anything before a completed
        # move is ignored so the model can freely continue past it.
        last_end = -1
        for i in range(len(gen) - plen, -1, -1):
            if tuple(gen[i : i + plen]) == prefix_tuple:
                after_i = tuple(gen[i + plen :])
                if not any(after_i == m for m in per_move_only):
                    last_end = i + plen
                    break
        if last_end < 0:
            return full_vocab
        after = tuple(gen[last_end:])
        allowed = set()
        for m in per_move_only:
            if len(after) >= len(m):
                continue
            if m[: len(after)] == after:
                allowed.add(m[len(after)])
        if not allowed:
            return full_vocab
        return sorted(allowed)

    return fn


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


def algebraic_to_action(
    move_str: str, board_state: np.ndarray, env: gym.Env
) -> Optional[int]:
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


def apply_pikafish_enemy_legal_mask(
    board_state: np.ndarray,
    env: gym.Env,
    pikafish_evaluator: Optional["PikafishEvaluator"],
) -> Tuple[np.ndarray, bool, int, int]:
    """Restrict ``env.enemy_actions`` to Pikafish-legal Black moves.

    Returns ``(legal_actions, mask_applied, engine_legal_count, gym_legal_count)``.
    """
    gym_legal = np.where(env.enemy_actions == 1)[0]
    gym_count = len(gym_legal)
    if pikafish_evaluator is None or not pikafish_evaluator.enabled or gym_count == 0:
        return gym_legal, False, 0, gym_count

    fen_before = board_to_uci_fen(board_state, side_to_move="b")
    engine_legal_moves = pikafish_evaluator.list_legal_moves(fen_before)
    if not engine_legal_moves:
        return gym_legal, False, 0, gym_count

    alg_to_action = {action_to_algebraic(int(a)): int(a) for a in gym_legal}
    engine_actions: List[int] = []
    for move_str in engine_legal_moves:
        alg = engine_uci_to_algebraic(move_str)
        if alg and alg in alg_to_action:
            engine_actions.append(alg_to_action[alg])

    if not engine_actions:
        return gym_legal, False, len(engine_legal_moves), gym_count

    unique_actions = np.array(sorted(set(engine_actions)), dtype=int)
    env.enemy_actions.fill(0)
    env.enemy_actions[unique_actions] = 1
    return unique_actions, True, len(engine_legal_moves), gym_count


def action_to_algebraic(action: int) -> str:
    _, start, end = action_space_to_move(int(action))
    return board_coords_to_algebraic(start[0], start[1], end[0], end[1])


def describe_action(action: int) -> str:
    piece_id, start, end = action_space_to_move(int(action))
    piece_name = (
        PIECE_ID_TO_NAME[piece_id]
        if piece_id < len(PIECE_ID_TO_NAME)
        else f"piece_{piece_id}"
    )
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
    # Discrete tactical indicators (Xiangqi-R1 §3.4 R_move components). These
    # are 1.0 when the candidate's parsed move matches Pikafish's bestmove or
    # falls within ``SIGMA_GOOD`` cp of it; ``0.0`` otherwise. They are only
    # populated when ``combine_gate_with_r_best`` is active for the parent
    # turn; otherwise they remain ``0.0`` and contribute nothing.
    r_best: float
    r_good: float
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
    grounding_strict: bool = False,
) -> float:
    """Score how well the ``<think>`` block justifies the move that will be played.

    The old implementation awarded points for generic keyword presence and any
    move-like token, which the model easily gamed without actually describing
    its own move. The new version only gives full credit when the reasoning
    references the move (UCI string or source/target square or the piece type
    name) that the candidate will actually execute.

    When ``grounding_strict`` is True, the total score is capped unless the
    exact UCI ``chosen_move`` substring appears inside the thinking block.
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

    score = float(min(score, 1.0))
    # Require the exact UCI string inside thinking so generic prose cannot
    # max out the rubric while describing a different move.
    if grounding_strict and chosen_move:
        cm = chosen_move.lower()
        if cm not in think:
            score = min(score, 0.32)
    return score


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


_CP_SCALE_WIDEN_CAP_MULT = 1.2
_CP_SCALE_WIDEN_SLOPE = 0.2


def adaptive_cp_scale(base_scale: float, cp_before: Optional[float]) -> float:
    """Stretch the tanh normalisation by how extreme the current position is.

    In a dead-lost or dead-won position the absolute |cp_before| can be several
    thousand centipawns (including mate-inflated values near 9000-10000). With a
    fixed scale of 250 the tanh saturates and every candidate move receives the
    same reward, destroying the advantage signal. We widen the scale
    proportionally to ``|cp_before|`` but cap it at
    ``_CP_SCALE_WIDEN_CAP_MULT * base_scale`` so it never becomes so wide that
    every move in a lost position looks identical to the engine reward.

    The cap was tightened from ``2x`` to ``1.2x`` (2026-05-15): the wider cap
    was producing reward ranges so narrow in saturated positions that GRPO's
    within-group advantage was collapsing toward zero. With a 1.2x cap, a
    ``cp_before = -8000``, ``cp_delta = -100`` candidate now scores
    ``5.5 + 4.5 * tanh(-100/300) ~= 4.07`` instead of ``~4.6`` at scale=500,
    so cp_delta ranges of [-500, 0] across 32 candidates now span roughly
    [1.0, 5.5] in reward instead of [2.0, 5.5] -- ~2x the ``reward_std`` and
    ~2x the advantage gradient for the same cp data.
    """
    base = max(10.0, float(base_scale))
    if cp_before is None:
        return base
    widened = max(base, _CP_SCALE_WIDEN_SLOPE * abs(float(cp_before)))
    return min(widened, _CP_SCALE_WIDEN_CAP_MULT * base)


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
    reward_format_mode: str = "mix",
    grounding_strict: bool = False,
    grounding_quality_min: float = 0.4,
    format_gate_fail_scale: float = 0.08,
    format_soft_fail_scale: float = 0.45,
    combine_gate_with_r_best: bool = False,
    position_best_uci_engine: Optional[str] = None,
    position_vb_best: Optional[float] = None,
    tactical_weight: float = 1.5,
    sigma_good_cp: float = SIGMA_GOOD,
    engine_only: bool = False,
) -> CandidateEval:
    """Evaluate a single LLM candidate.

    The model now emits both ``<think>`` reasoning and ``Move: <uci>`` on its own.
    ``forced_action`` is only honoured when the LLM failed to produce a parseable
    legal move; in that case we still evaluate the fallback move but heavily
    discount the reward so GRPO continues to push the policy toward self-picked
    moves. Illegal/unparseable candidates receive a zero reward which gives
    strong learning signal away from malformed outputs.

    ``reward_format_mode``:
      - ``mix``: ``reward = (1-mix)*engine + mix*format_reward`` (legacy).
      - ``gate``: engine-first; bad/missing format or low ``reasoning_quality``
        downscales ``engine_reward``; good grounding adds a small bonus only.
      - ``xiangqi_r1``: discrete ``R_move + R_analysis + R_format`` (paper §3.4);
        if format fails, total reward is **0** (move and analysis not counted).
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
    r_best_indicator: float = 0.0
    r_good_indicator: float = 0.0

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
        grounding_strict=grounding_strict,
    )

    if action is not None and move_str is not None:
        legal = True
        capture_value = _capture_value_for_move(board_before, move_str)

        fen_before = board_to_uci_fen(board_before, side_to_move="w")
        engine_move = algebraic_to_engine_move(move_str)

        if pikafish_evaluator and pikafish_evaluator.enabled:
            cp_before = pikafish_evaluator.evaluate_cp(fen_before, moves=None)
            cp_after_raw = (
                pikafish_evaluator.evaluate_cp(fen_before, moves=[engine_move])
                if engine_move
                else None
            )
            if cp_before is not None and cp_after_raw is not None:
                cp_after_from_ally = -float(cp_after_raw)
                cp_delta = cp_after_from_ally - float(cp_before)
                engine_eval_success = True

        r_format_gate = (
            1.0 if (has_reasoning and parsed_move_ok and bool(move_str)) else 0.0
        )

        if engine_only:
            format_reward = 0.0
            has_format = bool(parsed_move_ok)
            if (
                pikafish_evaluator
                and pikafish_evaluator.enabled
                and cp_delta is not None
            ):
                effective_scale = adaptive_cp_scale(cp_scale, cp_before)
                engine_reward = normalize_cp_delta_to_reward(
                    float(cp_delta), cp_scale=effective_scale
                )
            else:
                engine_reward = 0.0
            reward = float(engine_reward)
        elif reward_format_mode == "xiangqi_r1":
            format_subscore = 0.0
            if has_reasoning:
                format_subscore += 0.2
            if parsed_move_ok:
                format_subscore += 0.2
            format_subscore += 0.6 * reasoning_quality
            format_reward = 1.0 + 9.0 * float(min(format_subscore, 1.0))

            if (
                r_format_gate < 1.0
                or not pikafish_evaluator
                or not pikafish_evaluator.enabled
                or not engine_move
            ):
                engine_reward = 0.0
                reward = 0.0
            else:
                best_uci, root_cp = pikafish_evaluator.bestmove_root_cached(fen_before)
                vr = root_value_red_oriented(fen_before, root_cp)
                gold_sit = situation_3class(vr) if vr is not None else None
                pred_sit = parse_situation_from_response(response)
                r_analysis = (
                    1.0
                    if gold_sit is not None
                    and pred_sit is not None
                    and pred_sit == gold_sit
                    else 0.0
                )
                r_legal = 1.0
                r_best = (
                    1.0 if best_uci and engine_move.lower() == best_uci.lower() else 0.0
                )
                if best_uci:
                    good_ok, _, _ = is_good_move(
                        fen_before,
                        engine_move.lower(),
                        best_uci.lower(),
                        pikafish_evaluator.evaluate_cp,
                    )
                else:
                    good_ok = False
                r_good = 1.0 if (r_best >= 1.0 or good_ok) else 0.0
                r_move = r_legal + r_good + r_best
                engine_reward = float(r_move)
                reward = float(r_move + r_analysis + r_format_gate)
        else:
            # Baseline neutral engine reward; tanh of cp_delta overrides when the
            # engine is available.
            engine_reward = 5.5
            if (
                pikafish_evaluator
                and pikafish_evaluator.enabled
                and cp_delta is not None
            ):
                effective_scale = adaptive_cp_scale(cp_scale, cp_before)
                engine_reward = normalize_cp_delta_to_reward(
                    float(cp_delta), cp_scale=effective_scale
                )

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
            if reward_format_mode == "gate":
                if not (has_reasoning and parsed_move_ok and move_str):
                    reward = float(engine_reward * format_gate_fail_scale)
                elif reasoning_quality < grounding_quality_min:
                    reward = float(engine_reward * format_soft_fail_scale)
                else:
                    reward = float(engine_reward + mix * reasoning_quality * 2.0)
                    reward = float(min(reward, 10.0))
            else:
                reward = (1.0 - mix) * engine_reward + mix * format_reward

            # Discrete Xiangqi-R1 R_move bonus on top of the dense ``gate``
            # reward. ``r_best`` rewards exact match with Pikafish's deep-search
            # bestmove; ``r_good`` rewards a move within ``sigma_good_cp`` of
            # the best-move's resulting position. The reward is only added on
            # well-formed responses (so the format gate still applies); the
            # caller pre-computes ``position_best_uci_engine`` and
            # ``position_vb_best`` once per turn so we don't pay extra Pikafish
            # calls per candidate. ``vp_played`` reuses ``cp_after_from_ally``
            # which is already computed above (red-oriented value after the
            # candidate move).
            if (
                combine_gate_with_r_best
                and reward_format_mode == "gate"
                and has_reasoning
                and parsed_move_ok
                and move_str
                and reasoning_quality >= grounding_quality_min
                and engine_move
                and position_best_uci_engine
            ):
                if engine_move.lower() == position_best_uci_engine.lower():
                    r_best_indicator = 1.0
                    r_good_indicator = 1.0
                elif position_vb_best is not None and engine_eval_success:
                    vp_played = (
                        -float(cp_after_raw) if cp_after_raw is not None else None
                    )
                    if vp_played is not None and abs(
                        vp_played - float(position_vb_best)
                    ) <= float(sigma_good_cp):
                        r_good_indicator = 1.0
                tactical_bonus = float(tactical_weight) * (
                    r_good_indicator + r_best_indicator
                )
                if tactical_bonus > 0.0:
                    reward = float(min(reward + tactical_bonus, 13.0))

        # Discourage relying on the forced fallback: give only 30% of the
        # reward when the LLM didn't produce a parseable legal move of its own.
        if used_forced:
            reward *= 0.3

    response_ids = tokenizer(
        response, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
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
        r_best=float(r_best_indicator),
        r_good=float(r_good_indicator),
        query_ids=query_ids.cpu(),
        response_ids=response_ids.cpu(),
    )


# -----------------------------------------------------------------------------
# Logging and episode metrics
# -----------------------------------------------------------------------------

SYNC_LOG_FILE = "xiangqi_v2_board_sync.log"
EPISODE_METRICS_CSV = "chinese_chess_episode_metrics_v2.csv"
# Rank-0 JSON heartbeat (atomic replace). Disable with ``training/run_heartbeat_path: ""``.
RUN_HEARTBEAT_DEFAULT = "xiangqi_v2_run_heartbeat.json"

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
    "game_parsed_move_rate",
    "game_format_compliance_rate",
    "game_reasoning_rate",
    "game_mean_capture_value",
    "game_mean_best_candidate_reward",
    "game_mean_chosen_engine_reward",
    "game_move_diversity",
    "game_legal_move_diversity",
    "game_mean_legal_anchor_count",
    "game_cp_saturation_truncated",
    "game_mean_chosen_cp_delta_raw",
    "game_mean_chosen_cp_delta_clipped",
    "game_mean_ally_cp_after_move_red",
    "game_median_ally_cp_after_move_red",
    "game_ally_cp_after_move_red_ema",
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
    "grpo_update_rate",
    "grpo_skip_low_reward_std_rate",
    "grpo_engine_align_loss",
    "grpo_engine_align_kl",
    "grpo_engine_align_entropy",
    "grpo_engine_align_target_entropy",
    "grpo_engine_align_valid_count",
    "mfu",
    "hfu",
    "mfu_step_time_sec",
    # Engine-best agreement aggregates (per-episode means; rates are in %).
    "game_engine_best_known_rate",
    "game_engine_best_in_group_rate",
    "game_chosen_is_engine_argmax_in_group_rate",
    "game_chosen_is_engine_best_overall_rate",
    "game_mean_chosen_engine_rank_in_group",
    "game_median_chosen_engine_rank_in_group",
    "game_mean_chosen_minus_argmax_cp_delta",
    # Combined-reward (gate + Xiangqi-R1 R_move) aggregates. Only meaningful
    # when ``reward/combine_gate_with_r_best`` is True; otherwise stay at 0.
    "game_mean_r_best_in_group_rate",
    "game_mean_r_good_in_group_rate",
    "game_chosen_r_best_rate",
    "game_chosen_r_good_rate",
    # Opponent ε-random curriculum (GreedyEnemy only; unused in self-play).
    "game_enemy_epsilon_current",
    "game_enemy_pikafish_prune_rate",
    "game_consecutive_self_play_wins",
    "game_self_play_enemy_id",
    "game_global_train_step_end",
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


def ensure_episode_metrics_csv_schema(filepath: str) -> None:
    """Upgrade an existing CSV in-place to the current ``_EPISODE_METRICS_FIELDNAMES``.

    No-op if the file is missing or the header already matches. Otherwise the
    file is rewritten with the new header and all prior rows preserved
    (missing new columns become empty strings). The old file is backed up to
    ``<filepath>.bak`` once.
    """
    if not os.path.isfile(filepath):
        return
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        print(
            f"[csv-schema] failed to read {filepath!r} for schema check: {exc!r} -- leaving untouched",
            flush=True,
        )
        return
    if existing_fields == _EPISODE_METRICS_FIELDNAMES:
        return
    backup = filepath + ".bak"
    try:
        if not os.path.isfile(backup):
            shutil.copyfile(filepath, backup)
    except Exception as exc:
        print(
            f"[csv-schema] failed to write backup {backup!r}: {exc!r}",
            flush=True,
        )
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_EPISODE_METRICS_FIELDNAMES,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: row.get(field, "") for field in _EPISODE_METRICS_FIELDNAMES}
                )
        print(
            f"[csv-schema] upgraded {filepath!r} header "
            f"(added {len(set(_EPISODE_METRICS_FIELDNAMES) - set(existing_fields))} new columns; "
            f"backup at {backup!r})",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[csv-schema] failed to rewrite {filepath!r}: {exc!r}",
            flush=True,
        )


def append_episode_metrics_csv(filepath: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_METRICS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class _ResumeRunState:
    season_ally_return: float = 0.0
    season_enemy_return: float = 0.0
    ally_wins: int = 0
    enemy_wins: int = 0
    truncated_games: int = 0
    cp_saturation_truncations: int = 0
    lifetime_ally_turns: int = 0
    lifetime_random_fallback: int = 0
    consecutive_self_play_wins: int = 0


def preload_run_state_from_csv(
    filepath: str, up_to_episode_exclusive: int
) -> _ResumeRunState:
    """Reconstruct cumulative run-level counters from the existing CSV.

    Used when resuming training mid-run: rows for episodes
    ``[1, up_to_episode_exclusive)`` are summed so the stdout scoreboard,
    win-rate and lifetime metrics continue smoothly. Missing/invalid values
    are tolerated (the run still proceeds; only aggregates miss those rows).
    """
    state = _ResumeRunState()
    if up_to_episode_exclusive <= 1 or not os.path.isfile(filepath):
        return state
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(row.get("episode") or 0)
                except (TypeError, ValueError):
                    continue
                if ep < 1 or ep >= up_to_episode_exclusive:
                    continue
                try:
                    state.season_ally_return += float(row.get("ally_return") or 0.0)
                except (TypeError, ValueError):
                    pass
                try:
                    state.season_enemy_return += float(row.get("enemy_return") or 0.0)
                except (TypeError, ValueError):
                    pass
                outcome = (row.get("outcome") or "").strip().lower()
                if outcome == "ally_win":
                    state.ally_wins += 1
                elif outcome == "enemy_win":
                    state.enemy_wins += 1
                elif outcome.startswith("truncated"):
                    state.truncated_games += 1
                try:
                    if int(row.get("game_cp_saturation_truncated") or 0):
                        state.cp_saturation_truncations += 1
                except (TypeError, ValueError):
                    pass
                try:
                    state.lifetime_ally_turns += int(row.get("ally_turns_episode") or 0)
                except (TypeError, ValueError):
                    pass
                try:
                    state.lifetime_random_fallback += int(
                        row.get("random_fallback_episode") or 0
                    )
                except (TypeError, ValueError):
                    pass
                if ep == up_to_episode_exclusive - 1:
                    try:
                        state.consecutive_self_play_wins = int(
                            row.get("game_consecutive_self_play_wins") or 0
                        )
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:
        print(
            f"[resume] failed to preload run state from {filepath!r}: {exc!r} -- continuing with zeros",
            flush=True,
        )
    return state


def log_board_sync(lines: List[str], log_file: str = SYNC_LOG_FILE) -> None:
    payload = "\n".join(lines)
    print(payload)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(payload + "\n\n")


def format_episode_open_scoreboard(
    episode: int,
    season_ally_return: float,
    season_enemy_return: float,
) -> str:
    """Human-readable cumulative env rewards: this episode (zeros) vs all episodes so far."""
    return (
        f"[Ep {episode}] scoreboard: episode ally=0.00 enemy=0.00 | "
        f"all_episodes ally={season_ally_return:.2f} enemy={season_enemy_return:.2f}"
    )


def format_round_scoreboard(
    episode: int,
    round_idx: int,
    ally_episode_return: float,
    enemy_episode_return: float,
    season_ally_return: float,
    season_enemy_return: float,
) -> str:
    """Episode-to-date and season-to-date cumulative env rewards (printed to stdout / wandb capture)."""
    return (
        f"[Ep {episode} Rd {round_idx}] scoreboard: "
        f"episode ally={ally_episode_return:.2f} enemy={enemy_episode_return:.2f} | "
        f"all_episodes ally={season_ally_return + ally_episode_return:.2f} "
        f"enemy={season_enemy_return + enemy_episode_return:.2f}"
    )


@dataclass
class _RunHeartbeatController:
    """Process-local training progress for post-mortems (SIGKILL leaves last good JSON)."""

    path: Optional[str] = None
    rank: int = -1
    episode: Optional[int] = None
    round_idx: Optional[int] = None
    phase: str = "init"
    ally_return: float = 0.0
    enemy_return: float = 0.0
    global_train_steps: int = 0
    exit_normal: bool = False
    detail_written: bool = False

    def configure(self, *, path: Optional[str], rank: int) -> None:
        self.path = path
        self.rank = rank

    def write_atomic_json(self, record: Dict[str, Any]) -> None:
        if self.rank != 0 or not self.path:
            return
        tmp = f"{self.path}.tmp.{os.getpid()}"
        out = dict(record)
        out.setdefault("updated_unix", time.time())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

    def flush(
        self,
        status: str = "running",
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.rank != 0 or not self.path:
            return
        rec: Dict[str, Any] = {
            "status": status,
            "phase": self.phase,
            "episode": self.episode,
            "round_idx": self.round_idx,
            "ally_return": float(self.ally_return),
            "enemy_return": float(self.enemy_return),
            "global_train_step": int(self.global_train_steps),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "rank": int(self.rank),
        }
        if extra:
            for k, v in extra.items():
                if v is not None:
                    rec[k] = v
        self.write_atomic_json(rec)

    def touch(
        self,
        phase: str,
        *,
        episode: int,
        round_idx: int,
        ally_return: float,
        enemy_return: float,
        global_train_steps: int,
        status: str = "running",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.phase = phase
        self.episode = episode
        self.round_idx = round_idx
        self.ally_return = ally_return
        self.enemy_return = enemy_return
        self.global_train_steps = global_train_steps
        self.flush(status, extra=extra)

    def install_hooks(self) -> None:
        if self.rank != 0 or not self.path:
            return
        print(
            f"[run_status] heartbeat JSON -> {os.path.abspath(self.path)} "
            f"(updated each round; SIGINT/SIGTERM write reason)",
            flush=True,
        )

        def _on_signal(signum: int, _frame: Any) -> None:
            if self.rank != 0:
                raise KeyboardInterrupt
            sig = (
                "SIGINT"
                if signum == signal.SIGINT
                else "SIGTERM"
                if signum == signal.SIGTERM
                else f"SIGNUM_{signum}"
            )
            self.phase = f"received_{sig}"
            self.flush(
                f"signal:{sig}",
                extra={
                    "signal": sig,
                    "interrupted_at_episode": self.episode,
                    "interrupted_at_round": self.round_idx,
                },
            )
            self.detail_written = True
            print(
                f"\n[run_status] {sig} — last state written to {self.path!r}",
                flush=True,
            )
            try:
                wandb.finish()
            except Exception:
                pass
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + int(signum))

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        def _atexit_hb() -> None:
            if (
                self.rank != 0
                or not self.path
                or self.exit_normal
                or self.detail_written
            ):
                return
            self.phase = "atexit_without_clean_shutdown"
            self.flush(
                "process_exit_unclean",
                extra={
                    "note": "atexit without normal completion, signal handler, or "
                    "python_exception heartbeat (often OOM SIGKILL, kill -9, or CUDA abort)."
                },
            )

        atexit.register(_atexit_hb)


_RUN_HB = _RunHeartbeatController()


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
        self.trainable_params = sum(
            p.numel() for p in unwrapped.parameters() if p.requires_grad
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype_str = _resolve_compute_dtype_str(mp_policy)
        self.gpu_peak_tflops = _get_gpu_peak_tflops(device_index, self.dtype_str)
        self.gpu_peak_flops = self.gpu_peak_tflops * 1e12
        self.history: List[Dict[str, float]] = []

    def compute(
        self, total_tokens: int, elapsed_sec: float, num_fwd_per_sample: int = 2
    ) -> Dict[str, float]:
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

        mfu = (
            (mfu_flops / elapsed_sec) / self.gpu_peak_flops
            if self.gpu_peak_flops > 0
            else 0.0
        )
        hfu = (
            (hfu_flops / elapsed_sec) / self.gpu_peak_flops
            if self.gpu_peak_flops > 0
            else 0.0
        )

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
            2.0
            * float(self.total_params)
            * float(num_sequences)
            * float(prompt_len + generated_len)
        )


# -----------------------------------------------------------------------------
# Environment opponent
# -----------------------------------------------------------------------------


SELF_PLAY_ENEMY_META_FILENAME = "enemy_meta.json"


def ensure_frozen_sft_ref_adapter(
    peft_model,
    sft_ref_dir: str,
    *,
    rank: int = 0,
) -> bool:
    """Load a frozen ``sft_ref`` LoRA slot for KL anchoring (never trained).

    Returns True when ``sft_ref`` is available on ``peft_model``.
    """
    sft_ref_dir = (sft_ref_dir or "").strip()
    if not sft_ref_dir or not os.path.isdir(sft_ref_dir):
        if rank == 0:
            print(
                f"[kl-ref] sft_ref path not found ({sft_ref_dir!r}); "
                "KL will fall back to base model (adapters disabled).",
                flush=True,
            )
        return False

    peft_config = getattr(peft_model, "peft_config", {}) or {}
    if "sft_ref" not in peft_config:
        peft_model.load_adapter(sft_ref_dir, adapter_name="sft_ref")
        if rank == 0:
            print(
                f"[kl-ref] Loaded frozen sft_ref adapter from {sft_ref_dir!r}",
                flush=True,
            )

    for name, param in peft_model.named_parameters():
        if ".sft_ref." in name:
            param.requires_grad = False

    active = getattr(peft_model, "active_adapter", "default")
    if isinstance(active, list):
        active = active[0] if active else "default"
    if hasattr(peft_model, "set_adapter") and active != "default":
        peft_model.set_adapter("default")
    return True


def infer_self_play_enemy_id(start_episode: int) -> int:
    """Best-effort enemy generation when ``enemy_meta.json`` is missing.

    Known sync boundaries in the May-22 long run: after ep 15 and ep 30.
    """
    if start_episode >= 31:
        return 2
    if start_episode >= 16:
        return 1
    return 0


def load_self_play_enemy_id(
    enemy_dir: str, start_episode: int = 1, csv_path: Optional[str] = None
) -> int:
    meta_path = os.path.join(enemy_dir, SELF_PLAY_ENEMY_META_FILENAME)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                payload = json.load(f)
            return int(payload.get("enemy_id", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in reversed(rows):
                raw = (row.get("game_self_play_enemy_id") or "").strip()
                if raw:
                    return int(float(raw))
        except (OSError, ValueError, TypeError):
            pass
    return infer_self_play_enemy_id(start_episode)


def save_self_play_enemy_meta(
    enemy_dir: str,
    enemy_id: int,
    *,
    synced_at_episode: int,
    synced_at_global_step: int,
) -> None:
    os.makedirs(enemy_dir, exist_ok=True)
    meta_path = os.path.join(enemy_dir, SELF_PLAY_ENEMY_META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "enemy_id": int(enemy_id),
                "synced_at_episode": int(synced_at_episode),
                "synced_at_global_step": int(synced_at_global_step),
            },
            f,
            indent=2,
        )


def setup_wandb_episode_metric_axes() -> None:
    """Use ``episode`` as the x-axis for per-episode game/enemy metrics in W&B."""
    if rank != 0:
        return
    wandb.define_metric("episode")
    wandb.define_metric("train/global_step")
    episode_step_metrics = (
        "game/mean_chosen_engine_reward",
        "game/mean_chosen_cp_delta",
        "game/mean_chosen_cp_delta_clipped",
        "game/chosen_is_engine_argmax_in_group_rate",
        "game/median_chosen_engine_rank_in_group",
        "game/mean_chosen_engine_rank_in_group",
        "game/mean_ally_cp_after_move_red",
        "game/median_ally_cp_after_move_red",
        "game/ally_cp_after_move_red_ema",
        "game/episode_length",
        "game/ally_return",
        "game/ally_win_rate",
        "game/self_play_enemy_id",
        "enemy/self_play_enemy_id",
        "enemy/sync_marker",
    )
    for key in episode_step_metrics:
        wandb.define_metric(key, step_metric="episode")


def sync_enemy_adapter_with_default(
    model,
    *,
    enemy_dir: str = "checkpoints/self_play_enemy",
    new_enemy_id: Optional[int] = None,
    synced_at_episode: Optional[int] = None,
    synced_at_global_step: Optional[int] = None,
) -> None:
    state_dict = model.state_dict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".enemy." in name:
                default_name = name.replace(".enemy.", ".default.")
                if default_name in state_dict:
                    param.copy_(state_dict[default_name])

    # Also save the updated adapter persistently to checkpoints/self_play_enemy (rank 0 only)
    if rank == 0:
        os.makedirs(enemy_dir, exist_ok=True)
        unwrap_model(model).save_pretrained(enemy_dir, selected_adapters=["enemy"])
        print(
            f"[self-play] Synchronized and saved new frozen enemy adapter to {enemy_dir}",
            flush=True,
        )
        if new_enemy_id is not None:
            save_self_play_enemy_meta(
                enemy_dir,
                int(new_enemy_id),
                synced_at_episode=int(synced_at_episode or 0),
                synced_at_global_step=int(synced_at_global_step or 0),
            )
    dist_barrier()


def flip_move(move_str: str) -> str:
    parsed = algebraic_to_board_coords(move_str)
    if parsed is None:
        return move_str
    (from_row, from_col), (to_row, to_col) = parsed
    return board_coords_to_algebraic(9 - from_row, from_col, 9 - to_row, to_col)


def get_flipped_enemy_legal_actions(env: gym.Env) -> Tuple[List[int], Dict[int, int]]:
    original_actions = np.where(env.enemy_actions == 1)[0]
    flipped_actions = []
    flipped_to_original = {}
    flipped_board = -env.state[::-1, :]
    for orig_act in original_actions:
        orig_move_str = action_to_algebraic(int(orig_act))
        flipped_move_str = flip_move(orig_move_str)
        parsed = algebraic_to_board_coords(flipped_move_str)
        if parsed is not None:
            (from_row, from_col), (to_row, to_col) = parsed
            piece_id = int(flipped_board[from_row][from_col])
            if piece_id > 0:
                flipped_act = int(
                    move_to_action_space(
                        piece_id, (from_row, from_col), (to_row, to_col)
                    )
                )
                flipped_actions.append(flipped_act)
                flipped_to_original[flipped_act] = int(orig_act)
    return flipped_actions, flipped_to_original


def evaluate_moves_logprobs(
    agent: "XiangqiAgent",
    query_ids_batch: Optional[List[torch.Tensor]],
    response_ids_batch: Optional[List[torch.Tensor]],
) -> List[float]:
    """Score a batch of query/response pairs (legal moves) under the active adapter model."""
    if dist.is_initialized():
        synced_q, synced_r, _ = agent.grpo_trainer._broadcast_group(
            query_ids_batch,
            response_ids_batch,
            [0.0 for _ in (query_ids_batch or [])],
        )
    else:
        synced_q, synced_r = query_ids_batch, response_ids_batch

    scores: List[float] = []
    if synced_q and synced_r:
        model_obj = unwrap_model(agent.model)
        model_obj.eval()
        n_moves = len(synced_q)
        micro = max(1, int(agent.grpo_trainer.logprob_micro_batch))
        chunk_micro = micro
        oom_retries_left = 3
        try:
            with torch.no_grad():
                while oom_retries_left >= 0:
                    try:
                        score_chunks: List[float] = []
                        for s in range(0, n_moves, chunk_micro):
                            e = min(s + chunk_micro, n_moves)
                            qs = [synced_q[i] for i in range(s, e)]
                            rs = [synced_r[i] for i in range(s, e)]
                            fwd = agent.grpo_trainer._compute_response_log_probs_batch(
                                qs, rs
                            )
                            token_lp = fwd["token_lp"]
                            response_mask = fwd["resp_mask"]
                            scores_t = (
                                (token_lp * response_mask)
                                .sum(dim=1)
                                .detach()
                                .float()
                                .cpu()
                                .tolist()
                            )
                            score_chunks.extend(scores_t)
                        scores = score_chunks
                        break
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if chunk_micro <= 1:
                            raise
                        chunk_micro = max(1, chunk_micro // 2)
                        if rank == 0:
                            print(
                                "[self-play] CUDA OOM scoring moves; "
                                f"retrying with micro_batch={chunk_micro}."
                            )
                        oom_retries_left -= 1
                else:
                    raise RuntimeError(
                        "CUDA OOM scoring moves after repeated micro_batch shrink"
                    )
        finally:
            restore_policy_train_mode(agent.model)
    return scores


def generate_self_play_enemy_move(
    agent: "XiangqiAgent",
    env: gym.Env,
    flipped_enemy_move_desc: Optional[str],
    episode: int,
    round_idx: int,
    legality_prune_series: Optional[List[float]] = None,
) -> int:
    """Generate the enemy's move using the 'enemy' adapter on the flipped board."""
    # 1. Activate 'enemy' adapter on PeftModel
    peft_model = unwrap_model(agent.model)
    if hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter("enemy")

    try:
        # 2. Rank-0 prepares the query/response batch
        query_ids_batch = None
        response_ids_batch = None
        flipped_action_list = []
        flipped_to_original = {}
        legal_moves_hint = None

        if rank == 0:
            legal_actions, mask_applied, _engine_count, gym_before = (
                apply_pikafish_enemy_legal_mask(
                    env.state, env, agent.pikafish_evaluator
                )
            )
            pf_count = len(legal_actions)
            if mask_applied and gym_before > 0:
                prune_rate = 1.0 - (pf_count / gym_before)
                if legality_prune_series is not None:
                    legality_prune_series.append(prune_rate)
                if prune_rate > 0.0:
                    print(
                        f"[self-play opponent] Pikafish mask gym={gym_before} "
                        f"pf={pf_count} pruned={gym_before - pf_count} "
                        f"({100.0 * prune_rate:.1f}%)",
                        flush=True,
                    )
            if pf_count == 0:
                raise RuntimeError(
                    "[self-play] No Pikafish-legal enemy actions after mask."
                )

            flipped_action_list, flipped_to_original = get_flipped_enemy_legal_actions(
                env
            )
            if len(flipped_action_list) == 0:
                raise RuntimeError(
                    "[self-play] No legal enemy actions on flipped board."
                )

            # Prepare hint in algebraic moves for the model
            hint_actions = list(flipped_action_list)
            random.shuffle(hint_actions)
            legal_moves_hint = [action_to_algebraic(int(a)) for a in hint_actions]

            flipped_board = -env.state[::-1, :]
            messages = agent.format_turn_prompt(
                flipped_board,
                flipped_enemy_move_desc,
                legal_moves_hint=legal_moves_hint,
            )
            prompt_text = agent.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = agent.tokenizer(prompt_text, return_tensors="pt")
            if encoded.input_ids.size(1) > agent.max_prompt_length:
                encoded.input_ids = encoded.input_ids[:, -agent.max_prompt_length :]
            query_ids = encoded.input_ids[0].cpu()

            move_probe_texts = [
                f"Move: {action_to_algebraic(action)}" for action in flipped_action_list
            ]
            response_ids_batch = [
                agent.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .cpu()
                for text in move_probe_texts
            ]
            query_ids_batch = [query_ids for _ in response_ids_batch]

        # 3. Evaluate the move logprobs collectively
        scores = evaluate_moves_logprobs(agent, query_ids_batch, response_ids_batch)

        # 4. Rank 0 selects the best action and translates it back
        chosen_original_action_id = 0
        if rank == 0:
            if not scores or len(scores) != len(flipped_action_list):
                # Fallback to random if scoring failed or returned wrong length
                chosen_original_action_id = int(
                    random.choice(list(flipped_to_original.values()))
                )
            else:
                score_arr = np.array(scores, dtype=np.float64)
                # Select the highest probability move (argmax)
                best_idx = int(np.argmax(score_arr))
                chosen_flipped_action = flipped_action_list[best_idx]
                chosen_original_action_id = flipped_to_original[chosen_flipped_action]

                chosen_move_str_flipped = action_to_algebraic(chosen_flipped_action)
                chosen_move_str_original = action_to_algebraic(
                    chosen_original_action_id
                )
                print(
                    f"[self-play opponent] scored {len(flipped_action_list)} legal moves; "
                    f"highest policy score move: {chosen_move_str_flipped} (lp={score_arr[best_idx]:.4f}) -> "
                    f"original board move: {chosen_move_str_original}",
                    flush=True,
                )

        # 5. Broadcast the chosen original action to all ranks
        chosen_original_action_id = broadcast_int(chosen_original_action_id)

    finally:
        # 6. Set back the active adapter to 'default'
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")

    return chosen_original_action_id


class GreedyEnemyAgent:
    """Capture-greedy opponent with optional ε-random softening.

    With probability ``epsilon`` a uniformly random legal action is selected
    instead of the highest-value capture (ties on the greedy branch are
    already broken at random). ``epsilon=0`` reproduces the original
    deterministic-greedy behaviour. The caller is responsible for annealing
    epsilon across episodes; see ``_current_enemy_epsilon`` below.
    """

    def move(self, env: gym.Env, epsilon: float = 0.0) -> Tuple[int, str]:
        """Return ``(action_id, policy_tag)`` where ``policy_tag`` is ``random`` or ``greedy``.

        The tag is for logging only: the greedy branch still breaks ties among
        equal-capture moves with ``random.choice``, but that counts as
        ``greedy`` (ε-exploration did not fire).
        """
        actions = np.where(env.enemy_actions == 1)[0]
        if len(actions) == 0:
            raise RuntimeError("GreedyEnemyAgent: no legal enemy moves")
        epsilon = float(max(0.0, min(1.0, epsilon)))
        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice([int(a) for a in actions])), "random"
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
        return int(random.choice(best_actions)), "greedy"


def _current_enemy_epsilon(
    episode: int,
    anchor_episode: int,
    eps_start: float,
    eps_end: float,
    anneal_episodes: int,
) -> float:
    """Linear-decay ε for ``GreedyEnemyAgent`` over episodes.

    ``phase = (episode - anchor_episode) / anneal_episodes`` clipped to
    [0, 1]; epsilon = ``eps_start * (1 - phase) + eps_end * phase``. The
    anchor defaults to the run's start_episode at launch so resumed runs
    start at "stage 0" of the curriculum.
    """
    if anneal_episodes <= 0:
        return float(max(0.0, min(1.0, eps_end)))
    phase = (int(episode) - int(anchor_episode)) / float(max(1, anneal_episodes))
    phase = max(0.0, min(1.0, phase))
    eps = float(eps_start) * (1.0 - phase) + float(eps_end) * phase
    return float(max(0.0, min(1.0, eps)))


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
                print(
                    "[optimizer] bitsandbytes unavailable, falling back to torch AdamW."
                )
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
        min_batch_reward_std: float = 0.0,
        engine_policy_align_coef: float = 0.0,
        engine_policy_align_temperature: float = 0.5,
        kl_reference: str = "base",
        train_move_tokens_only: bool = False,
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
        self.min_batch_reward_std = max(0.0, float(min_batch_reward_std))
        self.engine_policy_align_coef = max(0.0, float(engine_policy_align_coef))
        self.engine_policy_align_temperature = max(
            1e-3, float(engine_policy_align_temperature)
        )
        self.kl_reference = str(kl_reference or "base").strip().lower()
        self.train_move_tokens_only = bool(train_move_tokens_only)
        self.optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name)

        unwrapped = unwrap_model(model)
        if hasattr(unwrapped, "gradient_checkpointing_enable"):
            unwrapped.gradient_checkpointing_enable()
        # Required for gradient checkpointing + frozen base + LoRA: otherwise the
        # first non-checkpoint segment can disconnect logits from trainable adapters.
        if hasattr(unwrapped, "enable_input_require_grads"):
            unwrapped.enable_input_require_grads()

        self.mfu_tracker = MFUTracker(
            model,
            device_index=device_obj.index or 0,
            mp_policy=mp_policy,
            gradient_checkpointing=getattr(
                unwrapped, "is_gradient_checkpointing", False
            ),
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
        token_log_probs = log_probs.gather(
            1, response_ids.to(self.device).unsqueeze(1)
        ).squeeze(1)
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

        input_ids = torch.full(
            (G, max_total), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (G, max_total), dtype=torch.long, device=self.device
        )
        resp_tokens_padded = torch.zeros(
            (G, max_total), dtype=torch.long, device=self.device
        )
        resp_lens = torch.tensor(r_lens, dtype=torch.long, device=self.device)

        for i, (q, r) in enumerate(zip(queries, responses)):
            seq = torch.cat([q, r]).to(self.device, dtype=torch.long)
            L = seq.numel()
            input_ids[i, max_total - L :] = seq
            attention_mask[i, max_total - L :] = 1
            resp_tokens_padded[i, max_total - r.numel() :] = r.to(
                self.device, dtype=torch.long
            )

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Shift by one so that column ``t-1`` predicts token at position ``t``.
        shifted = logits[:, :-1, :]
        log_probs = F.log_softmax(shifted.float(), dim=-1)
        # Align target tokens with the shifted positions.
        target = resp_tokens_padded[:, 1:]
        token_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(
            -1
        )  # (G, max_total - 1)
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
        """Best-effort adapter toggle for older PEFT builds.

        Prefer :meth:`_reference_policy_context` for KL reference forwards.
        """

        unwrapped = unwrap_model(self.model)
        fn_name = "enable_adapter_layers" if enable else "disable_adapter_layers"
        if hasattr(unwrapped, fn_name):
            getattr(unwrapped, fn_name)()

    @contextmanager
    def _reference_policy_context(self, unwrapped) -> Iterator[None]:
        """Context for KL reference log-probs: frozen ``sft_ref`` or base-only."""
        peft_config = getattr(unwrapped, "peft_config", {}) or {}
        if (
            self.kl_reference == "sft_ref"
            and hasattr(unwrapped, "set_adapter")
            and "sft_ref" in peft_config
        ):
            active = getattr(unwrapped, "active_adapter", "default")
            if isinstance(active, list):
                active = active[0] if active else "default"
            unwrapped.set_adapter("sft_ref")
            try:
                yield
            finally:
                unwrapped.set_adapter(active)
            return

        if hasattr(unwrapped, "disable_adapter"):
            with unwrapped.disable_adapter():
                yield
            return

        self._toggle_adapters(enable=False)
        try:
            yield
        finally:
            self._toggle_adapters(enable=True)

    def _candidate_align_logit(
        self,
        token_lp: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar policy logit for one candidate (sequential alignment path)."""
        think_start, move_start, move_end = _find_region_token_indices(
            self.tokenizer, response_ids
        )
        if move_start is not None and move_end is not None:
            return token_lp[move_start:move_end].sum()
        return token_lp.sum()

    def _backward_engine_policy_alignment(
        self,
        *,
        query_ids_batch: List[torch.Tensor],
        response_ids_batch: List[torch.Tensor],
        align_valid_mask_dev: torch.Tensor,
        align_target_probs: torch.Tensor,
    ) -> Tuple[bool, float, float, float]:
        """Engine alignment via sequential forwards (OOM-safe after GRPO backward).

        Phase 1: no-grad per-candidate logits -> softmax KL coefficients
        (d KL / d z_i = policy_i - target_i).
        Phase 2: one grad forward per valid candidate; accumulate (coef_i * z_i).
        """
        if not query_ids_batch or not response_ids_batch:
            return False, 0.0, 0.0, 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        try:
            logits_vals: List[float] = []
            for q, r in zip(query_ids_batch, response_ids_batch):
                with torch.no_grad():
                    tok_lp, _, _ = self._compute_response_log_probs(q, r)
                    logit = self._candidate_align_logit(tok_lp, r)
                    logits_vals.append(float(logit.item()))

            logits_all = torch.tensor(
                logits_vals, device=self.device, dtype=torch.float32
            )
            align_logits_valid = logits_all[align_valid_mask_dev]
            if align_logits_valid.numel() < 2:
                return False, 0.0, 0.0, 0.0

            with torch.no_grad():
                align_log_probs = F.log_softmax(align_logits_valid, dim=0)
                align_probs = align_log_probs.exp()
                align_kl = F.kl_div(
                    align_log_probs,
                    align_target_probs,
                    reduction="sum",
                )
                align_entropy = float(-(align_probs * align_log_probs).sum().item())
                align_loss_val = float(
                    (self.engine_policy_align_coef * align_kl).item()
                )
                align_kl_val = float(align_kl.item())
                coeffs = self.engine_policy_align_coef * (
                    align_probs - align_target_probs
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            valid_indices = align_valid_mask_dev.nonzero(as_tuple=False).view(-1)
            for local_i, global_i in enumerate(valid_indices.tolist()):
                q = query_ids_batch[int(global_i)]
                r = response_ids_batch[int(global_i)]
                tok_lp, _, _ = self._compute_response_log_probs(q, r)
                z = self._candidate_align_logit(tok_lp, r)
                coef = coeffs[local_i]
                if float(coef.abs().item()) < 1e-12:
                    continue
                (coef * z).backward()

            return True, align_loss_val, align_kl_val, align_entropy
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(
                    "[GRPO] CUDA OOM during engine-policy alignment; "
                    "skipping alignment backward (GRPO grads kept).",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, 0.0, 0.0, 0.0

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

    def _broadcast_optional_float_vector(
        self,
        values: Optional[List[float]],
        expected_len: int,
    ) -> Optional[torch.Tensor]:
        """Broadcast an optional rank-0 float vector; NaN marks invalid rows."""
        if not dist.is_initialized():
            if values is None or len(values) != expected_len:
                return None
            return torch.tensor(values, dtype=torch.float32)

        has_values_local = bool(values is not None and len(values) == expected_len)
        has_values = broadcast_bool(has_values_local, src=0)
        if not has_values:
            return None
        if rank == 0:
            tensor = torch.tensor(values, dtype=torch.float32, device=device)
        else:
            tensor = torch.zeros(expected_len, dtype=torch.float32, device=device)
        dist.broadcast(tensor, src=0)
        return tensor.cpu()

    def train_group(
        self,
        query_ids_batch: Optional[List[torch.Tensor]],
        response_ids_batch: Optional[List[torch.Tensor]],
        rewards: Optional[List[float]],
        engine_policy_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        if rank == 0:
            should_train_local = bool(
                query_ids_batch and response_ids_batch and rewards and len(rewards) > 0
            )
        else:
            should_train_local = False

        should_train = (
            broadcast_bool(should_train_local)
            if dist.is_initialized()
            else should_train_local
        )
        if not should_train:
            return {}

        if dist.is_initialized():
            query_ids_batch, response_ids_batch, rewards_t = self._broadcast_group(
                query_ids_batch, response_ids_batch, rewards
            )
        else:
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
        n = len(query_ids_batch)

        engine_scores_t = self._broadcast_optional_float_vector(
            engine_policy_scores,
            expected_len=n,
        )
        align_valid_mask_cpu: Optional[torch.Tensor] = None
        if engine_scores_t is not None and self.engine_policy_align_coef > 0.0:
            align_valid_mask_cpu = torch.isfinite(engine_scores_t)
            if int(align_valid_mask_cpu.sum().item()) < 2:
                align_valid_mask_cpu = None
                engine_scores_t = None
        else:
            engine_scores_t = None

        reward_std = float(rewards_t.std().item())
        skip_grpo_for_low_std = (
            self.min_batch_reward_std > 0.0 and reward_std < self.min_batch_reward_std
        )
        # Gate uninformative batches: when ``reward_std`` is below the
        # configured threshold, the within-group advantages are tiny and the
        # ``beta * KL`` term dominates the loss -- which actively pulls the
        # policy back toward the KL reference (i.e. *unlearns*). Skip the
        # GRPO part. If engine-policy alignment targets are present, still run
        # an alignment-only optimizer step so the turn is not wasted.
        if skip_grpo_for_low_std and engine_scores_t is None:
            mean_reward_val = float(rewards_t.mean().item())
            if rank == 0:
                print(
                    f"[GRPO] Skipping optimizer step: reward_std={reward_std:.4f}"
                    f" < min_batch_reward_std={self.min_batch_reward_std:.4f} "
                    f"(mean_reward={mean_reward_val:.4f}, n={len(rewards_t)}).",
                    flush=True,
                )
            return {
                "grpo/loss": 0.0,
                "grpo/mean_advantage": 0.0,
                "grpo/mean_kl": 0.0,
                "grpo/mean_kl_per_token": 0.0,
                "grpo/mean_kl_think": 0.0,
                "grpo/mean_kl_move": 0.0,
                "grpo/policy_entropy_move": 0.0,
                "grpo/pg_clip_frac": 0.0,
                "grpo/ratio_mean": 1.0,
                "grpo/ppo_epochs_completed": 0.0,
                "grpo/mean_reward": mean_reward_val,
                "grpo/batch_reward_std": reward_std,
                "grpo/samples_ok": 0.0,
                "grpo/samples_total": float(len(rewards_t)),
                "grpo/samples_skipped": float(len(rewards_t)),
                "grpo/skipped_low_reward_std": 1.0,
                "grpo/grad_norm_pre_clip": 0.0,
                "grpo/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "grpo/update_applied": 0.0,
                "grpo/engine_align_loss": 0.0,
                "grpo/engine_align_kl": 0.0,
                "grpo/engine_align_entropy": 0.0,
                "grpo/engine_align_target_entropy": 0.0,
                "grpo/engine_align_valid_count": 0.0,
            }
        if skip_grpo_for_low_std and engine_scores_t is not None and rank == 0:
            print(
                f"[GRPO] Skipping policy-gradient term: reward_std={reward_std:.4f}"
                f" < min_batch_reward_std={self.min_batch_reward_std:.4f}; "
                "running engine-policy alignment update.",
                flush=True,
            )
        if reward_std > 1e-6:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)
        else:
            advantages = rewards_t - rewards_t.mean()

        total_tokens = sum(
            q.numel() + r.numel() for q, r in zip(query_ids_batch, response_ids_batch)
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        restore_policy_train_mode(self.model)
        self.optimizer.zero_grad()

        advantages_dev = advantages.to(self.device)
        use_grpo_loss = not skip_grpo_for_low_std
        align_active = engine_scores_t is not None and align_valid_mask_cpu is not None
        engine_scores_dev: Optional[torch.Tensor] = None
        align_valid_mask_dev: Optional[torch.Tensor] = None
        align_target_probs: Optional[torch.Tensor] = None
        align_target_entropy = 0.0
        if align_active:
            engine_scores_dev = engine_scores_t.to(self.device)
            align_valid_mask_dev = align_valid_mask_cpu.to(self.device)
            valid_scores = engine_scores_dev[align_valid_mask_dev]
            target_logits = valid_scores / self.engine_policy_align_temperature
            align_target_probs = F.softmax(target_logits, dim=0).detach()
            align_target_entropy = float(
                -(align_target_probs * torch.log(align_target_probs.clamp_min(1e-12)))
                .sum()
                .item()
            )

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
                        unwrapped = unwrap_model(self.model)
                        with self._reference_policy_context(unwrapped):
                            ref_out = self._compute_response_log_probs_batch(qs, rs)
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
        total_engine_align_loss = 0.0
        total_engine_align_kl = 0.0
        total_engine_align_entropy = 0.0
        engine_align_steps = 0
        samples_ok = 0
        epochs_completed = 0
        # Max L2 norm of the trainable-parameter grad across PPO epochs *before*
        # clip_grad_norm_ rescales it. Surfaced as ``grpo/grad_norm_pre_clip``
        # so we can verify ``grpo/max_grad_norm`` isn't silently throttling
        # every update (the dominant suspect for the previous flat KL_move).
        grad_norm_pre_clip_max = 0.0

        eps_low = self.clip_eps_low
        eps_high = self.clip_eps_high
        beta = self.beta
        ent_coef = self.entropy_coef_move

        if not use_sequential_fallback and caches is not None:
            need_entropy = ent_coef != 0.0 and use_grpo_loss
            if not use_grpo_loss:
                # Alignment-only (flat reward_std): skip batched PPO forwards.
                self.optimizer.zero_grad()
                samples_ok = n
                did_backward = False
                if (
                    align_active
                    and align_valid_mask_dev is not None
                    and align_target_probs is not None
                ):
                    align_ok, align_loss_val, align_kl_val, align_ent = (
                        self._backward_engine_policy_alignment(
                            query_ids_batch=query_ids_batch,
                            response_ids_batch=response_ids_batch,
                            align_valid_mask_dev=align_valid_mask_dev,
                            align_target_probs=align_target_probs,
                        )
                    )
                    if align_ok:
                        did_backward = True
                        total_engine_align_loss += align_loss_val
                        total_engine_align_kl += align_kl_val
                        total_engine_align_entropy += align_ent
                        engine_align_steps += 1
                if did_backward:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad),
                        self.max_grad_norm,
                    )
                    try:
                        grad_norm_pre_clip_max = max(
                            grad_norm_pre_clip_max, float(pre_clip_norm)
                        )
                    except (TypeError, ValueError):
                        pass
                    self.optimizer.step()
            else:
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
                        resp_lens_fl = (
                            entry["resp_lens"].to(cur_tok_lp.dtype).clamp_min(1.0)
                        )
                        if self.train_move_tokens_only:
                            train_mask = entry["move_mask"]
                            train_lens_fl = (
                                train_mask.sum(dim=1)
                                .to(cur_tok_lp.dtype)
                                .clamp_min(1.0)
                            )
                        else:
                            train_mask = resp_mask
                            train_lens_fl = resp_lens_fl
                        ref_tok_lp = entry["ref_tok_lp"]
                        cur_old = entry["cur_tok_lp_old"]
                        adv_slice = advantages_dev[entry["start"] : entry["end"]]

                        # Importance ratio against the pre-update policy (PPO).
                        ratio_raw = torch.exp(cur_tok_lp - cur_old)
                        ratio = ratio_raw * train_mask

                        adv_tok = adv_slice.unsqueeze(1).expand_as(ratio)
                        surr1 = ratio * adv_tok
                        surr2 = (
                            torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv_tok
                        )
                        pg_tok = -torch.min(surr1, surr2)
                        pg_per_sample = (pg_tok * train_mask).sum(dim=1) / train_lens_fl

                        # k3 KL estimator against the (frozen) reference policy.
                        diff = ref_tok_lp - cur_tok_lp  # log(pi_ref / pi_cur)
                        kl_tok = torch.exp(diff) - diff - 1.0
                        kl_tok = kl_tok * train_mask
                        kl_per_sample = kl_tok.sum(dim=1) / train_lens_fl

                        # Entropy bonus on the Move: region.
                        move_entropy_mean = fwd["move_entropy_mean"]
                        if move_entropy_mean is None:
                            entropy_bonus_per_sample = torch.zeros_like(pg_per_sample)
                            move_entropy_for_log = torch.zeros_like(pg_per_sample)
                        else:
                            move_entropy_for_log = move_entropy_mean
                            entropy_bonus_per_sample = -ent_coef * move_entropy_mean

                        if use_grpo_loss:
                            loss_per = (
                                pg_per_sample
                                + beta * kl_per_sample
                                + entropy_bonus_per_sample
                            )
                            group_loss = loss_per.sum() / n
                            group_loss.backward()
                        else:
                            group_loss = torch.zeros((), device=self.device)

                        with torch.no_grad():
                            if use_grpo_loss:
                                total_loss += float(group_loss.item()) / max(
                                    1, self.ppo_epochs
                                )
                            # Per-token mean KL across all response tokens.
                            total_kl_per_token_sum += float(kl_tok.sum().item())
                            total_response_tokens += float(resp_mask.sum().item())
                            # Per-region KL.
                            think_mask_e = entry["think_mask"]
                            move_mask_e = entry["move_mask"]
                            if think_mask_e.sum() > 0:
                                total_kl_think_sum += float(
                                    (kl_tok * think_mask_e).sum().item()
                                )
                                total_think_tokens += float(think_mask_e.sum().item())
                            if move_mask_e.sum() > 0:
                                total_kl_move_sum += float(
                                    (kl_tok * move_mask_e).sum().item()
                                )
                                total_move_tokens += float(move_mask_e.sum().item())
                            # Entropy accumulator (per-sample mean, averaged across batch).
                            if move_entropy_mean is not None:
                                total_entropy_move_sum += float(
                                    move_entropy_for_log.sum().item()
                                )
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

                        epoch_samples_ok += entry["end"] - entry["start"]

                    if epoch_oom:
                        use_sequential_fallback = True
                        break

                    did_backward = use_grpo_loss and epoch_samples_ok > 0
                    if (
                        align_active
                        and align_valid_mask_dev is not None
                        and align_target_probs is not None
                        and epoch_samples_ok > 0
                    ):
                        align_ok, align_loss_val, align_kl_val, align_ent = (
                            self._backward_engine_policy_alignment(
                                query_ids_batch=query_ids_batch,
                                response_ids_batch=response_ids_batch,
                                align_valid_mask_dev=align_valid_mask_dev,
                                align_target_probs=align_target_probs,
                            )
                        )
                        if align_ok:
                            did_backward = True
                            total_engine_align_loss += align_loss_val
                            total_engine_align_kl += align_kl_val
                            total_engine_align_entropy += align_ent
                            engine_align_steps += 1

                    if not did_backward:
                        continue

                    try:
                        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            self.max_grad_norm,
                        )
                        try:
                            grad_norm_pre_clip_max = max(
                                grad_norm_pre_clip_max, float(pre_clip_norm)
                            )
                        except (TypeError, ValueError):
                            pass
                        self.optimizer.step()
                    except torch.cuda.OutOfMemoryError:
                        if rank == 0:
                            print(
                                "[GRPO] CUDA OOM during optimizer step; "
                                "falling back to sequential path.",
                                flush=True,
                            )
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        use_sequential_fallback = True
                        break
                    epochs_completed += 1
                    samples_ok = max(samples_ok, epoch_samples_ok)

        if use_sequential_fallback:
            # Fallback path: fully sequential, per-sample, single epoch, no
            # PPO clip (OOM-safe). Keeps the k3 KL estimator for consistency.
            if rank == 0:
                print(
                    "[GRPO] Falling back to fully sequential per-sample log-prob pass."
                )
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
                    unwrapped = unwrap_model(self.model)
                    with torch.no_grad():
                        with self._reference_policy_context(unwrapped):
                            ref_tok_lp, _, r_len = self._compute_response_log_probs(
                                q, r
                            )
                    cur_tok_lp, _, _ = self._compute_response_log_probs(q, r)
                    if self.train_move_tokens_only:
                        _, _, move_start = _find_region_token_indices(self.tokenizer, r)
                        r_len = int(r.numel())
                        train_mask = torch.zeros(
                            r_len, dtype=cur_tok_lp.dtype, device=cur_tok_lp.device
                        )
                        if move_start is not None and move_start < r_len:
                            train_mask[move_start:] = 1.0
                        else:
                            train_mask[:] = 1.0
                        train_len_fl = max(1.0, float(train_mask.sum().item()))
                        cur_lp_mean = (cur_tok_lp * train_mask).sum() / train_len_fl
                        diff = ref_tok_lp - cur_tok_lp
                        kl_tok = (torch.exp(diff) - diff - 1.0) * train_mask
                        kl_per_sample = kl_tok.sum() / train_len_fl
                    else:
                        r_len_fl = max(1.0, float(r.numel()))
                        diff = ref_tok_lp - cur_tok_lp
                        kl_tok = torch.exp(diff) - diff - 1.0
                        kl_per_sample = kl_tok.sum() / r_len_fl
                        cur_lp_mean = cur_tok_lp.sum() / r_len_fl
                    if use_grpo_loss:
                        sample_loss = (-adv * cur_lp_mean + beta * kl_per_sample) / n
                        sample_loss.backward()
                        total_loss += float(sample_loss.item())
                    total_kl_per_token_sum += float(kl_tok.sum().item())
                    total_response_tokens += float(
                        train_mask.sum().item()
                        if self.train_move_tokens_only
                        else r.numel()
                    )
                    samples_ok += 1
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"[GRPO] CUDA OOM on sample {idx + 1}/{n}; skipping.")
                    self._toggle_adapters(enable=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            did_backward = use_grpo_loss and samples_ok > 0
            if (
                align_active
                and align_valid_mask_dev is not None
                and align_target_probs is not None
                and samples_ok > 0
            ):
                align_ok, align_loss_val, align_kl_val, align_ent = (
                    self._backward_engine_policy_alignment(
                        query_ids_batch=query_ids_batch,
                        response_ids_batch=response_ids_batch,
                        align_valid_mask_dev=align_valid_mask_dev,
                        align_target_probs=align_target_probs,
                    )
                )
                if align_ok:
                    did_backward = True
                    total_engine_align_loss += align_loss_val
                    total_engine_align_kl += align_kl_val
                    total_engine_align_entropy += align_ent
                    engine_align_steps += 1

            if samples_ok > 0 and did_backward:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.max_grad_norm,
                )
                try:
                    grad_norm_pre_clip_max = max(
                        grad_norm_pre_clip_max, float(pre_clip_norm)
                    )
                except (TypeError, ValueError):
                    pass
                self.optimizer.step()
                epochs_completed = 1

        if samples_ok == 0:
            self.optimizer.zero_grad()
            return {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # MFU accounting: training forwards + ref/cur_old caches (+ optional align pass).
        align_extra_fwd = 1 if engine_align_steps > 0 else 0
        num_fwd_per_sample = (
            int(epochs_completed) + 2 + align_extra_fwd
            if not use_sequential_fallback
            else 2 + align_extra_fwd
        )
        mfu_stats = self.mfu_tracker.compute(
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            num_fwd_per_sample=max(2, num_fwd_per_sample),
        )
        mem_alloc = (
            (torch.cuda.memory_allocated(device) / 1e9)
            if torch.cuda.is_available()
            else 0.0
        )
        mem_res = (
            (torch.cuda.memory_reserved(device) / 1e9)
            if torch.cuda.is_available()
            else 0.0
        )

        mean_kl_per_token = total_kl_per_token_sum / max(1.0, total_response_tokens)
        mean_kl_think = (
            total_kl_think_sum / total_think_tokens if total_think_tokens > 0 else 0.0
        )
        mean_kl_move = (
            total_kl_move_sum / total_move_tokens if total_move_tokens > 0 else 0.0
        )
        mean_entropy_move = (
            total_entropy_move_sum / total_entropy_samples
            if total_entropy_samples > 0
            else 0.0
        )
        mean_engine_align_loss = (
            total_engine_align_loss / engine_align_steps
            if engine_align_steps > 0
            else 0.0
        )
        mean_engine_align_kl = (
            total_engine_align_kl / engine_align_steps
            if engine_align_steps > 0
            else 0.0
        )
        mean_engine_align_entropy = (
            total_engine_align_entropy / engine_align_steps
            if engine_align_steps > 0
            else 0.0
        )
        ratio_tokens_seen = float(total_response_tokens)
        ratio_mean = (
            total_ratio_sum / ratio_tokens_seen if ratio_tokens_seen > 0 else 0.0
        )
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
            "grpo/samples_skipped": float(n) if skip_grpo_for_low_std else 0.0,
            "grpo/skipped_low_reward_std": 1.0 if skip_grpo_for_low_std else 0.0,
            "grpo/grad_norm_pre_clip": float(grad_norm_pre_clip_max),
            "grpo/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "grpo/update_applied": 1.0
            if (epochs_completed > 0 or engine_align_steps > 0)
            else 0.0,
            "grpo/engine_align_loss": float(mean_engine_align_loss),
            "grpo/engine_align_kl": float(mean_engine_align_kl),
            "grpo/engine_align_entropy": float(mean_engine_align_entropy),
            "grpo/engine_align_target_entropy": float(align_target_entropy),
            "grpo/engine_align_valid_count": float(
                int(align_valid_mask_cpu.sum().item())
                if align_valid_mask_cpu is not None
                else 0
            ),
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
    terminal_query_batch: Optional[List[torch.Tensor]] = None
    terminal_response_batch: Optional[List[torch.Tensor]] = None
    terminal_reward_batch: Optional[List[float]] = None
    terminal_engine_policy_scores: Optional[List[float]] = None
    terminal_chosen_index: Optional[int] = None
    terminal_train_stats: Optional[Dict[str, float]] = None


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
        legal_anchor_count: int = 0,
        use_legal_move_sampler: bool = False,
        legal_move_sample_temperature: float = 1.0,
        legal_move_sample_epsilon: float = 0.05,
        legal_move_action_selection: str = "greedy",
        play_best_candidate: bool = True,
        reward_format_mix_mode: str = "mix",
        grounding_quality_min: float = 0.42,
        format_gate_fail_scale: float = 0.08,
        format_soft_fail_scale: float = 0.45,
        reward_grounding_strict_when_sampler: bool = True,
        generate_grounded_reasoning_for_sampler: bool = True,
        sampler_grounding_max_new_tokens: int = 192,
        sampler_grounding_temperature: float = 0.75,
        regen_generate_overrides: Optional[Dict[str, Any]] = None,
        combine_gate_with_r_best: bool = False,
        tactical_weight: float = 1.5,
        sigma_good_cp: float = SIGMA_GOOD,
        move_only: bool = False,
        reward_engine_only: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grpo_trainer = grpo_trainer
        self.move_only = bool(move_only)
        self.reward_engine_only = bool(reward_engine_only)
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.max_train_query_ctx = max_train_query_ctx
        self.generate_config = generate_config
        self.num_generations = num_generations
        self.pikafish_evaluator = pikafish_evaluator
        self.reward_cp_scale = float(reward_cp_scale)
        self.reward_format_weight = float(reward_format_weight)
        self.reward_format_weight_min = float(reward_format_weight_min)
        self.reward_format_weight_anneal_start = float(
            reward_format_weight_anneal_start
        )
        self.reward_format_weight_anneal_end = float(reward_format_weight_anneal_end)
        self.reward_format_compliance_ema_alpha = float(
            reward_format_compliance_ema_alpha
        )
        self.min_legal_candidates = max(0, int(min_legal_candidates))
        self.max_regeneration_rounds = max(0, int(max_regeneration_rounds))
        self.regeneration_batch_size = (
            max(1, int(regeneration_batch_size))
            if regeneration_batch_size and regeneration_batch_size > 0
            else max(1, int(num_generations) // 2)
        )
        self.min_distinct_legal_moves = max(0, int(min_distinct_legal_moves))
        self.dedupe_legal_by_move = bool(dedupe_legal_by_move)
        self.legal_anchor_count = max(0, int(legal_anchor_count))
        self.use_legal_move_sampler = bool(use_legal_move_sampler)
        self.legal_move_sample_temperature = max(
            1e-3, float(legal_move_sample_temperature)
        )
        self.legal_move_sample_epsilon = float(
            np.clip(legal_move_sample_epsilon, 0.0, 1.0)
        )
        # ``greedy`` puts the argmax-policy legal move at position 0 of the
        # sampler group so the agent plays its own most-confident legal move
        # while GRPO still trains on the full sampled group. ``first_sample``
        # preserves legacy behavior (play whatever the first weighted draw was).
        _selection = str(legal_move_action_selection).strip().lower()
        if _selection not in {"greedy", "first_sample"}:
            raise ValueError(
                "legal_move_action_selection must be 'greedy' or 'first_sample'; "
                f"got {legal_move_action_selection!r}"
            )
        self.legal_move_action_selection = _selection
        self.play_best_candidate = bool(play_best_candidate)
        self.reward_format_mix_mode = str(reward_format_mix_mode)
        self.grounding_quality_min = float(grounding_quality_min)
        self.format_gate_fail_scale = float(format_gate_fail_scale)
        self.format_soft_fail_scale = float(format_soft_fail_scale)
        self.reward_grounding_strict_when_sampler = bool(
            reward_grounding_strict_when_sampler
        )
        self.generate_grounded_reasoning_for_sampler = bool(
            generate_grounded_reasoning_for_sampler
        )
        if self.move_only:
            self.generate_grounded_reasoning_for_sampler = False
            self.generate_config = {
                **self.generate_config,
                "max_new_tokens": min(
                    int(self.generate_config.get("max_new_tokens", 384)), 32
                ),
            }
        self.sampler_grounding_max_new_tokens = max(
            32, int(sampler_grounding_max_new_tokens)
        )
        self.sampler_grounding_temperature = max(
            1e-3, float(sampler_grounding_temperature)
        )
        # Extra generate(...) kwargs applied only on regeneration passes
        # (pass_idx > 0). Typically widens temperature / top_p to boost
        # exploration once the initial pass lands in a low-diversity basin.
        self.regen_generate_overrides: Dict[str, Any] = dict(
            regen_generate_overrides or {}
        )
        # Combined-reward configuration: when ``combine_gate_with_r_best`` is
        # True the ``gate`` reward path additionally awards a discrete
        # ``tactical_weight * (r_good + r_best)`` bonus per candidate (see
        # ``evaluate_candidate_response``). Default off so existing runs
        # continue to use the smooth tanh-of-cp reward only.
        self.combine_gate_with_r_best = bool(combine_gate_with_r_best)
        self.tactical_weight = float(tactical_weight)
        self.sigma_good_cp = float(sigma_good_cp)
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
                1.0 - alpha
            ) * self._format_compliance_ema + alpha * rate

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
        success_t = torch.tensor(
            [1 if local_success else 0], dtype=torch.long, device=device
        )
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
        if self.move_only:
            return (
                "You are a Xiangqi (Chinese Chess) player. You always play the uppercase side.\n"
                "Piece letters: K=General A=Advisor B=Elephant N=Horse R=Chariot C=Cannon P=Soldier.\n"
                "Coordinates: files a-i (left to right), ranks 0-9 (top to bottom as shown in the graphic).\n"
                "Uppercase (your) pieces start on the BOTTOM (ranks 7-9); lowercase enemy pieces start on the TOP\n"
                "(ranks 0-2). The river sits between ranks 4 and 5 (shown as '~~~' in the graphic).\n"
                "A move is written <from_file><from_rank><to_file><to_rank>, e.g. b7b4 moves the piece on b7 to b4.\n\n"
                "Respond with exactly one line and nothing else:\n"
                "Move: <from><to>"
            )
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
        prefix = (
            f"Enemy previous move: {enemy_move_desc}\n"
            if enemy_move_desc
            else "Enemy previous move: none\n"
        )
        # Hint only a trimmed list to avoid ballooning prompt tokens.
        hint_line = ""
        if legal_moves_hint:
            trimmed = legal_moves_hint[:48]
            more = (
                ""
                if len(legal_moves_hint) <= 48
                else f" (+{len(legal_moves_hint) - 48} more)"
            )
            hint_line = f"Legal moves (subset): {' '.join(trimmed)}{more}\n"
        user_msg = (
            f"{prefix}"
            f"Current board FEN: {fen}\n"
            f"Current board graphic:\n{graphic}\n"
            f"{hint_line}"
            + (
                "Pick the single best legal move for the uppercase side.\n"
                if self.move_only
                else "Pick the single best legal move for the uppercase side and output reasoning + move."
            )
        )
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def format_grounded_move_prompt(
        self,
        board_state: np.ndarray,
        enemy_move_desc: Optional[str],
        legal_moves_hint: Optional[List[str]],
        fixed_move: str,
    ) -> List[Dict[str, str]]:
        """User prompt that fixes the played move so the model only explains *that* move."""
        fen = board_to_fen(board_state)
        graphic = board_to_graphic(board_state)
        prefix = (
            f"Enemy previous move: {enemy_move_desc}\n"
            if enemy_move_desc
            else "Enemy previous move: none\n"
        )
        hint_line = ""
        if legal_moves_hint:
            trimmed = legal_moves_hint[:48]
            more = (
                ""
                if len(legal_moves_hint) <= 48
                else f" (+{len(legal_moves_hint) - 48} more)"
            )
            hint_line = f"Legal moves (subset): {' '.join(trimmed)}{more}\n"
        mv = fixed_move.strip().lower()
        user_msg = (
            f"{prefix}"
            f"Current board FEN: {fen}\n"
            f"Current board graphic:\n{graphic}\n"
            f"{hint_line}"
            f"You have already committed to play Move: {mv} for the uppercase side.\n"
            "In <think>, explain ONLY why this specific move is good: "
            f"reference the exact string {mv} at least once, name the piece type, "
            "and cite the from/to squares using the same coordinate system as the graphic.\n"
            "Do not describe or recommend any other move.\n\n"
            "Respond exactly in this format (two lines, nothing else):\n"
            "<think>your tactical justification for this move only</think>\n"
            f"Move: {mv}"
        )
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _format_move_response(move: str) -> str:
        return f"Move: {move.strip().lower()}"

    def _build_legal_anchor_response(
        self,
        action: int,
        enemy_move_desc: Optional[str],
    ) -> str:
        move = action_to_algebraic(action)
        if self.move_only:
            return self._format_move_response(move)
        piece_id, _, _ = action_space_to_move(int(action))
        piece_name = (
            PIECE_ID_TO_NAME[piece_id].split("_")[0]
            if 0 <= piece_id < len(PIECE_ID_TO_NAME)
            else "piece"
        )
        enemy_clause = (
            f"After the enemy move {enemy_move_desc}, "
            if enemy_move_desc
            else "With no previous enemy move to answer, "
        )
        return (
            "<think>"
            f"{enemy_clause}I will play the listed legal move {move} with my {piece_name}. "
            "This keeps the turn inside the Xiangqi legal-move mask and avoids training on an illegal fallback. "
            "The opponent's most dangerous reply is an immediate capture or check on the opened line."
            "</think>\n"
            f"Move: {move}"
        )

    def _build_policy_sampled_response(
        self,
        action: int,
        enemy_move_desc: Optional[str],
    ) -> str:
        move = action_to_algebraic(action)
        if self.move_only:
            return self._format_move_response(move)
        piece_id, start, end = action_space_to_move(int(action))
        piece_name = (
            PIECE_ID_TO_NAME[piece_id].split("_")[0]
            if 0 <= piece_id < len(PIECE_ID_TO_NAME)
            else "piece"
        )
        enemy_clause = (
            f"After {enemy_move_desc}, "
            if enemy_move_desc
            else "From the current board, "
        )
        return (
            "<think>"
            f"{enemy_clause}I choose the legal move {move}: my {piece_name} moves "
            f"from {COLS[start[1]]}{start[0]} to {COLS[end[1]]}{end[0]}. "
            "This candidate is sampled from the model's distribution over legal Xiangqi moves, "
            "so the reasoning is tied to the move that will be evaluated. "
            "The opponent's most dangerous reply is an immediate capture, check, or block on the same line."
            "</think>\n"
            f"Move: {move}"
        )

    def _generate_grounded_reasoning_for_move(
        self,
        board_state: np.ndarray,
        enemy_move_desc: Optional[str],
        legal_moves_hint: Optional[List[str]],
        action: int,
        episode: int,
        round_idx: int,
    ) -> str:
        """Generate `<think>` + fixed ``Move:`` line for one legal action."""
        move = action_to_algebraic(action)
        messages = self.format_grounded_move_prompt(
            board_state,
            enemy_move_desc,
            legal_moves_hint,
            fixed_move=move,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        if encoded.input_ids.size(1) > self.max_prompt_length:
            encoded.input_ids = encoded.input_ids[:, -self.max_prompt_length :]
            encoded.attention_mask = encoded.attention_mask[
                :, -self.max_prompt_length :
            ]

        decode_out = self._generate_one_batch(
            encoded=encoded,
            num_generations=1,
            max_new_tokens_override=self.sampler_grounding_max_new_tokens,
            episode=episode,
            round_idx=round_idx,
            pass_label="grounding",
            generate_override={
                "temperature": self.sampler_grounding_temperature,
                "do_sample": True,
            },
        )
        decoded_batch = decode_out[0]
        if not decoded_batch:
            return self._build_policy_sampled_response(action, enemy_move_desc)
        text = (decoded_batch[0] or "").strip()
        parsed = _extract_move(text)
        if parsed is None or parsed.lower() != move.lower():
            return self._build_policy_sampled_response(action, enemy_move_desc)
        return text

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
                encoded.attention_mask = encoded.attention_mask[
                    :, -self.max_prompt_length :
                ]
            query_ids = encoded.input_ids[0].cpu()
            # Full UCI FEN (side to move + counters): Pikafish ``position fen`` and
            # ``bestmove_root_cached`` require this; placement-only strings from
            # ``board_to_fen`` yield ``bestmove (none)`` and ``engine_best_overall=unknown``.
            fen = board_to_uci_fen(board_state, side_to_move="w")
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

        # Build an engine-constrained ``prefix_allowed_tokens_fn`` once per
        # turn (rank 0 only) so regen passes are forced to sample moves from
        # Pikafish's legal action list. For multi-rank we'd need to broadcast
        # the legal move list; single-rank torchrun is the current deployment.
        prompt_len_for_constraint = (
            int(encoded.input_ids.size(1)) if (rank == 0 and encoded is not None) else 0
        )
        move_constraint_fn = None
        if rank == 0 and legal_moves_hint:
            move_constraint_fn = _build_move_constraint_fn(
                legal_moves=list(legal_moves_hint),
                tokenizer=self.tokenizer,
                prompt_len=prompt_len_for_constraint,
            )

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
                    if move_constraint_fn is not None:
                        generate_override = generate_override or {}
                        generate_override["prefix_allowed_tokens_fn"] = (
                            move_constraint_fn
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
                        legal_threshold_met and distinct_threshold_met
                    ) or regen_passes >= self.max_regeneration_rounds
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

    def _generate_policy_sampled_legal_candidates(
        self,
        board_state: Optional[np.ndarray],
        env: Optional[gym.Env],
        enemy_move_desc: Optional[str],
        episode: int,
        round_idx: int,
        legal_actions: Optional[np.ndarray],
        legal_moves_hint: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[str], str, str, int]:
        if rank == 0:
            messages = self.format_turn_prompt(
                board_state,
                enemy_move_desc,
                legal_moves_hint=legal_moves_hint,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            if encoded.input_ids.size(1) > self.max_prompt_length:
                encoded.input_ids = encoded.input_ids[:, -self.max_prompt_length :]
                encoded.attention_mask = encoded.attention_mask[
                    :, -self.max_prompt_length :
                ]
            query_ids = encoded.input_ids[0].cpu()
            # Full UCI FEN for board-sync logs and ``bestmove_root_cached`` (see
            # ``_generate_candidates`` — placement-only FEN breaks Pikafish bestmove).
            fen = board_to_uci_fen(board_state, side_to_move="w")
            graphic = board_to_graphic(board_state)
            action_list = (
                [int(action) for action in list(legal_actions)]
                if legal_actions is not None
                else []
            )
            move_probe_texts = [
                f"Move: {action_to_algebraic(action)}" for action in action_list
            ]
            response_ids_batch = [
                self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .cpu()
                for text in move_probe_texts
            ]
            query_ids_batch = [query_ids for _ in response_ids_batch]
            dummy_rewards = [0.0 for _ in response_ids_batch]
        else:
            query_ids = None
            fen = ""
            graphic = ""
            action_list = []
            query_ids_batch = None
            response_ids_batch = None
            dummy_rewards = None

        if dist.is_initialized():
            synced_q, synced_r, _ = self.grpo_trainer._broadcast_group(
                query_ids_batch,
                response_ids_batch,
                dummy_rewards,
            )
        else:
            synced_q, synced_r = query_ids_batch, response_ids_batch

        scores: List[float] = []
        score_wall_start = time.perf_counter()
        if synced_q and synced_r:
            model_obj = unwrap_model(self.model)
            model_obj.eval()
            n_moves = len(synced_q)
            micro = max(1, int(self.grpo_trainer.logprob_micro_batch))
            chunk_micro = micro
            oom_retries_left = 3
            try:
                with torch.no_grad():
                    while oom_retries_left >= 0:
                        try:
                            score_chunks: List[float] = []
                            for s in range(0, n_moves, chunk_micro):
                                e = min(s + chunk_micro, n_moves)
                                qs = [synced_q[i] for i in range(s, e)]
                                rs = [synced_r[i] for i in range(s, e)]
                                fwd = (
                                    self.grpo_trainer._compute_response_log_probs_batch(
                                        qs, rs
                                    )
                                )
                                token_lp = fwd["token_lp"]
                                response_mask = fwd["resp_mask"]
                                scores_t = (
                                    (token_lp * response_mask)
                                    .sum(dim=1)
                                    .detach()
                                    .float()
                                )
                                score_chunks.extend(float(x) for x in scores_t.tolist())
                            scores = score_chunks
                            break
                        except torch.cuda.OutOfMemoryError:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if chunk_micro <= 1:
                                raise
                            chunk_micro = max(1, chunk_micro // 2)
                            if rank == 0:
                                print(
                                    "[legal_sampler] CUDA OOM scoring legal moves; "
                                    f"retrying with micro_batch={chunk_micro}."
                                )
                        oom_retries_left -= 1
                    else:
                        raise RuntimeError(
                            "CUDA OOM scoring legal moves after repeated micro_batch shrink"
                        )
            finally:
                restore_policy_train_mode(self.model)
        score_wall_sec = time.perf_counter() - score_wall_start

        if rank == 0:
            if not action_list:
                self._last_generation_stats = {
                    "num_sequences": 0.0,
                    "prompt_len": 0.0,
                    "generated_len": 0.0,
                    "wall_sec": float(score_wall_sec),
                }
                return query_ids, [], fen, graphic, 0

            k = min(self.num_generations, len(action_list))
            score_arr = np.array(scores[: len(action_list)], dtype=np.float64)
            # Per-legal-move summed log p(Move: uci | prompt) — same scores used
            # for the legal-move sampler and GRPO group construction.
            logprob_rows: List[Tuple[str, float]] = []
            for i, act in enumerate(action_list):
                uci = action_to_algebraic(int(act))
                if i < int(score_arr.size):
                    lp = float(score_arr[i])
                else:
                    lp = float("nan")
                logprob_rows.append((uci, lp))
            logprob_rows.sort(key=lambda t: t[1], reverse=True)
            print(
                f"[Ep {episode} Rd {round_idx}] Legal-move policy log-prob sums "
                f"(response tokens only, n={len(logprob_rows)}):",
                flush=True,
            )
            for uci, lp in logprob_rows:
                if np.isfinite(lp):
                    print(f"  {uci}\t{lp:.6f}", flush=True)
                else:
                    print(f"  {uci}\t(nan / missing)", flush=True)

            if score_arr.size != len(action_list) or not np.all(np.isfinite(score_arr)):
                probs = np.full(
                    len(action_list), 1.0 / len(action_list), dtype=np.float64
                )
            else:
                temp = self.legal_move_sample_temperature
                logits = score_arr / temp
                logits -= float(np.max(logits))
                probs = np.exp(logits)
                denom = float(np.sum(probs))
                if denom <= 0.0 or not np.isfinite(denom):
                    probs = np.full(
                        len(action_list), 1.0 / len(action_list), dtype=np.float64
                    )
                else:
                    probs /= denom
                eps = self.legal_move_sample_epsilon
                if eps > 0.0:
                    probs = (1.0 - eps) * probs + eps / len(action_list)
                    probs /= float(np.sum(probs))

            selected_idx = np.random.choice(
                len(action_list),
                size=k,
                replace=False,
                p=probs,
            )

            # On-policy greedy action selection: force the argmax-policy legal
            # move into slot 0 of the sampler group so `choice_pool[0]` becomes
            # the agent's most-confident legal move under the current policy
            # (still its own distribution; no Pikafish oracle used to pick).
            # The remaining sampled moves are unchanged so GRPO sees the same
            # diversity it would have without the swap.
            greedy_played_idx: Optional[int] = None
            if (
                self.legal_move_action_selection == "greedy"
                and score_arr.size == len(action_list)
                and np.all(np.isfinite(score_arr))
            ):
                greedy_idx = int(np.argmax(score_arr))
                sel_list = list(int(i) for i in selected_idx)
                if greedy_idx in sel_list:
                    sel_list.remove(greedy_idx)
                else:
                    # Drop the lowest-policy-prob slot to make room without
                    # growing the group beyond ``k`` (preserves group budget).
                    sel_probs = [probs[i] for i in sel_list]
                    drop_pos = int(np.argmin(sel_probs))
                    sel_list.pop(drop_pos)
                sel_list.insert(0, greedy_idx)
                selected_idx = np.array(sel_list, dtype=np.int64)
                greedy_played_idx = greedy_idx

            selected_actions = [action_list[int(i)] for i in selected_idx]
            ground_wall = 0.0
            if self.generate_grounded_reasoning_for_sampler:
                responses = []
                for action in selected_actions:
                    t0 = time.perf_counter()
                    responses.append(
                        self._generate_grounded_reasoning_for_move(
                            board_state,
                            enemy_move_desc,
                            legal_moves_hint,
                            action,
                            episode,
                            round_idx,
                        )
                    )
                    ground_wall += time.perf_counter() - t0
            else:
                responses = [
                    self._build_policy_sampled_response(action, enemy_move_desc)
                    for action in selected_actions
                ]
            response_lens = [
                int(
                    self.tokenizer(
                        response,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.size(1)
                )
                for response in responses
            ]
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            self._last_generation_stats = {
                "num_sequences": float(len(responses)),
                "prompt_len": float(query_ids.numel()),
                "generated_len": float(np.mean(response_lens))
                if response_lens
                else 0.0,
                "wall_sec": float(score_wall_sec + ground_wall),
                "sampler/grounding_wall_sec": float(ground_wall),
                "legal_action_policy_entropy": entropy,
                "legal_action_policy_top_prob": float(np.max(probs)),
            }
            greedy_suffix = ""
            if greedy_played_idx is not None:
                greedy_move_str = action_to_algebraic(action_list[greedy_played_idx])
                greedy_suffix = (
                    f", played_greedy={greedy_move_str}"
                    f"@p={float(probs[greedy_played_idx]):.3f}"
                )
            print(
                f"[Ep {episode} Rd {round_idx}] Policy-sampled {len(responses)} "
                f"distinct legal moves from {len(action_list)} legal actions "
                f"(top_prob={float(np.max(probs)):.3f}, entropy={entropy:.3f}"
                f"{greedy_suffix}).",
                flush=True,
            )
            return query_ids, responses, fen, graphic, len(responses)

        self._last_generation_stats = {
            "num_sequences": float(len(scores)),
            "prompt_len": 0.0,
            "generated_len": 0.0,
            "wall_sec": float(score_wall_sec),
        }
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
            legal_actions, using_pikafish_legality, engine_legal_count = (
                apply_pikafish_legal_mask(
                    board_state=board_state,
                    env=env,
                    pikafish_evaluator=self.pikafish_evaluator,
                )
            )
            if len(legal_actions) == 0:
                raise RuntimeError("XiangqiAgent: no legal ally moves available")
            # Shuffle to avoid positional bias when we have to trim the hint list.
            hint_actions = list(legal_actions)
            random.shuffle(hint_actions)
            legal_moves_hint = [action_to_algebraic(int(a)) for a in hint_actions]

        if self.use_legal_move_sampler:
            query_ids, responses, fen, graphic, _prefilter_legal_count = (
                self._generate_policy_sampled_legal_candidates(
                    board_state=board_state,
                    env=env,
                    enemy_move_desc=enemy_move_desc,
                    episode=episode,
                    round_idx=round_idx,
                    legal_actions=legal_actions if rank == 0 else None,
                    legal_moves_hint=legal_moves_hint,
                )
            )
        else:
            query_ids, responses, fen, graphic, _prefilter_legal_count = (
                self._generate_candidates(
                    board_state=board_state,
                    env=env,
                    enemy_move_desc=enemy_move_desc,
                    episode=episode,
                    round_idx=round_idx,
                    legal_moves_hint=legal_moves_hint,
                )
            )

        if rank == 0:
            evals: List[CandidateEval] = []
            legal_count = 0
            format_count = 0
            reasoning_count = 0
            move_parsed_count = 0
            move_strings: List[str] = []

            effective_format_weight = self._current_format_weight()
            train_query_ids = query_ids
            if train_query_ids.numel() > self.max_train_query_ctx:
                train_query_ids = train_query_ids[-self.max_train_query_ctx :]

            # ----------------------------------------------------------------
            # Per-turn precompute for the combined ``gate + R_best/R_good``
            # reward (paper §3.4). One Pikafish ``bestmove_root_cached`` call
            # (already cached for the engine-best comparison metric below) and
            # one ``evaluate_cp`` of the best-move position are reused across
            # all 32 candidates, so we add ~1 Pikafish call per turn rather
            # than 32. Only runs when the feature is enabled to keep the
            # default code path identical.
            # ----------------------------------------------------------------
            position_best_uci_engine: Optional[str] = None
            position_vb_best: Optional[float] = None
            if (
                self.combine_gate_with_r_best
                and self.pikafish_evaluator is not None
                and self.pikafish_evaluator.enabled
                and board_state is not None
            ):
                try:
                    _fen_full_for_best = board_to_uci_fen(board_state, side_to_move="w")
                    _bm_uci, _bm_root_cp = self.pikafish_evaluator.bestmove_root_cached(
                        _fen_full_for_best
                    )
                    if isinstance(_bm_uci, str) and _bm_uci:
                        position_best_uci_engine = _bm_uci.lower()
                        position_vb_best = red_value_after_uci_move(
                            _fen_full_for_best,
                            position_best_uci_engine,
                            self.pikafish_evaluator.evaluate_cp,
                        )
                except Exception as _exc:
                    print(
                        f"[Ep {episode} Rd {round_idx}] [reward] failed to "
                        f"precompute position best/vb for combined reward: "
                        f"{_exc!r}; falling back to gate-only this turn.",
                        flush=True,
                    )
                    position_best_uci_engine = None
                    position_vb_best = None
            for response in responses:
                ev = evaluate_candidate_response(
                    response=response,
                    board_before=board_state,
                    env=env,
                    query_ids=train_query_ids,
                    tokenizer=self.tokenizer,
                    enemy_move_desc=enemy_move_desc,
                    pikafish_evaluator=self.pikafish_evaluator,
                    cp_scale=self.reward_cp_scale,
                    format_weight=effective_format_weight,
                    forced_action=None,
                    reward_format_mode=self.reward_format_mix_mode,
                    grounding_strict=self.use_legal_move_sampler
                    and self.reward_grounding_strict_when_sampler,
                    grounding_quality_min=self.grounding_quality_min,
                    format_gate_fail_scale=self.format_gate_fail_scale,
                    format_soft_fail_scale=self.format_soft_fail_scale,
                    combine_gate_with_r_best=self.combine_gate_with_r_best,
                    position_best_uci_engine=position_best_uci_engine,
                    position_vb_best=position_vb_best,
                    tactical_weight=self.tactical_weight,
                    sigma_good_cp=self.sigma_good_cp,
                    engine_only=self.reward_engine_only,
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
                    if not (
                        ev.legal and ev.action is not None and ev.move_str is not None
                    ):
                        continue
                    prev = best_idx_by_move.get(ev.move_str)
                    if prev is None or ev.reward > evals[prev].reward:
                        best_idx_by_move[ev.move_str] = idx_ev
                keep_legal_indices = set(best_idx_by_move.values())
                train_evals: List[CandidateEval] = [
                    ev
                    for i, ev in enumerate(evals)
                    if (not (ev.legal and ev.action is not None))
                    or i in keep_legal_indices
                ]
            else:
                train_evals = list(evals)

            deduped_legal_evals = [
                ev for ev in train_evals if ev.legal and ev.action is not None
            ]
            distinct_legal_move_count = len(
                {ev.move_str for ev in deduped_legal_evals if ev.move_str is not None}
            )
            generated_best_reward = (
                max(ev.reward for ev in deduped_legal_evals)
                if deduped_legal_evals
                else 0.0
            )

            anchor_evals: List[CandidateEval] = []
            anchor_budget = max(
                0,
                min(
                    self.legal_anchor_count,
                    self.min_legal_candidates - len(deduped_legal_evals),
                ),
            )
            if anchor_budget > 0:
                generated_legal_moves = {
                    ev.move_str for ev in deduped_legal_evals if ev.move_str is not None
                }
                anchor_actions = [
                    int(action)
                    for action in legal_actions
                    if action_to_algebraic(int(action)) not in generated_legal_moves
                ]
                if not anchor_actions:
                    anchor_actions = [int(action) for action in legal_actions]
                random.shuffle(anchor_actions)
                for anchor_action in anchor_actions[:anchor_budget]:
                    anchor_response = self._build_legal_anchor_response(
                        anchor_action,
                        enemy_move_desc=enemy_move_desc,
                    )
                    anchor_eval = evaluate_candidate_response(
                        response=anchor_response,
                        board_before=board_state,
                        env=env,
                        query_ids=train_query_ids,
                        tokenizer=self.tokenizer,
                        enemy_move_desc=enemy_move_desc,
                        pikafish_evaluator=self.pikafish_evaluator,
                        cp_scale=self.reward_cp_scale,
                        format_weight=effective_format_weight,
                        forced_action=None,
                        reward_format_mode=self.reward_format_mix_mode,
                        grounding_strict=self.use_legal_move_sampler
                        and self.reward_grounding_strict_when_sampler,
                        grounding_quality_min=self.grounding_quality_min,
                        format_gate_fail_scale=self.format_gate_fail_scale,
                        format_soft_fail_scale=self.format_soft_fail_scale,
                        combine_gate_with_r_best=self.combine_gate_with_r_best,
                        position_best_uci_engine=position_best_uci_engine,
                        position_vb_best=position_vb_best,
                        tactical_weight=self.tactical_weight,
                        sigma_good_cp=self.sigma_good_cp,
                        engine_only=self.reward_engine_only,
                    )
                    if anchor_eval.legal and anchor_eval.action is not None:
                        anchor_evals.append(anchor_eval)
                train_evals.extend(anchor_evals)

            # ----------------------------------------------------------------
            # Engine-best tracking: are we picking the strongest move of the
            # 32 group, and does the 32-group even contain Pikafish's overall
            # best move for this position? One cached Pikafish call per FEN.
            # ----------------------------------------------------------------
            engine_best_uci_overall: Optional[str] = None
            # Pikafish ``position fen`` needs a full UCI FEN (side to move, etc.).
            # Generators attach ``fen`` for logging; if it is placement-only, upgrade.
            fen_for_bestmove = fen or ""
            if board_state is not None and (
                not fen_for_bestmove.strip() or " " not in fen_for_bestmove.strip()
            ):
                fen_for_bestmove = board_to_uci_fen(board_state, side_to_move="w")
            if (
                self.pikafish_evaluator is not None
                and self.pikafish_evaluator.enabled
                and fen_for_bestmove
            ):
                try:
                    bm_uci, _bm_cp = self.pikafish_evaluator.bestmove_root_cached(
                        fen_for_bestmove
                    )
                    # Pikafish returns its UCI move in **bottom-origin** rank
                    # coordinates (e.g. ``i0h0`` = ally rook on internal row 9
                    # capturing piece on row 9). Our group/chosen move_strs
                    # live in **top-origin** internal algebraic (``i9h9``),
                    # so the bestmove must be converted before any string
                    # comparison or it always reads as "not in group".
                    bm_internal: Optional[str] = None
                    if isinstance(bm_uci, str):
                        bm_internal = engine_uci_to_algebraic(bm_uci.lower())
                        if bm_internal is None:
                            print(
                                f"[Ep {episode} Rd {round_idx}] [engine_best] "
                                f"could not convert Pikafish UCI {bm_uci!r} to "
                                "internal algebraic; engine-best comparison "
                                "will read as 'unknown' this turn.",
                                flush=True,
                            )
                    engine_best_uci_overall = bm_internal
                except Exception as exc:
                    print(
                        f"[Ep {episode} Rd {round_idx}] [engine_best] Pikafish "
                        f"bestmove_root_cached failed: {exc!r}",
                        flush=True,
                    )
                    engine_best_uci_overall = None

            group_move_strings = {
                ev.move_str.lower()
                for ev in deduped_legal_evals
                if ev.move_str is not None
            }
            engine_best_in_group = bool(
                engine_best_uci_overall is not None
                and engine_best_uci_overall in group_move_strings
            )
            # Engine-best within the 32 sampled group (always defined when we
            # have at least one legal eval): the move with the highest
            # ``engine_reward`` is the model's argmax-policy choice's natural
            # benchmark for "did the policy pick the strongest move it saw?".
            engine_argmax_eval_in_group: Optional[CandidateEval] = None
            if deduped_legal_evals:
                engine_argmax_eval_in_group = max(
                    deduped_legal_evals, key=lambda ev: float(ev.engine_reward)
                )

            used_random_fallback = False
            chosen_capture = 0.0
            chosen_response = ""
            chosen_eval: Optional[CandidateEval] = None

            choice_pool = deduped_legal_evals if deduped_legal_evals else anchor_evals
            if choice_pool:
                best_reward = max(ev.reward for ev in choice_pool)
                if self.play_best_candidate:
                    best_candidates = [
                        ev for ev in choice_pool if ev.reward == best_reward
                    ]
                    chosen = random.choice(best_candidates)
                else:
                    # The legal-move sampler already drew this group from the
                    # current policy; play from that sampled policy group
                    # instead of using Pikafish as an action-selection oracle.
                    chosen = (
                        choice_pool[0]
                        if self.use_legal_move_sampler
                        else random.choice(choice_pool)
                    )
                chosen_eval = chosen
                chosen_action = int(chosen.action)
                chosen_move = str(chosen.move_str)
                chosen_capture = float(chosen.capture_value)
                chosen_response = chosen.response
                used_random_fallback = not bool(deduped_legal_evals)
            else:
                chosen_action = int(np.random.choice(legal_actions))
                chosen_move = action_to_algebraic(chosen_action)
                if evals:
                    chosen_response = "<think>Unable to evaluate generated reasoning, using fallback legal move.</think>"
                else:
                    chosen_response = "<think>Generation failed due to CUDA memory pressure, using fallback legal move.</think>"
                used_random_fallback = True
                best_reward = 0.0

            query_batch = [ev.query_ids for ev in train_evals]
            response_batch = [ev.response_ids for ev in train_evals]
            reward_batch = [float(ev.reward) for ev in train_evals]
            engine_policy_scores = [
                float(ev.engine_reward)
                if ev.legal and ev.engine_eval_success
                else float("nan")
                for ev in train_evals
            ]
            chosen_train_index: Optional[int] = None
            if chosen_eval is not None:
                for idx_ev, ev in enumerate(train_evals):
                    if ev is chosen_eval:
                        chosen_train_index = idx_ev
                        break
            train_stats = self.grpo_trainer.train_group(
                query_batch,
                response_batch,
                reward_batch,
                engine_policy_scores=engine_policy_scores,
            )
            successful_cp_deltas = [
                float(ev.cp_delta) for ev in legal_evals if ev.cp_delta is not None
            ]
            mean_cp_delta_success = (
                float(np.mean(successful_cp_deltas)) if successful_cp_deltas else None
            )

            round_format_rate = format_count / max(1, len(evals))
            self._update_format_compliance_ema(round_format_rate)

            # ----------------------------------------------------------------
            # Engine-best agreement metrics for this ally turn.
            # All flags are 0/1 (so per-episode means become percentages).
            # ----------------------------------------------------------------
            chosen_move_str_lower = (
                chosen_eval.move_str.lower()
                if (chosen_eval is not None and chosen_eval.move_str is not None)
                else None
            )
            chosen_is_engine_argmax_in_group = 0.0
            if (
                engine_argmax_eval_in_group is not None
                and chosen_move_str_lower is not None
                and engine_argmax_eval_in_group.move_str is not None
                and chosen_move_str_lower
                == engine_argmax_eval_in_group.move_str.lower()
            ):
                chosen_is_engine_argmax_in_group = 1.0
            chosen_is_engine_best_overall = 0.0
            if (
                engine_best_uci_overall is not None
                and chosen_move_str_lower == engine_best_uci_overall
            ):
                chosen_is_engine_best_overall = 1.0
            # Rank of chosen by engine_reward inside the deduped group (1 =
            # best, len(group) = worst). ``nan`` if no chosen / no legal evals.
            chosen_engine_rank_in_group: float = float("nan")
            if chosen_eval is not None and deduped_legal_evals:
                sorted_group = sorted(
                    deduped_legal_evals,
                    key=lambda ev: float(ev.engine_reward),
                    reverse=True,
                )
                for rank_idx, ev in enumerate(sorted_group, start=1):
                    if ev is chosen_eval:
                        chosen_engine_rank_in_group = float(rank_idx)
                        break
            # cp_delta gap: chosen minus group-argmax-engine. Negative means
            # we chose a strictly worse move than the strongest one in the
            # group. ``None`` if either side missing cp data.
            chosen_minus_argmax_cp_delta: Optional[float] = None
            if (
                chosen_eval is not None
                and chosen_eval.cp_delta is not None
                and engine_argmax_eval_in_group is not None
                and engine_argmax_eval_in_group.cp_delta is not None
            ):
                chosen_minus_argmax_cp_delta = float(chosen_eval.cp_delta) - float(
                    engine_argmax_eval_in_group.cp_delta
                )

            candidate_metrics = {
                "game/legal_move_rate": legal_count / max(1, len(evals)),
                "game/parsed_move_rate": move_parsed_count / max(1, len(evals)),
                "game/format_compliance_rate": round_format_rate,
                "game/format_compliance_ema": float(self._format_compliance_ema),
                "game/effective_format_weight": float(effective_format_weight),
                "game/reasoning_rate": reasoning_count / max(1, len(evals)),
                "game/reasoning_quality_rate": float(
                    np.mean([ev.reasoning_quality for ev in evals])
                )
                if evals
                else 0.0,
                "game/mean_engine_reward": float(
                    np.mean([ev.engine_reward for ev in legal_evals])
                )
                if legal_evals
                else 0.0,
                "game/mean_format_reward": float(
                    np.mean([ev.format_reward for ev in legal_evals])
                )
                if legal_evals
                else 0.0,
                "game/engine_eval_success_rate": float(
                    np.mean(
                        [1.0 if ev.engine_eval_success else 0.0 for ev in legal_evals]
                    )
                )
                if legal_evals
                else 0.0,
                "game/mean_cp_delta_success": float(mean_cp_delta_success)
                if mean_cp_delta_success is not None
                else 0.0,
                "game/move_diversity": len(set(move_strings)) / max(1, len(evals)),
                "game/legal_move_diversity": distinct_legal_move_count
                / max(1, len(evals)),
                "game/distinct_legal_moves": float(distinct_legal_move_count),
                "game/deduped_train_group_size": float(len(train_evals)),
                "game/dedupe_dropped_count": float(
                    len(evals) + len(anchor_evals) - len(train_evals)
                ),
                "game/legal_anchor_count": float(len(anchor_evals)),
                "game/mean_best_candidate_reward": float(generated_best_reward),
                "game/using_pikafish_legality": 1.0 if using_pikafish_legality else 0.0,
                "game/engine_legal_action_count": float(len(legal_actions)),
                "game/engine_legal_move_count_raw": float(engine_legal_count),
                # Engine-best agreement.
                "game/engine_best_known": 1.0
                if engine_best_uci_overall is not None
                else 0.0,
                "game/engine_best_in_group": 1.0 if engine_best_in_group else 0.0,
                "game/chosen_is_engine_argmax_in_group": float(
                    chosen_is_engine_argmax_in_group
                ),
                "game/chosen_is_engine_best_overall": float(
                    chosen_is_engine_best_overall
                ),
                # Combined-reward indicators (paper §3.4 R_move components):
                # mean fraction of legal candidates this turn whose move
                # exactly matched Pikafish's bestmove (``r_best``) or was
                # within ``sigma_good_cp`` cp of it (``r_good``). These are
                # always 0 when ``combine_gate_with_r_best`` is False, which
                # is the default.
                "game/mean_r_best_in_group": float(
                    np.mean([ev.r_best for ev in legal_evals])
                )
                if legal_evals
                else 0.0,
                "game/mean_r_good_in_group": float(
                    np.mean([ev.r_good for ev in legal_evals])
                )
                if legal_evals
                else 0.0,
                "game/r_best_in_group_any": 1.0
                if legal_evals and any(ev.r_best > 0.0 for ev in legal_evals)
                else 0.0,
            }
            if not np.isnan(chosen_engine_rank_in_group):
                candidate_metrics["game/chosen_engine_rank_in_group"] = float(
                    chosen_engine_rank_in_group
                )
            if chosen_minus_argmax_cp_delta is not None:
                candidate_metrics["game/chosen_minus_argmax_cp_delta"] = float(
                    chosen_minus_argmax_cp_delta
                )
            # Per-candidate r_best/r_good on the *chosen* move (proxy for the
            # tactical bonus the optimizer actually saw this turn).
            if chosen_eval is not None:
                candidate_metrics["game/chosen_r_best"] = float(chosen_eval.r_best)
                candidate_metrics["game/chosen_r_good"] = float(chosen_eval.r_good)
            invalid_previews: List[str] = []
            for ev in evals:
                if ev.legal:
                    continue
                snippet = " ".join((ev.response or "").strip().split())
                invalid_previews.append(
                    f"  parsed={ev.move_str or 'None'} "
                    f"format={int(ev.has_format)} reasoning={int(ev.has_reasoning)} :: "
                    f"{snippet[:240]}"
                )
                if len(invalid_previews) >= 3:
                    break

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
                        f"best_generated_reward={generated_best_reward:.4f}, "
                        f"legal_anchors={len(anchor_evals)}"
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
                    *(
                        ["Invalid sample responses (first 3):", *invalid_previews]
                        if invalid_previews
                        else []
                    ),
                    (f"Chosen move: {chosen_move} = {describe_action(chosen_action)}"),
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
                    (
                        "Engine-best comparison: "
                        f"engine_best_overall={engine_best_uci_overall or 'unknown'} "
                        f"in_group={int(engine_best_in_group)} "
                        f"argmax_in_group={engine_argmax_eval_in_group.move_str if engine_argmax_eval_in_group is not None else 'none'} "
                        f"argmax_engine_reward={engine_argmax_eval_in_group.engine_reward:.4f} "
                        f"argmax_cp_delta={_fmt_optional_float(engine_argmax_eval_in_group.cp_delta)} "
                        f"chosen={chosen_move_str_lower or 'none'} "
                        f"chosen_is_argmax_in_group={int(chosen_is_engine_argmax_in_group)} "
                        f"chosen_is_engine_best_overall={int(chosen_is_engine_best_overall)} "
                        f"chosen_rank_in_group={chosen_engine_rank_in_group if not np.isnan(chosen_engine_rank_in_group) else 'nan'} "
                        f"chosen_minus_argmax_cp_delta={_fmt_optional_float(chosen_minus_argmax_cp_delta)}"
                    )
                    if engine_argmax_eval_in_group is not None
                    else (
                        "Engine-best comparison: engine_best_overall="
                        f"{engine_best_uci_overall or 'unknown'} (no in-group argmax; no legal evals)"
                    ),
                ]
            )

            return TurnResult(
                action=chosen_action,
                move_algebraic=chosen_move,
                used_random_fallback=used_random_fallback,
                chosen_capture_value=chosen_capture,
                best_candidate_reward=float(generated_best_reward),
                chosen_engine_reward=float(chosen_eval.engine_reward)
                if chosen_eval is not None
                else 0.0,
                chosen_format_reward=float(chosen_eval.format_reward)
                if chosen_eval is not None
                else 0.0,
                chosen_cp_before=None if chosen_eval is None else chosen_eval.cp_before,
                chosen_cp_after_raw=None
                if chosen_eval is None
                else chosen_eval.cp_after_raw,
                chosen_cp_delta=None if chosen_eval is None else chosen_eval.cp_delta,
                chosen_engine_eval_success=bool(chosen_eval.engine_eval_success)
                if chosen_eval is not None
                else False,
                candidate_metrics=candidate_metrics,
                train_stats=train_stats,
                chosen_response=chosen_response,
                generation_stats=dict(self._last_generation_stats),
                terminal_query_batch=query_batch,
                terminal_response_batch=response_batch,
                terminal_reward_batch=reward_batch,
                terminal_engine_policy_scores=engine_policy_scores,
                terminal_chosen_index=chosen_train_index,
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

    def train_terminal_outcome_update(
        self,
        turn_result: TurnResult,
        *,
        terminal_value: float,
        outcome: str,
    ) -> Dict[str, float]:
        """Post-``env.step`` GRPO on the move ally actually played.

        ``outcome='win'``: boost the chosen candidate (ally delivered +100).
        ``outcome='loss'``: penalize the chosen candidate (enemy +100 or ally
        was hopelessly lost on cp-saturation truncation).
        """
        if (
            turn_result.terminal_query_batch is None
            or turn_result.terminal_response_batch is None
            or turn_result.terminal_reward_batch is None
            or turn_result.terminal_chosen_index is None
        ):
            return {}
        chosen_idx = int(turn_result.terminal_chosen_index)
        rewards = [float(v) for v in turn_result.terminal_reward_batch]
        if chosen_idx < 0 or chosen_idx >= len(rewards):
            return {}
        if outcome == "win":
            rewards[chosen_idx] = max(float(rewards[chosen_idx]), float(terminal_value))
        elif outcome == "loss":
            rewards[chosen_idx] = min(
                float(rewards[chosen_idx]), -float(terminal_value)
            )
        else:
            return {}
        stats = self.grpo_trainer.train_group(
            turn_result.terminal_query_batch,
            turn_result.terminal_response_batch,
            rewards,
            engine_policy_scores=turn_result.terminal_engine_policy_scores,
        )
        if not stats:
            return {}
        prefix = "terminal_win" if outcome == "win" else "terminal_loss"
        stats[f"{prefix}/update_applied"] = float(stats.get("grpo/update_applied", 0.0))
        stats[f"{prefix}/reward"] = float(terminal_value)
        stats[f"{prefix}/chosen_index"] = float(chosen_idx)
        return stats

    def train_terminal_win_update(
        self,
        turn_result: TurnResult,
        terminal_reward: float = 10.0,
    ) -> Dict[str, float]:
        """Run one extra GRPO step when the played move immediately wins."""
        stats = self.train_terminal_outcome_update(
            turn_result,
            terminal_value=terminal_reward,
            outcome="win",
        )
        if stats:
            stats["terminal_win/update_applied"] = float(
                stats.get("terminal_win/update_applied", 0.0)
            )
            stats["terminal_win/reward"] = float(terminal_reward)
            stats["terminal_win/chosen_index"] = float(
                stats.get("terminal_win/chosen_index", 0.0)
            )
        return stats

    def train_terminal_loss_update(
        self,
        turn_result: TurnResult,
        terminal_penalty: float = 10.0,
    ) -> Dict[str, float]:
        """Run one extra GRPO step penalizing the ally move that led to defeat."""
        stats = self.train_terminal_outcome_update(
            turn_result,
            terminal_value=terminal_penalty,
            outcome="loss",
        )
        if stats:
            stats["terminal_loss/update_applied"] = float(
                stats.get("terminal_loss/update_applied", 0.0)
            )
            stats["terminal_loss/penalty"] = float(terminal_penalty)
            stats["terminal_loss/chosen_index"] = float(
                stats.get("terminal_loss/chosen_index", 0.0)
            )
        return stats


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
    # Lowered from 1e-5 after the self-play resume showed bursty update episodes
    # with episode-level ``grpo/mean_kl_move`` above 1.0. With the auxiliary
    # engine-policy alignment loss below, use a smoother base GRPO step.
    "grpo/lr": 5e-6,
    # Lowered from 0.05 to 0.01 (2026-05-15 followup): per-token k3 KL was
    # dominating the loss on uninformative batches and pulling the policy
    # back toward the KL reference faster than advantages could push it
    # forward. With ``min_batch_reward_std`` gating in place this is a safer
    # weight; KL is still strong enough to prevent runaway drift in
    # informative batches.
    "grpo/beta": 0.01,
    # PPO-style ratio clipping over ``grpo/ppo_epochs`` inner optimizer steps.
    # ``clip_eps_high > clip_eps_low`` (DAPO "Clip Higher") gives exploration
    # tokens a little more headroom to increase their probability while still
    # bounding the step on the downside.
    # Use 1 inner epoch to match the OOM-safe sequential fallback path (which
    # already performs a single optimizer step) and reduce batched peak memory.
    "grpo/ppo_epochs": 1,
    "grpo/clip_eps_low": 0.2,
    "grpo/clip_eps_high": 0.28,
    # Small entropy bonus on just the ``Move:`` region tokens (3-5 tokens).
    # Directly counteracts move-level mode collapse without affecting <think>
    # reasoning. Set to 0 to disable.
    "grpo/entropy_coef_move": 0.01,
    # If generation produces too few legal moves, add a small number of
    # synthetic legal "anchor" completions to the GRPO group. This gives
    # all-invalid turns a positive reference instead of an all-zero reward
    # group, which otherwise has no advantage signal and cannot recover.
    "grpo/legal_anchor_count": 0,
    # Score every legal move under the current policy, then sample a distinct
    # legal candidate group without replacement. This makes diversity operate
    # over legal actions directly instead of over free-form generated strings.
    "grpo/use_legal_move_sampler": True,
    "grpo/legal_move_sample_temperature": 1.0,
    # Small uniform mixture keeps exploration alive when the model's move
    # distribution is prematurely sharp or miscalibrated.
    "grpo/legal_move_sample_epsilon": 0.05,
    # How the env's action is chosen from the legal-move sampler group:
    # - "greedy"       : play the argmax-policy legal move (on-policy, matches
    #                    inference-time greedy decoding; recommended default).
    # - "first_sample" : play the first weighted draw (legacy behavior; equal
    #                    to a single temperature=1 policy sample).
    # In both cases GRPO trains on the full sampled group regardless.
    "grpo/legal_move_action_selection": "greedy",
    # False means the environment plays from the model-sampled candidate group
    # instead of picking the highest Pikafish reward as a best-of-N oracle.
    "game/play_best_candidate": False,
    # Disabled: batched engine-policy alignment rarely ran in practice (forward
    # OOM → sequential fallback skips it) and the combined backward OOM'd when
    # alignment did run. Re-enable only if batched GRPO becomes memory-stable.
    # Re-enabled 2026-05-29 with a separate alignment backward (after GRPO
    # micro-backwards) so peak memory is one forward pass, not a fused graph.
    "grpo/engine_policy_align_coef": 0.05,
    "grpo/engine_policy_align_temperature": 0.5,
    # Raised from 0.1 to 0.5 (2026-05-15 followup): with a 0.1 cap the global
    # grad norm was almost certainly being clipped on every step (grad norms
    # in 7B LoRA + GRPO regularly hit O(1)). That throttled the AdamW update
    # magnitude well below ``lr`` and was the primary reason ``KL_move``
    # stayed in float-rounding territory.
    "grpo/max_grad_norm": 0.5,
    "grpo/optim": "adamw_8bit",
    # Skip the optimizer step (but still roll the env forward) when the
    # within-group reward spread is too small to give a meaningful gradient.
    # When ``reward_std`` is near zero, GRPO's policy-gradient term vanishes
    # and the loss is dominated by ``beta * KL``, which actively pulls the
    # policy back toward the KL reference and *unlearns*. Set to 0 to
    # disable. Lowered from 0.3 so fewer turns are GRPO-skipped; alignment
    # still contributes on flat-reward turns when engine scores are available.
    "grpo/min_batch_reward_std": 0.15,
    "generate/max_new_tokens": 32,
    "generate/do_sample": True,
    # Higher sampling temperature + mild top_p widening discourage the 32
    # candidates from collapsing onto a single move, which was previously
    # destroying GRPO's group-relative advantage signal and wasting the
    # regeneration budget. Paired with ``grpo/dedupe_legal_by_move`` the
    # effective group keeps distinct-move diversity without blowing up the
    # group size.
    "generate/temperature": 1.0,
    "generate/top_p": 0.98,
    "generate/repetition_penalty": 1.1,
    # Regeneration passes (pass_idx > 0) use a higher temperature + top_p to
    # inject move diversity once the initial pass lands in a low-entropy
    # basin. Leave unset (None) to reuse the base generate config.
    "generate/regen_temperature": 1.2,
    "generate/regen_top_p": 0.98,
    "episodes": 500,
    # Cap ally+enemy plies per episode (``truncated_cap`` when reached without terminal).
    "max_rounds_per_episode": 100,
    "seed": 42069,
    # When resuming, set this to False so the per-episode CSV from the prior
    # run is preserved and new rows are appended after it. The reconstructed
    # cumulative state (season returns, win counts, etc.) is preloaded from
    # the CSV based on ``checkpoint/start_episode``.
    "metrics/clear_csv_on_start": False,
    "checkpoint/dir": "./checkpoints/xiangqi_grpo_v2",
    "checkpoint/every_n_episodes": 1,
    # Adapter to load at process start. For a fresh run after SFT this is
    # ``checkpoints/xiangqi_sft``; when resuming an RL run, point this to a
    # specific ``ep_N`` directory and bump ``checkpoint/start_episode`` to
    # ``N + 1`` so the loop continues from where it left off.
    "checkpoint/load_adapter_path": "checkpoints/xiangqi_sft",
    # Frozen LoRA slot for KL reference (never trained). Keeps RL from drifting
    # off the SFT prior; falls back to base-only KL if missing.
    "checkpoint/sft_ref_adapter_path": "checkpoints/xiangqi_sft",
    # KL reference policy: ``sft_ref`` (frozen SFT adapter) or ``base`` (adapters off).
    "grpo/kl_reference": "sft_ref",
    # 1-indexed episode number to start the training loop from. Must match
    # the loaded adapter (``ep_{start_episode - 1}``) for the run-level
    # counters and CSV continuity to make sense.
    "checkpoint/start_episode": 1,
    # Rank-0 JSON updated after each round boundary (``xiangqi_v2_run_heartbeat.json``).
    # Set to ``""`` to disable.
    "training/run_heartbeat_path": RUN_HEARTBEAT_DEFAULT,
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
    "reward/format_weight": 0.08,
    # ``mix``: legacy (1-mix)*engine + mix*format_reward. ``gate``: engine
    # dominates; missing template or weak grounding sharply downscales reward;
    # strong grounding adds only a small bonus (mix * reasoning * 2.0).
    # ``xiangqi_r1``: discrete R_move+R_analysis+R_format (paper §3.4); requires
    # a ``Situation:`` line (Balanced / Advantage_Red / Advantage_Black) for analysis reward.
    "reward/format_mix_mode": "gate",
    "reward/grounding_quality_min": 0.42,
    "reward/format_gate_fail_scale": 0.08,
    "reward/format_soft_fail_scale": 0.45,
    # Combined reward (paper §3.4 R_move on top of the dense ``gate`` reward).
    # When True and ``reward/format_mix_mode == "gate"``: each candidate's
    # reward gets ``reward/tactical_weight * (r_good + r_best)`` added, where
    # ``r_best`` matches Pikafish's deep-search bestmove exactly and ``r_good``
    # is within ``reward/sigma_good_cp`` cp of it. Pikafish ``bestmove`` and
    # the best-position cp are computed once per turn and shared across the
    # 32 candidates so the extra Pikafish cost is ~1 call/turn.
    # Disabled for the next controlled restart: the May-16 gate-only phase had
    # the cleanest engine-alignment trend, and the combined bonus adds extra
    # Pikafish calls/noise while we are isolating the evaluator reliability fix.
    "reward/combine_gate_with_r_best": False,
    "reward/tactical_weight": 1.5,
    "reward/sigma_good_cp": float(SIGMA_GOOD),
    "reward/grounding_strict_when_sampler": True,
    # The environment's +100 terminal win reward is only known after the
    # pre-step candidate GRPO call. Map it back onto the dense 1-10 reward scale
    # for a second, terminal-only update on the actually played winning move.
    "reward/terminal_win_grpo_reward": 10.0,
    # Post-step GRPO penalty on the ally's last played move when the game ends
    # with enemy +100 (checkmate or cp-saturation truncation). Mirrors the
    # terminal-win update so losses also produce a clear group-relative signal.
    "reward/terminal_loss_grpo_penalty": 10.0,
    "sampler/generate_grounded_reasoning": True,
    "sampler/grounding_max_new_tokens": 192,
    "sampler/grounding_temperature": 0.75,
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
    # micro=16 OOMs on batched PPO forward (5090 32GB); 4 is a safer default.
    "grpo/logprob_micro_batch": 4,
    # Move-only experiment: prompt/response are just ``Move: <uci>`` (no
    # ``<think>``). GRPO KL/PG apply only to Move: tokens.
    # Reward is pure Pikafish cp-shaped engine signal (no format/reasoning).
    "prompt/move_only": True,
    "grpo/train_move_tokens_only": True,
    "reward/engine_only": True,
    # One catastrophic blunder is worth thousands of cp, which dominates the
    # ``mean_chosen_cp_delta`` metric (and anything derived from it) across an
    # otherwise normal ~90-round game. Clip the raw per-turn cp_delta before it
    # feeds into aggregates so a single -10000 outlier can't swing the mean by
    # -100+. The reward itself is already bounded by tanh and is not touched.
    "metrics/cp_delta_clip_abs": 400.0,
    # Exponential moving average of ``game/mean_ally_cp_after_move_red`` (0 = off).
    "metrics/ally_cp_after_ema_alpha": 0.2,
    # When the engine's score stays at mate saturation (cp_before <=
    # ``game/cp_saturation_threshold`` for the ally) for
    # ``game/cp_saturation_consecutive`` ally turns in a row, the position is
    # effectively terminal and continuing the episode just pads the logs with
    # 0-delta rounds. Truncate early and award +100 to the winning side.
    # Winning saturated positions (cp_before >= threshold) are **not** truncated
    # so the ally can practice converting advantages to checkmate.
    # Set ``game/cp_saturation_consecutive`` to 0 to disable.
    "game/cp_saturation_threshold": 4000.0,
    "game/cp_saturation_consecutive": 3,
    # Self-play: ally trains on ``default`` adapter; enemy plays from frozen
    # ``enemy`` adapter (checkpoints/self_play_enemy). Enemy syncs to ally
    # after ``game/self_play_wins_to_sync`` consecutive ally wins.
    "game/self_play": True,
    "game/self_play_wins_to_sync": 3,
    # Legacy GreedyEnemy ε-random curriculum (only when ``game/self_play`` is
    # False). With probability ``epsilon`` the enemy picks a uniformly random
    # legal action instead of the highest-value capture.
    "enemy/epsilon_start": 0.5,
    "enemy/epsilon_end": 0.0,
    "enemy/epsilon_anneal_episodes": 25,
    "enemy/epsilon_anchor_episode": None,
}

if args.episodes is not None:
    hyperparams["episodes"] = int(args.episodes)
if args.resume_from is not None:
    hyperparams["checkpoint/load_adapter_path"] = str(args.resume_from)
if args.start_episode is not None:
    hyperparams["checkpoint/start_episode"] = int(args.start_episode)

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
    setup_wandb_episode_metric_axes()
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
    peft_model = unwrap_model(model)
    if (
        str(hyperparams.get("grpo/kl_reference", "sft_ref")).strip().lower()
        == "sft_ref"
    ):
        sft_ref_dir = (
            hyperparams.get("checkpoint/sft_ref_adapter_path")
            or "checkpoints/xiangqi_sft"
        )
        ensure_frozen_sft_ref_adapter(peft_model, str(sft_ref_dir), rank=rank)
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

if hyperparams.get("game/self_play", False):
    peft_model = unwrap_model(model)
    enemy_dir = "checkpoints/self_play_enemy"
    start_fresh = (
        bool(hyperparams.get("metrics/clear_csv_on_start", True))
        and int(hyperparams.get("checkpoint/start_episode", 1) or 1) <= 1
    )
    enemy_exists = os.path.isdir(enemy_dir) and os.path.isfile(
        os.path.join(enemy_dir, "adapter_config.json")
    )
    if enemy_exists and not start_fresh:
        if rank == 0:
            print(
                f"[self-play] Loading existing frozen enemy adapter from {enemy_dir}",
                flush=True,
            )
        dist_barrier()
        peft_model.load_adapter(enemy_dir, adapter_name="enemy")
    else:
        if rank == 0:
            if enemy_exists and start_fresh:
                print(
                    f"[self-play] Fresh run: resetting frozen enemy adapter at {enemy_dir}",
                    flush=True,
                )
            else:
                print(
                    "[self-play] No existing enemy adapter found. Creating 'enemy' from current 'default'.",
                    flush=True,
                )
            os.makedirs(enemy_dir, exist_ok=True)
            peft_model.save_pretrained(enemy_dir, selected_adapters=["default"])
        dist_barrier()
        peft_model.load_adapter(enemy_dir, adapter_name="enemy")

    # Ensure only 'default' is trainable, and 'enemy' has requires_grad = False
    for name, param in peft_model.named_parameters():
        if ".enemy." in name:
            param.requires_grad = False
        elif ".default." in name:
            param.requires_grad = True

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

mesh = None
mp_policy = None
fsdp_kwargs = {}
if args.mixed_precision:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
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
        raise RuntimeError(
            "Could not find transformer layers for fully_shard on unsloth model."
        )
    for layer in layer_list:
        fully_shard(layer, mesh=mesh, **fsdp_kwargs)
    fully_shard(model, mesh=mesh, **fsdp_kwargs)
elif dist.is_initialized() and not args.use_ddp and world_size == 1 and rank == 0:
    print("[setup] Skipping FSDP because WORLD_SIZE=1; using plain single-GPU model.")
elif args.use_ddp and dist.is_initialized():
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )


def count_trainable_params(model_obj):
    total = sum(p.numel() for p in model_obj.parameters())
    trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


trainable_params, total_params, pct_trainable = count_trainable_params(model)
if rank == 0:
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} ({pct_trainable:.2f}%)"
    )
    print(f"Model: {hyperparams['model_name']}")


def save_lora_checkpoint(
    model_obj,
    tokenizer_obj,
    checkpoint_path: str,
    episode: int,
    label: str = "",
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_train_step: Optional[int] = None,
) -> None:
    """Save LoRA adapter + (optionally) Adam optimizer moments.

    The default save path persists only the LoRA ``state_dict``. When an
    ``optimizer`` is passed, this also writes ``optimizer.pt`` alongside the
    adapter directory containing the Adam ``m``/``v`` moments and the
    ``global_train_step`` counter. On resume, ``--resume-from`` will look for
    this file and restore the optimizer state so the first few post-resume
    GRPO updates pick up where the previous run left off instead of starting
    with cold ``m=0`` / ``v=0``.
    """
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
        peft_config = getattr(unwrapped, "peft_config", {}) or {}
        if "default" in peft_config and hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(
                checkpoint_path,
                state_dict=full_sd,
                selected_adapters=["default"],
            )
        else:
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
        with open(
            os.path.join(checkpoint_path, "training_meta.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)
        print(f"[checkpoint] Saved LoRA checkpoint to {checkpoint_path!r}")

        if optimizer is not None:
            opt_path = os.path.join(checkpoint_path, "optimizer.pt")
            try:
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "global_train_step": int(global_train_step or 0),
                    },
                    opt_path,
                )
                print(
                    f"[checkpoint] Saved optimizer state to {opt_path!r} "
                    f"(global_train_step={int(global_train_step or 0)})"
                )
            except Exception as opt_exc:
                print(
                    f"[checkpoint] WARN: failed to save optimizer state to "
                    f"{opt_path!r}: {opt_exc!r}"
                )
    dist_barrier()


grpo_trainer = GRPOTrainerOnline(
    model=model,
    tokenizer=tokenizer,
    device_obj=device,
    lr=float(hyperparams["grpo/lr"]),
    beta=float(hyperparams["grpo/beta"]),
    max_grad_norm=float(hyperparams["grpo/max_grad_norm"]),
    kl_reference=str(hyperparams.get("grpo/kl_reference", "sft_ref") or "sft_ref"),
    ppo_epochs=int(hyperparams.get("grpo/ppo_epochs", 2)),
    clip_eps_low=float(hyperparams.get("grpo/clip_eps_low", 0.2)),
    clip_eps_high=float(hyperparams.get("grpo/clip_eps_high", 0.28)),
    entropy_coef_move=float(hyperparams.get("grpo/entropy_coef_move", 0.0)),
    mp_policy=mp_policy,
    optimizer_name=str(hyperparams["grpo/optim"]),
    logprob_micro_batch=int(hyperparams.get("grpo/logprob_micro_batch", 4)),
    min_batch_reward_std=float(
        hyperparams.get("grpo/min_batch_reward_std", 0.0) or 0.0
    ),
    engine_policy_align_coef=float(
        hyperparams.get("grpo/engine_policy_align_coef", 0.0) or 0.0
    ),
    engine_policy_align_temperature=float(
        hyperparams.get("grpo/engine_policy_align_temperature", 0.5) or 0.5
    ),
    train_move_tokens_only=bool(hyperparams.get("grpo/train_move_tokens_only", False)),
)

# ---------------------------------------------------------------------------
# Optimizer state resume: when ``--resume-from`` was used and the loaded
# adapter directory contains an ``optimizer.pt`` saved by a prior run, restore
# Adam ``m``/``v`` moments + the global training step counter so post-resume
# updates pick up the per-parameter step-size scaling instead of starting cold.
# ---------------------------------------------------------------------------
_resumed_global_train_step: int = 0
if adapter_dir:
    _opt_state_path = os.path.join(adapter_dir, "optimizer.pt")
    if os.path.isfile(_opt_state_path):
        try:
            _opt_blob = torch.load(_opt_state_path, map_location="cpu")
            _opt_sd = (
                _opt_blob.get("optimizer", _opt_blob)
                if isinstance(_opt_blob, dict)
                else _opt_blob
            )
            grpo_trainer.optimizer.load_state_dict(_opt_sd)
            if isinstance(_opt_blob, dict):
                _resumed_global_train_step = int(
                    _opt_blob.get("global_train_step", 0) or 0
                )
            if rank == 0:
                print(
                    f"[checkpoint] Loaded optimizer state from {_opt_state_path!r} "
                    f"(global_train_step={_resumed_global_train_step})"
                )
        except Exception as _opt_exc:
            if rank == 0:
                print(
                    f"[checkpoint] WARN: failed to load optimizer state from "
                    f"{_opt_state_path!r}: {_opt_exc!r}; Adam moments stay cold"
                )
            _resumed_global_train_step = 0
    else:
        if rank == 0:
            print(
                f"[checkpoint] No optimizer.pt at {_opt_state_path!r}; "
                "Adam moments will start cold this run"
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
    legal_anchor_count=int(hyperparams["grpo/legal_anchor_count"]),
    use_legal_move_sampler=bool(hyperparams["grpo/use_legal_move_sampler"]),
    legal_move_sample_temperature=float(
        hyperparams["grpo/legal_move_sample_temperature"]
    ),
    legal_move_sample_epsilon=float(hyperparams["grpo/legal_move_sample_epsilon"]),
    legal_move_action_selection=str(
        hyperparams.get("grpo/legal_move_action_selection", "greedy")
    ),
    play_best_candidate=bool(hyperparams["game/play_best_candidate"]),
    reward_format_mix_mode=str(hyperparams["reward/format_mix_mode"]),
    grounding_quality_min=float(hyperparams["reward/grounding_quality_min"]),
    format_gate_fail_scale=float(hyperparams["reward/format_gate_fail_scale"]),
    format_soft_fail_scale=float(hyperparams["reward/format_soft_fail_scale"]),
    reward_grounding_strict_when_sampler=bool(
        hyperparams["reward/grounding_strict_when_sampler"]
    ),
    generate_grounded_reasoning_for_sampler=bool(
        hyperparams["sampler/generate_grounded_reasoning"]
    ),
    sampler_grounding_max_new_tokens=int(
        hyperparams["sampler/grounding_max_new_tokens"]
    ),
    sampler_grounding_temperature=float(hyperparams["sampler/grounding_temperature"]),
    regen_generate_overrides=regen_generate_overrides,
    combine_gate_with_r_best=bool(
        hyperparams.get("reward/combine_gate_with_r_best", False)
    ),
    tactical_weight=float(hyperparams.get("reward/tactical_weight", 1.5)),
    sigma_good_cp=float(hyperparams.get("reward/sigma_good_cp", SIGMA_GOOD)),
    move_only=bool(hyperparams.get("prompt/move_only", False)),
    reward_engine_only=bool(hyperparams.get("reward/engine_only", False)),
)
enemy_agent = GreedyEnemyAgent()


env = None
if rank == 0:
    env = gym.make(hyperparams["env"])
    if hyperparams.get("metrics/clear_csv_on_start", True):
        reset_episode_metrics_csv(EPISODE_METRICS_CSV)
        with open(SYNC_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    else:
        # Resume path: leave the prior CSV in place but upgrade its header
        # to the current schema so the new engine-best aggregates can be
        # appended cleanly.
        ensure_episode_metrics_csv_schema(EPISODE_METRICS_CSV)

_SIGNAL_ALLY_TURN = 1
_SIGNAL_EPISODE_DONE = 2
_SIGNAL_ENEMY_TURN = 3

episodes = int(hyperparams["episodes"])
start_episode = max(1, int(hyperparams.get("checkpoint/start_episode", 1) or 1))
max_rounds = int(hyperparams["max_rounds_per_episode"])
ckpt_root = hyperparams["checkpoint/dir"]
ckpt_every = int(hyperparams["checkpoint/every_n_episodes"])
cp_delta_clip_abs = float(hyperparams["metrics/cp_delta_clip_abs"])
cp_saturation_threshold = float(hyperparams["game/cp_saturation_threshold"])
cp_saturation_consecutive = int(hyperparams["game/cp_saturation_consecutive"])
self_play_enabled = bool(hyperparams.get("game/self_play", False))
self_play_wins_to_sync = max(
    1, int(hyperparams.get("game/self_play_wins_to_sync", 3) or 3)
)

# Legacy GreedyEnemy ε-random curriculum (ignored when ``self_play_enabled``).
enemy_epsilon_start = float(hyperparams.get("enemy/epsilon_start", 0.0) or 0.0)
enemy_epsilon_end = float(hyperparams.get("enemy/epsilon_end", 0.0) or 0.0)
enemy_epsilon_anneal_episodes = int(
    hyperparams.get("enemy/epsilon_anneal_episodes", 0) or 0
)
_anchor_cfg = hyperparams.get("enemy/epsilon_anchor_episode")
enemy_epsilon_anchor_episode = (
    int(_anchor_cfg) if _anchor_cfg is not None else int(start_episode)
)
if rank == 0:
    if hyperparams.get("prompt/move_only", False):
        print(
            "[prompt] move_only=True: responses are ``Move: <uci>`` only; "
            f"GRPO train_move_tokens_only={bool(hyperparams.get('grpo/train_move_tokens_only', False))}; "
            f"reward engine_only={bool(hyperparams.get('reward/engine_only', False))}",
            flush=True,
        )
    if self_play_enabled:
        print(
            "[self-play] opponent=frozen 'enemy' LoRA adapter "
            f"(checkpoints/self_play_enemy); sync after "
            f"{self_play_wins_to_sync} consecutive ally wins",
            flush=True,
        )
        print(
            f"[self-play] cp_sat_trunc asymmetric (ally losing only): "
            f"threshold={cp_saturation_threshold:.0f} cp, "
            f"consecutive={cp_saturation_consecutive}; "
            f"+100 terminal reward to winner on truncation",
            flush=True,
        )
    else:
        print(
            f"[curriculum] GreedyEnemy ε: start={enemy_epsilon_start:.2f} -> "
            f"end={enemy_epsilon_end:.2f} linearly across "
            f"{enemy_epsilon_anneal_episodes} episodes "
            f"(anchor=ep{enemy_epsilon_anchor_episode}); "
            f"cp_sat_trunc threshold={cp_saturation_threshold:.0f} cp, "
            f"consecutive={cp_saturation_consecutive}",
            flush=True,
        )

ally_wins = 0
enemy_wins = 0
truncated_games = 0
cp_saturation_truncations = 0
lifetime_ally_turns = 0
lifetime_random_fallback = 0
consecutive_self_play_wins = 0
self_play_enemy_id = 0
if self_play_enabled:
    self_play_enemy_id = load_self_play_enemy_id(
        "checkpoints/self_play_enemy",
        start_episode=start_episode,
        csv_path=EPISODE_METRICS_CSV if rank == 0 else None,
    )
    if rank == 0:
        print(
            f"[self-play] active frozen enemy generation id={self_play_enemy_id}",
            flush=True,
        )
global_train_steps = int(_resumed_global_train_step)
ally_cp_after_ema: Optional[float] = None
ally_cp_after_ema_alpha = float(
    hyperparams.get("metrics/ally_cp_after_ema_alpha", 0.0) or 0.0
)
# Sum of per-episode env returns over completed episodes (for stdout scoreboard).
season_ally_return = 0.0
season_enemy_return = 0.0

# Resume: preload cumulative counters from the existing CSV so the scoreboard,
# win-rate and lifetime random-fallback rate carry over after a checkpoint
# reload. No-op when ``start_episode == 1`` or the CSV is missing.
if rank == 0 and start_episode > 1:
    _resume_state = preload_run_state_from_csv(EPISODE_METRICS_CSV, start_episode)
    season_ally_return = _resume_state.season_ally_return
    season_enemy_return = _resume_state.season_enemy_return
    ally_wins = _resume_state.ally_wins
    enemy_wins = _resume_state.enemy_wins
    truncated_games = _resume_state.truncated_games
    cp_saturation_truncations = _resume_state.cp_saturation_truncations
    lifetime_ally_turns = _resume_state.lifetime_ally_turns
    lifetime_random_fallback = _resume_state.lifetime_random_fallback
    consecutive_self_play_wins = _resume_state.consecutive_self_play_wins
    print(
        f"[resume] start_episode={start_episode} | preloaded "
        f"ally_wins={ally_wins} enemy_wins={enemy_wins} "
        f"truncated={truncated_games} cp_sat_trunc={cp_saturation_truncations} "
        f"season_ally={season_ally_return:.2f} season_enemy={season_enemy_return:.2f} "
        f"lifetime_ally_turns={lifetime_ally_turns} "
        f"lifetime_random_fallback={lifetime_random_fallback} "
        f"consecutive_self_play_wins={consecutive_self_play_wins}",
        flush=True,
    )

_raw_hb = hyperparams.get("training/run_heartbeat_path", RUN_HEARTBEAT_DEFAULT)
_hb_path_opt: Optional[str] = None
if _raw_hb is not None:
    _hb_strip = str(_raw_hb).strip()
    if _hb_strip != "":
        _hb_path_opt = _hb_strip
_RUN_HB.configure(path=_hb_path_opt, rank=rank)
_RUN_HB.install_hooks()
if rank == 0:
    _RUN_HB.touch(
        "before_episode_for_loop",
        episode=0,
        round_idx=0,
        ally_return=0.0,
        enemy_return=0.0,
        global_train_steps=global_train_steps,
        status="starting",
    )

try:
    for episode in range(start_episode, episodes + 1):
        should_sync = False
        if rank == 0:
            observation = env.reset()
            done = False
            round_idx = 1
            ally_reward_terminal = 0.0
            enemy_reward_terminal = 0.0
            ally_return = 0.0
            enemy_return = 0.0
            enemy_move_desc_for_prompt: Optional[str] = None
            flipped_ally_move_desc_for_enemy: Optional[str] = None
            ally_turns_episode = 0
            random_fallback_episode = 0
            legal_rate_series: List[float] = []
            parsed_rate_series: List[float] = []
            format_rate_series: List[float] = []
            reasoning_rate_series: List[float] = []
            capture_series: List[float] = []
            best_reward_series: List[float] = []
            diversity_series: List[float] = []
            legal_diversity_series: List[float] = []
            legal_anchor_count_series: List[float] = []
            engine_eval_success_series: List[float] = []
            chosen_engine_reward_series: List[float] = []
            chosen_format_reward_series: List[float] = []
            chosen_cp_delta_series: List[float] = []
            chosen_cp_delta_raw_series: List[float] = []
            chosen_ally_cp_after_red_series: List[float] = []
            engine_best_known_series: List[float] = []
            engine_best_in_group_series: List[float] = []
            chosen_is_engine_argmax_series: List[float] = []
            chosen_is_engine_best_overall_series: List[float] = []
            chosen_engine_rank_series: List[float] = []
            chosen_minus_argmax_cp_delta_series: List[float] = []
            r_best_in_group_series: List[float] = []
            r_good_in_group_series: List[float] = []
            chosen_r_best_series: List[float] = []
            chosen_r_good_series: List[float] = []
            enemy_pikafish_prune_series: List[float] = []
            cp_saturation_streak = 0
            cp_saturation_truncated_this_episode = False
            train_stats_series: List[Dict[str, float]] = []
            last_ally_turn_result: Optional[TurnResult] = None
            episode_train_mfu_flops = 0.0
            episode_train_hfu_flops = 0.0
            episode_gen_flops = 0.0
            episode_train_wall_sec = 0.0
            episode_gen_wall_sec = 0.0
            episode_wall_start = time.perf_counter()

            episode_enemy_epsilon = 0.0
            if self_play_enabled:
                print(
                    f"\n[Ep {episode}] Opponent: frozen self-play copy "
                    f"(enemy_id={self_play_enemy_id}; sync after "
                    f"{self_play_wins_to_sync} consecutive ally wins; "
                    f"current streak={consecutive_self_play_wins})",
                    flush=True,
                )
            else:
                episode_enemy_epsilon = _current_enemy_epsilon(
                    episode=episode,
                    anchor_episode=enemy_epsilon_anchor_episode,
                    eps_start=enemy_epsilon_start,
                    eps_end=enemy_epsilon_end,
                    anneal_episodes=enemy_epsilon_anneal_episodes,
                )
                print(
                    f"\n[Ep {episode}] Opponent: GreedyEnemy "
                    f"(ε={episode_enemy_epsilon:.3f} random; anchor=ep{enemy_epsilon_anchor_episode}, "
                    f"anneal={enemy_epsilon_anneal_episodes} eps)",
                    flush=True,
                )
            print(
                format_episode_open_scoreboard(
                    episode, season_ally_return, season_enemy_return
                ),
                flush=True,
            )
            _RUN_HB.touch(
                "episode_reset",
                episode=episode,
                round_idx=1,
                ally_return=0.0,
                enemy_return=0.0,
                global_train_steps=global_train_steps,
            )

        while True:
            if rank == 0:
                if done:
                    signal = _SIGNAL_EPISODE_DONE
                elif env.turn != ALLY:
                    if self_play_enabled:
                        signal = _SIGNAL_ENEMY_TURN
                    else:
                        while not done and env.turn != ALLY:
                            board_before_enemy = env.state.copy()
                            enemy_action, enemy_policy_tag = enemy_agent.move(
                                env, epsilon=episode_enemy_epsilon
                            )
                            observation, enemy_reward, done, _ = env.step(enemy_action)
                            enemy_return += float(enemy_reward)
                            enemy_reward_terminal = float(enemy_reward)

                            if (
                                enemy_reward_terminal == 100.0
                                and last_ally_turn_result is not None
                            ):
                                loss_stats = ally_agent.train_terminal_loss_update(
                                    last_ally_turn_result,
                                    terminal_penalty=float(
                                        hyperparams["reward/terminal_loss_grpo_penalty"]
                                    ),
                                )
                                last_ally_turn_result.terminal_train_stats = (
                                    loss_stats or None
                                )
                                print(
                                    "[terminal-loss] enemy_reward=100.0; "
                                    "penalized ally's last move via terminal GRPO "
                                    f"(applied={float(loss_stats.get('grpo/update_applied', 0.0)) if loss_stats else 0.0:.0f}, "
                                    f"penalty={float(hyperparams['reward/terminal_loss_grpo_penalty']):.2f})",
                                    flush=True,
                                )
                                if loss_stats:
                                    loss_payload = {
                                        key: float(value)
                                        for key, value in loss_stats.items()
                                        if isinstance(value, (int, float))
                                    }
                                    loss_payload.update(
                                        {
                                            "train/global_step": global_train_steps,
                                            "episode": episode,
                                            "round": round_idx,
                                            "terminal_loss/enemy_reward": enemy_reward_terminal,
                                        }
                                    )
                                    wandb.log(loss_payload)

                            enemy_move_desc_for_prompt = describe_action(enemy_action)
                            log_board_sync(
                                [
                                    f"[Ep {episode} Rd {round_idx}] Enemy move "
                                    f"[{enemy_policy_tag}]: {enemy_move_desc_for_prompt}",
                                    f"Enemy board_before FEN: {board_to_fen(board_before_enemy)}",
                                    "Enemy board_before graphic:",
                                    board_to_graphic(board_before_enemy),
                                    "Enemy board_after numpy:",
                                    np.array2string(env.state),
                                    format_round_scoreboard(
                                        episode,
                                        round_idx,
                                        ally_return,
                                        enemy_return,
                                        season_ally_return,
                                        season_enemy_return,
                                    ),
                                ]
                            )
                            _RUN_HB.touch(
                                "after_enemy_env_step",
                                episode=episode,
                                round_idx=round_idx,
                                ally_return=ally_return,
                                enemy_return=enemy_return,
                                global_train_steps=global_train_steps,
                            )

                            round_idx += 1
                            if round_idx >= max_rounds and not done:
                                done = True
                                truncated_games += 1
                                break
                        signal = _SIGNAL_EPISODE_DONE if done else _SIGNAL_ALLY_TURN
                else:
                    signal = _SIGNAL_ALLY_TURN
            else:
                signal = 0

            signal = broadcast_int(signal)
            if signal == _SIGNAL_EPISODE_DONE:
                break

            if signal == _SIGNAL_ENEMY_TURN:
                # Collective enemy turn generation under self-play
                if rank == 0:
                    board_before_enemy = env.state.copy()
                    flipped_board = -board_before_enemy[::-1, :]
                else:
                    flipped_board = None

                enemy_action = generate_self_play_enemy_move(
                    agent=ally_agent,
                    env=env,
                    flipped_enemy_move_desc=flipped_ally_move_desc_for_enemy
                    if rank == 0
                    else None,
                    episode=episode,
                    round_idx=round_idx,
                    legality_prune_series=enemy_pikafish_prune_series
                    if rank == 0
                    else None,
                )

                if rank == 0:
                    observation, enemy_reward, done, _ = env.step(enemy_action)
                    enemy_return += float(enemy_reward)
                    enemy_reward_terminal = float(enemy_reward)

                    if (
                        enemy_reward_terminal == 100.0
                        and last_ally_turn_result is not None
                    ):
                        loss_stats = ally_agent.train_terminal_loss_update(
                            last_ally_turn_result,
                            terminal_penalty=float(
                                hyperparams["reward/terminal_loss_grpo_penalty"]
                            ),
                        )
                        last_ally_turn_result.terminal_train_stats = loss_stats or None
                        print(
                            "[terminal-loss] enemy_reward=100.0; "
                            "penalized ally's last move via terminal GRPO "
                            f"(applied={float(loss_stats.get('grpo/update_applied', 0.0)) if loss_stats else 0.0:.0f}, "
                            f"penalty={float(hyperparams['reward/terminal_loss_grpo_penalty']):.2f})",
                            flush=True,
                        )
                        if loss_stats:
                            loss_payload = {
                                key: float(value)
                                for key, value in loss_stats.items()
                                if isinstance(value, (int, float))
                            }
                            loss_payload.update(
                                {
                                    "train/global_step": global_train_steps,
                                    "episode": episode,
                                    "round": round_idx,
                                    "terminal_loss/enemy_reward": enemy_reward_terminal,
                                }
                            )
                            wandb.log(loss_payload)

                    enemy_move_desc_for_prompt = describe_action(enemy_action)
                    log_board_sync(
                        [
                            f"[Ep {episode} Rd {round_idx}] Enemy move [self_play]: {enemy_move_desc_for_prompt}",
                            f"Enemy board_before FEN: {board_to_fen(board_before_enemy)}",
                            "Enemy board_before graphic:",
                            board_to_graphic(board_before_enemy),
                            "Enemy board_after numpy:",
                            np.array2string(env.state),
                            format_round_scoreboard(
                                episode,
                                round_idx,
                                ally_return,
                                enemy_return,
                                season_ally_return,
                                season_enemy_return,
                            ),
                        ]
                    )
                    _RUN_HB.touch(
                        "after_enemy_env_step",
                        episode=episode,
                        round_idx=round_idx,
                        ally_return=ally_return,
                        enemy_return=enemy_return,
                        global_train_steps=global_train_steps,
                    )

                    round_idx += 1
                    if round_idx >= max_rounds and not done:
                        done = True
                        truncated_games += 1
                continue

            if signal == _SIGNAL_ALLY_TURN:
                if rank == 0:
                    print(
                        format_round_scoreboard(
                            episode,
                            round_idx,
                            ally_return,
                            enemy_return,
                            season_ally_return,
                            season_enemy_return,
                        ),
                        flush=True,
                    )
                    _RUN_HB.touch(
                        "before_ally_grpo",
                        episode=episode,
                        round_idx=round_idx,
                        ally_return=ally_return,
                        enemy_return=enemy_return,
                        global_train_steps=global_train_steps,
                    )
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
                    if ally_reward_terminal == 100.0:
                        terminal_stats = ally_agent.train_terminal_win_update(
                            turn_result,
                            terminal_reward=float(
                                hyperparams["reward/terminal_win_grpo_reward"]
                            ),
                        )
                        turn_result.terminal_train_stats = terminal_stats or None
                        print(
                            "[terminal-win] ally_reward=100.0; "
                            "ran terminal GRPO update "
                            f"(applied={float(terminal_stats.get('grpo/update_applied', 0.0)) if terminal_stats else 0.0:.0f}, "
                            f"reward={float(hyperparams['reward/terminal_win_grpo_reward']):.2f})",
                            flush=True,
                        )

                    last_ally_turn_result = turn_result

                    # Track ally move description in flipped perspective for enemy
                    ally_move_desc = action_to_algebraic(turn_result.action)
                    flipped_ally_move_desc_for_enemy = flip_move(ally_move_desc)

                    print(
                        format_round_scoreboard(
                            episode,
                            round_idx,
                            ally_return,
                            enemy_return,
                            season_ally_return,
                            season_enemy_return,
                        ),
                        flush=True,
                    )
                    _RUN_HB.touch(
                        "after_ally_env_step",
                        episode=episode,
                        round_idx=round_idx,
                        ally_return=ally_return,
                        enemy_return=enemy_return,
                        global_train_steps=global_train_steps,
                    )

                legal_rate_series.append(
                    float(turn_result.candidate_metrics["game/legal_move_rate"])
                )
                parsed_rate_series.append(
                    float(turn_result.candidate_metrics["game/parsed_move_rate"])
                )
                format_rate_series.append(
                    float(turn_result.candidate_metrics["game/format_compliance_rate"])
                )
                reasoning_rate_series.append(
                    float(turn_result.candidate_metrics["game/reasoning_rate"])
                )
                diversity_series.append(
                    float(turn_result.candidate_metrics["game/move_diversity"])
                )
                legal_diversity_series.append(
                    float(turn_result.candidate_metrics["game/legal_move_diversity"])
                )
                legal_anchor_count_series.append(
                    float(turn_result.candidate_metrics["game/legal_anchor_count"])
                )
                best_reward_series.append(float(turn_result.best_candidate_reward))
                capture_series.append(float(turn_result.chosen_capture_value))
                engine_eval_success_series.append(
                    float(
                        turn_result.candidate_metrics["game/engine_eval_success_rate"]
                    )
                )
                chosen_engine_reward_series.append(
                    float(turn_result.chosen_engine_reward)
                )
                chosen_format_reward_series.append(
                    float(turn_result.chosen_format_reward)
                )
                if turn_result.chosen_cp_delta is not None:
                    raw_delta = float(turn_result.chosen_cp_delta)
                    chosen_cp_delta_raw_series.append(raw_delta)
                    if cp_delta_clip_abs > 0.0:
                        chosen_cp_delta_series.append(
                            float(
                                np.clip(
                                    raw_delta, -cp_delta_clip_abs, cp_delta_clip_abs
                                )
                            )
                        )
                    else:
                        chosen_cp_delta_series.append(raw_delta)
                if (
                    turn_result.chosen_engine_eval_success
                    and turn_result.chosen_cp_after_raw is not None
                ):
                    chosen_ally_cp_after_red_series.append(
                        -float(turn_result.chosen_cp_after_raw)
                    )

                _cm_turn = turn_result.candidate_metrics or {}
                if "game/engine_best_known" in _cm_turn:
                    engine_best_known_series.append(
                        float(_cm_turn["game/engine_best_known"])
                    )
                if "game/engine_best_in_group" in _cm_turn:
                    engine_best_in_group_series.append(
                        float(_cm_turn["game/engine_best_in_group"])
                    )
                if "game/chosen_is_engine_argmax_in_group" in _cm_turn:
                    chosen_is_engine_argmax_series.append(
                        float(_cm_turn["game/chosen_is_engine_argmax_in_group"])
                    )
                if "game/chosen_is_engine_best_overall" in _cm_turn:
                    chosen_is_engine_best_overall_series.append(
                        float(_cm_turn["game/chosen_is_engine_best_overall"])
                    )
                if "game/chosen_engine_rank_in_group" in _cm_turn:
                    chosen_engine_rank_series.append(
                        float(_cm_turn["game/chosen_engine_rank_in_group"])
                    )
                if "game/chosen_minus_argmax_cp_delta" in _cm_turn:
                    chosen_minus_argmax_cp_delta_series.append(
                        float(_cm_turn["game/chosen_minus_argmax_cp_delta"])
                    )
                if "game/mean_r_best_in_group" in _cm_turn:
                    r_best_in_group_series.append(
                        float(_cm_turn["game/mean_r_best_in_group"])
                    )
                if "game/mean_r_good_in_group" in _cm_turn:
                    r_good_in_group_series.append(
                        float(_cm_turn["game/mean_r_good_in_group"])
                    )
                if "game/chosen_r_best" in _cm_turn:
                    chosen_r_best_series.append(float(_cm_turn["game/chosen_r_best"]))
                if "game/chosen_r_good" in _cm_turn:
                    chosen_r_good_series.append(float(_cm_turn["game/chosen_r_good"]))

                if (
                    cp_saturation_consecutive > 0
                    and turn_result.chosen_cp_before is not None
                    and float(turn_result.chosen_cp_before) <= -cp_saturation_threshold
                ):
                    cp_saturation_streak += 1
                else:
                    cp_saturation_streak = 0

                if turn_result.train_stats:
                    train_stats_series.append(dict(turn_result.train_stats))
                if turn_result.terminal_train_stats:
                    train_stats_series.append(dict(turn_result.terminal_train_stats))

                gen_stats_turn = turn_result.generation_stats or {}
                if gen_stats_turn.get("num_sequences", 0.0) > 0.0:
                    episode_gen_flops += (
                        ally_agent.grpo_trainer.mfu_tracker.generation_flops(
                            num_sequences=int(gen_stats_turn.get("num_sequences", 0.0)),
                            prompt_len=int(gen_stats_turn.get("prompt_len", 0.0)),
                            generated_len=int(gen_stats_turn.get("generated_len", 0.0)),
                        )
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
                if turn_result.terminal_train_stats:
                    episode_train_mfu_flops += float(
                        turn_result.terminal_train_stats.get("mfu/mfu_flops_step", 0.0)
                    )
                    episode_train_hfu_flops += float(
                        turn_result.terminal_train_stats.get("mfu/hfu_flops_step", 0.0)
                    )
                    episode_train_wall_sec += float(
                        turn_result.terminal_train_stats.get("mfu/step_time_sec", 0.0)
                    )

                if turn_result.train_stats:
                    step_payload = dict(turn_result.train_stats)
                    step_payload.update(
                        {
                            "train/global_step": global_train_steps,
                            "episode": episode,
                            "round": round_idx,
                            "train/candidate_legal_rate": turn_result.candidate_metrics[
                                "game/legal_move_rate"
                            ],
                            "train/candidate_parsed_move_rate": turn_result.candidate_metrics[
                                "game/parsed_move_rate"
                            ],
                            "train/candidate_format_rate": turn_result.candidate_metrics[
                                "game/format_compliance_rate"
                            ],
                            "train/candidate_reasoning_rate": turn_result.candidate_metrics[
                                "game/reasoning_rate"
                            ],
                            "train/candidate_reasoning_quality_rate": turn_result.candidate_metrics[
                                "game/reasoning_quality_rate"
                            ],
                            "train/candidate_move_diversity": turn_result.candidate_metrics[
                                "game/move_diversity"
                            ],
                            "train/candidate_legal_move_diversity": turn_result.candidate_metrics[
                                "game/legal_move_diversity"
                            ],
                            "train/legal_anchor_count": turn_result.candidate_metrics[
                                "game/legal_anchor_count"
                            ],
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
                    if "legal_action_policy_entropy" in gen_stats_turn:
                        step_payload["train/legal_action_policy_entropy"] = float(
                            gen_stats_turn["legal_action_policy_entropy"]
                        )
                    if "legal_action_policy_top_prob" in gen_stats_turn:
                        step_payload["train/legal_action_policy_top_prob"] = float(
                            gen_stats_turn["legal_action_policy_top_prob"]
                        )
                    if "sampler/grounding_wall_sec" in gen_stats_turn:
                        step_payload["train/sampler_grounding_wall_sec"] = float(
                            gen_stats_turn["sampler/grounding_wall_sec"]
                        )
                    if turn_result.chosen_cp_before is not None:
                        step_payload["train/chosen_cp_before"] = float(
                            turn_result.chosen_cp_before
                        )
                    if turn_result.chosen_cp_after_raw is not None:
                        step_payload["train/chosen_cp_after_raw"] = float(
                            turn_result.chosen_cp_after_raw
                        )
                    if turn_result.chosen_cp_delta is not None:
                        step_payload["train/chosen_cp_delta"] = float(
                            turn_result.chosen_cp_delta
                        )
                    if self_play_enabled:
                        step_payload["game/self_play_enemy_id"] = float(
                            self_play_enemy_id
                        )
                        step_payload["enemy/self_play_enemy_id"] = float(
                            self_play_enemy_id
                        )
                    wandb.log(step_payload)
                if turn_result.terminal_train_stats:
                    terminal_payload: Dict[str, float] = {}
                    for key, value in turn_result.terminal_train_stats.items():
                        if key.startswith(("terminal_win/", "terminal_loss/", "grpo/")):
                            terminal_payload[key] = float(value)
                        elif key.startswith("terminal_"):
                            terminal_payload[key] = float(value)
                        else:
                            terminal_payload[f"terminal_win/{key}"] = float(value)
                    terminal_payload.update(
                        {
                            "train/global_step": global_train_steps,
                            "episode": episode,
                            "round": round_idx,
                        }
                    )
                    if ally_reward_terminal == 100.0:
                        terminal_payload["terminal_win/ally_reward"] = (
                            ally_reward_terminal
                        )
                    wandb.log(terminal_payload)

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

                    # Asymmetric truncation: we only truncate when ally is losing.
                    # Award 100.0 winning points to the enemy!
                    enemy_reward_terminal = 100.0
                    enemy_return += 100.0

                    print(
                        f"[Ep {episode} Rd {round_idx - 1}] cp-saturation truncation "
                        f"(cp_before <= -{cp_saturation_threshold:.0f} for "
                        f"{cp_saturation_streak} consecutive ally turns; "
                        f"last cp_before={_fmt_optional_float(last_cp)})"
                    )
                    loss_stats = ally_agent.train_terminal_loss_update(
                        turn_result,
                        terminal_penalty=float(
                            hyperparams["reward/terminal_loss_grpo_penalty"]
                        ),
                    )
                    turn_result.terminal_train_stats = loss_stats or None
                    print(
                        "[terminal-loss] cp-saturation trunc (enemy +100); "
                        "penalized ally's last move via terminal GRPO "
                        f"(applied={float(loss_stats.get('grpo/update_applied', 0.0)) if loss_stats else 0.0:.0f}, "
                        f"penalty={float(hyperparams['reward/terminal_loss_grpo_penalty']):.2f})",
                        flush=True,
                    )

        if rank == 0:
            lifetime_ally_turns += ally_turns_episode
            lifetime_random_fallback += random_fallback_episode

            if enemy_reward_terminal == 100:
                enemy_wins += 1
            elif ally_reward_terminal == 100:
                ally_wins += 1

            legal_move_rate = (
                float(np.mean(legal_rate_series)) if legal_rate_series else 0.0
            )
            parsed_move_rate = (
                float(np.mean(parsed_rate_series)) if parsed_rate_series else 0.0
            )
            format_rate = (
                float(np.mean(format_rate_series)) if format_rate_series else 0.0
            )
            reasoning_rate = (
                float(np.mean(reasoning_rate_series)) if reasoning_rate_series else 0.0
            )
            move_diversity = (
                float(np.mean(diversity_series)) if diversity_series else 0.0
            )
            legal_move_diversity = (
                float(np.mean(legal_diversity_series))
                if legal_diversity_series
                else 0.0
            )
            mean_legal_anchor_count = (
                float(np.mean(legal_anchor_count_series))
                if legal_anchor_count_series
                else 0.0
            )
            mean_capture = float(np.mean(capture_series)) if capture_series else 0.0
            mean_best_reward = (
                float(np.mean(best_reward_series)) if best_reward_series else 0.0
            )
            mean_engine_eval_success_rate = (
                float(np.mean(engine_eval_success_series))
                if engine_eval_success_series
                else 0.0
            )
            mean_chosen_engine_reward = (
                float(np.mean(chosen_engine_reward_series))
                if chosen_engine_reward_series
                else 0.0
            )
            mean_chosen_format_reward = (
                float(np.mean(chosen_format_reward_series))
                if chosen_format_reward_series
                else 0.0
            )
            mean_chosen_cp_delta = (
                float(np.mean(chosen_cp_delta_series))
                if chosen_cp_delta_series
                else None
            )
            mean_chosen_cp_delta_raw = (
                float(np.mean(chosen_cp_delta_raw_series))
                if chosen_cp_delta_raw_series
                else None
            )
            mean_ally_cp_after_red = (
                float(np.mean(chosen_ally_cp_after_red_series))
                if chosen_ally_cp_after_red_series
                else None
            )
            median_ally_cp_after_red = (
                float(np.median(chosen_ally_cp_after_red_series))
                if chosen_ally_cp_after_red_series
                else None
            )
            if ally_cp_after_ema_alpha > 0.0 and mean_ally_cp_after_red is not None:
                if ally_cp_after_ema is None:
                    ally_cp_after_ema = mean_ally_cp_after_red
                else:
                    ally_cp_after_ema = float(
                        ally_cp_after_ema_alpha * mean_ally_cp_after_red
                        + (1.0 - ally_cp_after_ema_alpha) * ally_cp_after_ema
                    )
            random_rate_episode = (
                100.0 * random_fallback_episode / ally_turns_episode
                if ally_turns_episode
                else 0.0
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

            train_stats_episode: Dict[str, float] = {}
            if train_stats_series:
                stat_keys = sorted(
                    {key for stats in train_stats_series for key in stats.keys()}
                )
                for key in stat_keys:
                    vals: List[float] = []
                    for stats in train_stats_series:
                        raw_val = stats.get(key)
                        if isinstance(raw_val, (int, float)):
                            val = float(raw_val)
                            if math.isfinite(val):
                                vals.append(val)
                    if vals:
                        train_stats_episode[key] = float(np.mean(vals))
                train_stats_episode["grpo/update_rate"] = float(
                    100.0
                    * np.mean(
                        [
                            float(stats.get("grpo/update_applied", 0.0))
                            for stats in train_stats_series
                        ]
                    )
                )
                train_stats_episode["grpo/skip_low_reward_std_rate"] = float(
                    100.0
                    * np.mean(
                        [
                            float(stats.get("grpo/skipped_low_reward_std", 0.0))
                            for stats in train_stats_series
                        ]
                    )
                )

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
                "game/parsed_move_rate": parsed_move_rate,
                "game/format_compliance_rate": format_rate,
                "game/reasoning_rate": reasoning_rate,
                "game/mean_capture_value": mean_capture,
                "game/mean_best_candidate_reward": mean_best_reward,
                "game/move_diversity": move_diversity,
                "game/legal_move_diversity": legal_move_diversity,
                "game/mean_legal_anchor_count": mean_legal_anchor_count,
                "game/engine_eval_success_rate": mean_engine_eval_success_rate,
                "game/mean_chosen_engine_reward": mean_chosen_engine_reward,
                "game/mean_chosen_format_reward": mean_chosen_format_reward,
            }
            if mean_chosen_cp_delta is not None:
                episode_stats["game/mean_chosen_cp_delta"] = mean_chosen_cp_delta
                episode_stats["game/mean_chosen_cp_delta_clipped"] = (
                    mean_chosen_cp_delta
                )
            if mean_chosen_cp_delta_raw is not None:
                episode_stats["game/mean_chosen_cp_delta_raw"] = (
                    mean_chosen_cp_delta_raw
                )
            if mean_ally_cp_after_red is not None:
                episode_stats["game/mean_ally_cp_after_move_red"] = (
                    mean_ally_cp_after_red
                )
            if median_ally_cp_after_red is not None:
                episode_stats["game/median_ally_cp_after_move_red"] = (
                    median_ally_cp_after_red
                )
            if ally_cp_after_ema is not None:
                episode_stats["game/ally_cp_after_move_red_ema"] = ally_cp_after_ema
            episode_stats["game/cp_delta_clip_abs"] = cp_delta_clip_abs
            episode_stats["game/cp_saturation_truncated"] = (
                1.0 if cp_saturation_truncated_this_episode else 0.0
            )
            episode_stats["game/cp_saturation_truncation_rate"] = (
                100.0 * cp_saturation_truncations / episode
            )
            if self_play_enabled:
                episode_stats["game/self_play_enabled"] = 1.0
                episode_stats["game/self_play_wins_to_sync"] = float(
                    self_play_wins_to_sync
                )
                episode_stats["game/self_play_enemy_id"] = float(self_play_enemy_id)
                episode_stats["enemy/self_play_enemy_id"] = float(self_play_enemy_id)
                if enemy_pikafish_prune_series:
                    episode_stats["game/enemy_pikafish_prune_rate"] = float(
                        100.0 * np.mean(enemy_pikafish_prune_series)
                    )
            else:
                episode_stats["game/enemy_epsilon_current"] = float(
                    episode_enemy_epsilon
                )

            # Engine-best agreement (episode aggregates). Percent rates use *100.
            if engine_best_known_series:
                episode_stats["game/engine_best_known_rate"] = float(
                    100.0 * np.mean(engine_best_known_series)
                )
            if engine_best_in_group_series:
                episode_stats["game/engine_best_in_group_rate"] = float(
                    100.0 * np.mean(engine_best_in_group_series)
                )
            if chosen_is_engine_argmax_series:
                episode_stats["game/chosen_is_engine_argmax_in_group_rate"] = float(
                    100.0 * np.mean(chosen_is_engine_argmax_series)
                )
            if chosen_is_engine_best_overall_series:
                episode_stats["game/chosen_is_engine_best_overall_rate"] = float(
                    100.0 * np.mean(chosen_is_engine_best_overall_series)
                )
            if chosen_engine_rank_series:
                episode_stats["game/mean_chosen_engine_rank_in_group"] = float(
                    np.mean(chosen_engine_rank_series)
                )
                episode_stats["game/median_chosen_engine_rank_in_group"] = float(
                    np.median(chosen_engine_rank_series)
                )
            if chosen_minus_argmax_cp_delta_series:
                episode_stats["game/mean_chosen_minus_argmax_cp_delta"] = float(
                    np.mean(chosen_minus_argmax_cp_delta_series)
                )
            # Combined-reward aggregates (active only when
            # ``reward/combine_gate_with_r_best`` is True; otherwise stay at 0).
            if r_best_in_group_series:
                episode_stats["game/mean_r_best_in_group_rate"] = float(
                    100.0 * np.mean(r_best_in_group_series)
                )
            if r_good_in_group_series:
                episode_stats["game/mean_r_good_in_group_rate"] = float(
                    100.0 * np.mean(r_good_in_group_series)
                )
            if chosen_r_best_series:
                episode_stats["game/chosen_r_best_rate"] = float(
                    100.0 * np.mean(chosen_r_best_series)
                )
            if chosen_r_good_series:
                episode_stats["game/chosen_r_good_rate"] = float(
                    100.0 * np.mean(chosen_r_good_series)
                )

            for key, value in train_stats_episode.items():
                episode_stats[f"episode/{key}"] = float(value)

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
            episode_stats["mfu/episode_train_flops_mfu"] = float(
                episode_train_mfu_flops
            )
            episode_stats["mfu/episode_train_flops_hfu"] = float(
                episode_train_hfu_flops
            )
            episode_stats["mfu/episode_gen_flops"] = float(episode_gen_flops)
            if episode_wall_sec > 0:
                episode_stats["mfu/episode_gen_time_fraction"] = float(
                    episode_gen_wall_sec / episode_wall_sec
                )
                episode_stats["mfu/episode_train_time_fraction"] = float(
                    episode_train_wall_sec / episode_wall_sec
                )

            should_sync = False
            if self_play_enabled:
                if outcome == "ally_win":
                    consecutive_self_play_wins += 1
                else:
                    consecutive_self_play_wins = 0
                print(
                    f"[self-play] consecutive_self_play_wins={consecutive_self_play_wins}",
                    flush=True,
                )
                if consecutive_self_play_wins >= self_play_wins_to_sync:
                    should_sync = True
                    consecutive_self_play_wins = 0
                    print(
                        f"[self-play] Ally beat frozen enemy "
                        f"{self_play_wins_to_sync} times in a row; "
                        "syncing weights to enemy adapter.",
                        flush=True,
                    )
            episode_stats["game/consecutive_self_play_wins"] = int(
                consecutive_self_play_wins
            )
            episode_stats["enemy/sync_marker"] = 1.0 if should_sync else 0.0

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
                "game_parsed_move_rate": round(parsed_move_rate, 6),
                "game_format_compliance_rate": round(format_rate, 6),
                "game_reasoning_rate": round(reasoning_rate, 6),
                "game_mean_capture_value": round(mean_capture, 6),
                "game_mean_best_candidate_reward": round(mean_best_reward, 6),
                "game_mean_chosen_engine_reward": round(mean_chosen_engine_reward, 6),
                "game_move_diversity": round(move_diversity, 6),
                "game_legal_move_diversity": round(legal_move_diversity, 6),
                "game_mean_legal_anchor_count": round(mean_legal_anchor_count, 6),
                "game_cp_saturation_truncated": int(
                    cp_saturation_truncated_this_episode
                ),
                "game_mean_chosen_cp_delta_raw": _fmt_metric(mean_chosen_cp_delta_raw),
                "game_mean_chosen_cp_delta_clipped": _fmt_metric(mean_chosen_cp_delta),
                "game_mean_ally_cp_after_move_red": _fmt_metric(mean_ally_cp_after_red),
                "game_median_ally_cp_after_move_red": _fmt_metric(
                    median_ally_cp_after_red
                ),
                "game_ally_cp_after_move_red_ema": _fmt_metric(ally_cp_after_ema),
                "grpo_loss": _fmt_metric(train_stats_episode.get("grpo/loss")),
                "grpo_mean_kl": _fmt_metric(train_stats_episode.get("grpo/mean_kl")),
                "grpo_mean_kl_per_token": _fmt_metric(
                    train_stats_episode.get("grpo/mean_kl_per_token")
                ),
                "grpo_mean_kl_think": _fmt_metric(
                    train_stats_episode.get("grpo/mean_kl_think")
                ),
                "grpo_mean_kl_move": _fmt_metric(
                    train_stats_episode.get("grpo/mean_kl_move")
                ),
                "grpo_policy_entropy_move": _fmt_metric(
                    train_stats_episode.get("grpo/policy_entropy_move")
                ),
                "grpo_pg_clip_frac": _fmt_metric(
                    train_stats_episode.get("grpo/pg_clip_frac")
                ),
                "grpo_ratio_mean": _fmt_metric(
                    train_stats_episode.get("grpo/ratio_mean")
                ),
                "grpo_ppo_epochs_completed": _fmt_metric(
                    train_stats_episode.get("grpo/ppo_epochs_completed")
                ),
                "grpo_mean_reward": _fmt_metric(
                    train_stats_episode.get("grpo/mean_reward")
                ),
                "grpo_update_rate": _fmt_metric(
                    train_stats_episode.get("grpo/update_rate")
                ),
                "grpo_skip_low_reward_std_rate": _fmt_metric(
                    train_stats_episode.get("grpo/skip_low_reward_std_rate")
                ),
                "grpo_engine_align_loss": _fmt_metric(
                    train_stats_episode.get("grpo/engine_align_loss")
                ),
                "grpo_engine_align_kl": _fmt_metric(
                    train_stats_episode.get("grpo/engine_align_kl")
                ),
                "grpo_engine_align_entropy": _fmt_metric(
                    train_stats_episode.get("grpo/engine_align_entropy")
                ),
                "grpo_engine_align_target_entropy": _fmt_metric(
                    train_stats_episode.get("grpo/engine_align_target_entropy")
                ),
                "grpo_engine_align_valid_count": _fmt_metric(
                    train_stats_episode.get("grpo/engine_align_valid_count")
                ),
                "mfu": _fmt_metric(train_stats_episode.get("mfu/mfu")),
                "hfu": _fmt_metric(train_stats_episode.get("mfu/hfu")),
                "mfu_step_time_sec": _fmt_metric(
                    train_stats_episode.get("mfu/step_time_sec")
                ),
                "game_engine_best_known_rate": _fmt_metric(
                    episode_stats.get("game/engine_best_known_rate")
                ),
                "game_engine_best_in_group_rate": _fmt_metric(
                    episode_stats.get("game/engine_best_in_group_rate")
                ),
                "game_chosen_is_engine_argmax_in_group_rate": _fmt_metric(
                    episode_stats.get("game/chosen_is_engine_argmax_in_group_rate")
                ),
                "game_chosen_is_engine_best_overall_rate": _fmt_metric(
                    episode_stats.get("game/chosen_is_engine_best_overall_rate")
                ),
                "game_mean_chosen_engine_rank_in_group": _fmt_metric(
                    episode_stats.get("game/mean_chosen_engine_rank_in_group")
                ),
                "game_median_chosen_engine_rank_in_group": _fmt_metric(
                    episode_stats.get("game/median_chosen_engine_rank_in_group")
                ),
                "game_mean_chosen_minus_argmax_cp_delta": _fmt_metric(
                    episode_stats.get("game/mean_chosen_minus_argmax_cp_delta")
                ),
                "game_mean_r_best_in_group_rate": _fmt_metric(
                    episode_stats.get("game/mean_r_best_in_group_rate")
                ),
                "game_mean_r_good_in_group_rate": _fmt_metric(
                    episode_stats.get("game/mean_r_good_in_group_rate")
                ),
                "game_chosen_r_best_rate": _fmt_metric(
                    episode_stats.get("game/chosen_r_best_rate")
                ),
                "game_chosen_r_good_rate": _fmt_metric(
                    episode_stats.get("game/chosen_r_good_rate")
                ),
                "game_enemy_epsilon_current": (
                    ""
                    if self_play_enabled
                    else _fmt_metric(episode_stats.get("game/enemy_epsilon_current"))
                ),
                "game_enemy_pikafish_prune_rate": (
                    _fmt_metric(episode_stats.get("game/enemy_pikafish_prune_rate"))
                    if self_play_enabled
                    else ""
                ),
                "game_consecutive_self_play_wins": _fmt_metric(
                    consecutive_self_play_wins
                ),
                "game_self_play_enemy_id": (
                    _fmt_metric(self_play_enemy_id) if self_play_enabled else ""
                ),
                "game_global_train_step_end": _fmt_metric(global_train_steps),
            }
            append_episode_metrics_csv(EPISODE_METRICS_CSV, csv_row)

            all_eps_ally = season_ally_return + ally_return
            all_eps_enemy = season_enemy_return + enemy_return
            print(
                f"[Ep {episode}] ally_return={ally_return:.2f} enemy_return={enemy_return:.2f} "
                f"all_episodes_ally={all_eps_ally:.2f} all_episodes_enemy={all_eps_enemy:.2f} "
                f"ally_win_rate={episode_stats['game/ally_win_rate']:.1f}% "
                f"enemy_win_rate={episode_stats['game/enemy_win_rate']:.1f}% "
                f"legal_rate={legal_move_rate:.3f} parsed_rate={parsed_move_rate:.3f} "
                f"format_rate={format_rate:.3f} "
                f"legal_diversity={legal_move_diversity:.3f} "
                f"legal_anchors={mean_legal_anchor_count:.2f} "
                f"reasoning_rate={reasoning_rate:.3f} mean_capture={mean_capture:.3f} "
                f"engine_eval_success_rate={mean_engine_eval_success_rate:.3f} "
                f"mean_chosen_engine_reward={mean_chosen_engine_reward:.3f} "
                f"mean_chosen_format_reward={mean_chosen_format_reward:.3f} "
                f"mean_chosen_cp_delta={_fmt_optional_float(mean_chosen_cp_delta)} "
                f"mean_chosen_cp_delta_raw={_fmt_optional_float(mean_chosen_cp_delta_raw)} "
                f"mean_ally_cp_after_red={_fmt_optional_float(mean_ally_cp_after_red)} "
                f"ally_cp_ema={_fmt_optional_float(ally_cp_after_ema)} "
                f"outcome={outcome} "
                f"cp_sat_trunc_rate={episode_stats['game/cp_saturation_truncation_rate']:.1f}%"
                + (
                    f" self_play_streak={consecutive_self_play_wins}"
                    + (
                        f" enemy_pf_prune={episode_stats.get('game/enemy_pikafish_prune_rate', 0.0):.1f}%"
                        if enemy_pikafish_prune_series
                        else ""
                    )
                    if self_play_enabled
                    else f" enemy_epsilon={episode_enemy_epsilon:.3f}"
                )
            )
            season_ally_return += ally_return
            season_enemy_return += enemy_return
            _RUN_HB.touch(
                "episode_csv_appended",
                episode=episode,
                round_idx=round_idx,
                ally_return=ally_return,
                enemy_return=enemy_return,
                global_train_steps=global_train_steps,
            )

        should_sync = broadcast_bool(should_sync if rank == 0 else False, src=0)
        if should_sync:
            self_play_enemy_id += 1
            sync_enemy_adapter_with_default(
                model,
                new_enemy_id=self_play_enemy_id,
                synced_at_episode=episode,
                synced_at_global_step=global_train_steps,
            )
            if rank == 0:
                print(
                    f"[self-play] enemy_id advanced to {self_play_enemy_id} "
                    f"(next episode faces synced ally@ep{episode})",
                    flush=True,
                )

        if ckpt_every > 0 and episode % ckpt_every == 0:
            save_lora_checkpoint(
                model_obj=model,
                tokenizer_obj=tokenizer,
                checkpoint_path=os.path.join(ckpt_root, f"ep_{episode}"),
                episode=episode,
                label=f"every_{ckpt_every}",
                optimizer=grpo_trainer.optimizer,
                global_train_step=global_train_steps,
            )

    if rank == 0:
        save_lora_checkpoint(
            model_obj=model,
            tokenizer_obj=tokenizer,
            checkpoint_path=os.path.join(ckpt_root, "final"),
            episode=episodes,
            label="normal_completion",
            optimizer=grpo_trainer.optimizer,
            global_train_step=global_train_steps,
        )
        _RUN_HB.phase = "saved_final_checkpoint"
        _RUN_HB.episode = episodes
        _RUN_HB.flush(status="training_loop_complete")
        _RUN_HB.exit_normal = True
        print(
            f"[run_status] training finished cleanly; heartbeat={_RUN_HB.path!r}",
            flush=True,
        )
except Exception:
    try:
        crash_ep = episode
    except NameError:
        crash_ep = None
    try:
        crash_rd = round_idx
    except NameError:
        crash_rd = None
    if rank == 0:
        if _RUN_HB.path:
            _RUN_HB.phase = "python_exception"
            _RUN_HB.episode = crash_ep
            _RUN_HB.round_idx = crash_rd
            try:
                _RUN_HB.global_train_steps = int(global_train_steps)
            except NameError:
                pass
            _tb = traceback.format_exc(limit=32)
            _RUN_HB.flush(
                "python_exception",
                extra={"traceback_excerpt": _tb[:8000]},
            )
            _RUN_HB.detail_written = True
        print("\n" + "=" * 60)
        print("TRAINING CRASHED")
        if _RUN_HB.path:
            print(
                f"[run_status] see {_RUN_HB.path!r} for episode/round + traceback excerpt"
            )
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
    if crash_ep is not None and crash_ep >= 1:
        try:
            save_lora_checkpoint(
                model_obj=model,
                tokenizer_obj=tokenizer,
                checkpoint_path=os.path.join(ckpt_root, f"interrupted_ep{crash_ep}"),
                episode=crash_ep,
                label="interrupted",
                optimizer=grpo_trainer.optimizer,
                global_train_step=global_train_steps,
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
