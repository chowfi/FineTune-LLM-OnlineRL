import argparse
import csv
import json
import os
import random
import re
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gym
import matplotlib.pyplot as plt

try:
    from gym.wrappers import RecordVideo
except ImportError:
    try:
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        RecordVideo = None  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
import peft.tuners.tuners_utils as _peft_tuner_utils

# PEFT's BaseTuner.forward() calls self.model.forward() directly, which
# bypasses __call__() and therefore skips FSDP's pre-forward hooks.
# Patch it to use __call__() so FSDP can unshard parameters before the forward pass.
_peft_tuner_utils.BaseTuner.forward = lambda self, *args, **kwargs: self.model(*args, **kwargs)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, StateDictOptions,
)
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from gym_xiangqi.constants import ALLY, PIECE_ID_TO_NAME, PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move, move_to_action_space

parser = argparse.ArgumentParser()
parser.add_argument("--mixed-precision", action="store_true")
args = parser.parse_args()

# Read local_rank, rank, world_size from env 
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)

# Initialize communication between devices
dist.init_process_group(backend="nccl", init_method="env://")
mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

fsdp_kwargs = {}
if args.mixed_precision:
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }

# ─── Distributed helpers: broadcast arbitrary data from rank 0 ───

def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from src to all ranks (in-place)."""
    tensor = tensor.contiguous().cuda()
    dist.broadcast(tensor, src=src)
    return tensor

def broadcast_int(value: int, src: int = 0) -> int:
    t = torch.tensor([value], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()

def broadcast_bool(value: bool, src: int = 0) -> bool:
    return bool(broadcast_int(int(value), src=src))

def broadcast_input_ids(input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], src: int = 0):
    """Broadcast tokenized inputs from rank 0 to all ranks.
    Rank 0 passes the real tensors; other ranks pass None and receive them."""
    if rank == src:
        seq_len = input_ids.size(1)
    else:
        seq_len = 0
    seq_len = broadcast_int(seq_len, src=src)

    if rank == src:
        ids = input_ids.long().cuda()
        mask = attention_mask.long().cuda()
    else:
        ids = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")
        mask = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")

    dist.broadcast(ids, src=src)
    dist.broadcast(mask, src=src)
    return ids, mask


class GreedyEnemyAgent:
    """
    One-step greedy opponent: among legal enemy moves, pick one that captures
    the highest immediate material (ally pieces on the board are positive IDs).
    Ties and non-captures: uniform random among tied / all legal moves.
    """

    def move(self, env):
        actions = np.where(env.enemy_actions == 1)[0]
        if len(actions) == 0:
            raise RuntimeError("GreedyEnemyAgent: no legal enemy moves")
        board = env.state
        best_score = -1.0
        best_actions: List[int] = []
        for a in actions:
            _, _start, end = action_space_to_move(int(a))
            target = int(board[end[0]][end[1]])
            if target > 0:
                score = float(PIECE_POINTS[target])
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_actions = [int(a)]
            elif score == best_score:
                best_actions.append(int(a))
        return int(random.choice(best_actions))


# ─── Reward shaping ───

def compute_shaped_reward(
    board_before: np.ndarray,
    ally_action: int,
    env_step_reward: float,
    scale: float = 0.1,
) -> float:
    """Dense per-turn reward: env step reward + scaled value of captured piece."""
    _, _, end = action_space_to_move(ally_action)
    target = int(board_before[end[0]][end[1]])
    capture_value = PIECE_POINTS[abs(target)] if target < 0 else 0.0
    return env_step_reward + scale * capture_value


def distribute_terminal_bonus(
    terminal_reward: float,
    num_turns: int,
    gamma: float = 0.99,
) -> List[float]:
    """Discounted terminal bonus: later turns get more credit."""
    if num_turns <= 0:
        return []
    return [terminal_reward * (gamma ** (num_turns - 1 - i)) for i in range(num_turns)]


def describe_enemy_move(
    board_before: np.ndarray,
    board_after: np.ndarray,
    enemy_action: int,
) -> str:
    """One-line text description of what the enemy just did."""
    pid, start, end = action_space_to_move(enemy_action)
    enemy_name = PIECE_ID_TO_NAME[abs(pid)] if abs(pid) < len(PIECE_ID_TO_NAME) else "UNKNOWN"
    target = int(board_before[end[0]][end[1]])
    if target > 0:
        captured_name = PIECE_ID_TO_NAME[target] if target < len(PIECE_ID_TO_NAME) else "UNKNOWN"
        return (
            f"Enemy moved {enemy_name} from ({start[0]},{start[1]}) to "
            f"({end[0]},{end[1]}), capturing your {captured_name}."
        )
    return (
        f"Enemy moved {enemy_name} from ({start[0]},{start[1]}) to "
        f"({end[0]},{end[1]})."
    )


# ─── Evaluation metrics ───

def calculate_win_rate(wins, total_games):
    return (wins / total_games) * 100 if total_games > 0 else 0

def average_reward(rewards):
    return sum(rewards) / len(rewards) if rewards else 0

def average_episode_length(rounds_list):
    return sum(rounds_list) / len(rounds_list) if rounds_list else 0

def reward_variability(rewards):
    return np.std(rewards)


def _rolling_ma(series: List[float], window: int):
    if len(series) < window:
        return None, None
    kernel = np.ones(window) / window
    return range(window, len(series) + 1), np.convolve(series, kernel, mode="valid")


# ─── Evaluation visualizations ───

def plot_episode_lengths(episode_lengths, window_size=3, ally="LLM-Agent", enemy="Random"):
    plt.figure(figsize=(10, 5))
    xs, ma = _rolling_ma(episode_lengths, window_size)
    if xs is not None:
        plt.plot(xs, ma, color="red", label=f"{window_size}-ep MA")
    plt.title(f"Episode lengths ({ally} vs {enemy})")
    plt.xlabel("Episode")
    plt.ylabel("Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_episode_rewards(ally_rewards, enemy_rewards, window_size=3, ally="LLM-Agent", enemy="Random"):
    plt.figure(figsize=(10, 5))
    for rewards, color, label in (
        (ally_rewards, "cyan", ally),
        (enemy_rewards, "lime", enemy),
    ):
        xs, ma = _rolling_ma(rewards, window_size)
        if xs is not None:
            plt.plot(xs, ma, color=color, label=f"{label} MA")
    plt.title(f"Cumulative episode rewards ({ally} vs {enemy})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_grpo_diagnostics(
    ally_win_rate_cumulative,
    batch_reward_std_per_episode,
    trained_turn_fraction,
    window_size=5,
):
    episodes = np.arange(1, len(ally_win_rate_cumulative) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, ally_win_rate_cumulative, color="tab:blue", label="Cumulative ally win %")
    xs, ma = _rolling_ma(list(ally_win_rate_cumulative), window_size)
    if xs is not None:
        axes[0].plot(xs, ma, color="tab:orange", label=f"{window_size}-ep MA")
    axes[0].set_ylabel("Ally win (%)")
    axes[0].set_title("Cumulative ally win rate")
    axes[0].legend()
    axes[0].grid(True)

    std_masked = np.ma.masked_invalid(np.asarray(batch_reward_std_per_episode, dtype=float))
    axes[1].plot(episodes, std_masked, color="tab:green", alpha=0.85, label="Batch σ (episode mean)")
    axes[1].set_ylabel("Batch reward σ (pre-norm)")
    axes[1].set_title("GRPO reward spread per batch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(episodes, trained_turn_fraction, color="tab:purple", label="Trained / ally turns")
    xs, ma = _rolling_ma(list(trained_turn_fraction), window_size)
    if xs is not None:
        axes[2].plot(xs, ma, color="tab:brown", label=f"{window_size}-ep MA")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Non-random ally turns")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# ─── Log helpers ───

def training_log_token_truncate(file, num_episode, round, input_token_length=2048):
    with open(file, "a", encoding="utf-8") as f:
        f.write(
            f">>Episode {num_episode} round {round}, the input token length got to {input_token_length}\n"
        )


def training_log_llm_output(file, num_episode, round, llm_output):
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"Episode {num_episode} round {round}: response: {llm_output}\n")


EPISODE_METRICS_CSV = "chinese_chess_episode_metrics.csv"

_EPISODE_METRICS_FIELDNAMES = [
    "episode",
    "rounds",
    "ally_return",
    "enemy_return",
    "total_return",
    "outcome",
    "episode_hit_round_cap",
    "trained_turn_fraction",
    "random_move_pct_episode",
    "enemy_policy",
    "grpo_loss",
    "grpo_mean_advantage",
    "grpo_mean_kl",
    "grpo_mean_reward",
    "grpo_batch_reward_std_mean",
    "grpo_batch_reward_std_last",
    "grpo_train_steps",
    "mfu",
    "hfu",
    "mfu_achieved_tflops",
    "hfu_achieved_tflops",
    "mfu_step_time_sec",
]


def _fmt_metric(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return repr(val)
    return str(val)


def append_episode_metrics_csv(
    filepath: str,
    episode: int,
    rounds: int,
    ally_return: float,
    enemy_return: float,
    ally_reward_term: float,
    enemy_reward_term: float,
    episode_hit_round_cap: bool,
    trained_turn_fraction: float,
    random_move_pct_episode: float,
    train_stats: Optional[Dict] = None,
    enemy_policy: str = "random",
):
    if enemy_reward_term == 100:
        outcome = "enemy_win"
    elif ally_reward_term == 100:
        outcome = "ally_win"
    elif episode_hit_round_cap:
        outcome = "truncated_cap"
    else:
        outcome = "other"

    ts = train_stats or {}
    row = {
        "episode": episode,
        "rounds": rounds,
        "ally_return": round(ally_return, 6),
        "enemy_return": round(enemy_return, 6),
        "total_return": round(ally_return + enemy_return, 6),
        "outcome": outcome,
        "episode_hit_round_cap": episode_hit_round_cap,
        "trained_turn_fraction": round(trained_turn_fraction, 6),
        "random_move_pct_episode": round(random_move_pct_episode, 6),
        "enemy_policy": enemy_policy,
        "grpo_loss": _fmt_metric(ts.get("grpo/loss")),
        "grpo_mean_advantage": _fmt_metric(ts.get("grpo/mean_advantage")),
        "grpo_mean_kl": _fmt_metric(ts.get("grpo/mean_kl")),
        "grpo_mean_reward": _fmt_metric(ts.get("grpo/mean_reward")),
        "grpo_batch_reward_std_mean": _fmt_metric(ts.get("grpo/batch_reward_std_mean")),
        "grpo_batch_reward_std_last": _fmt_metric(ts.get("grpo/batch_reward_std")),
        "grpo_train_steps": _fmt_metric(ts.get("grpo/grpo_train_steps")),
        "mfu": _fmt_metric(ts.get("mfu/mfu")),
        "hfu": _fmt_metric(ts.get("mfu/hfu")),
        "mfu_achieved_tflops": _fmt_metric(ts.get("mfu/mfu_achieved_tflops")),
        "hfu_achieved_tflops": _fmt_metric(ts.get("mfu/hfu_achieved_tflops")),
        "mfu_step_time_sec": _fmt_metric(ts.get("mfu/step_time_sec")),
    }
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_METRICS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def reset_episode_metrics_csv(filepath: str) -> None:
    """Overwrite metrics CSV with header only (call once per training run)."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_EPISODE_METRICS_FIELDNAMES).writeheader()


# ─── MFU (Model FLOPs Utilization) profiling ───

_GPU_PEAK_TFLOPS: Dict[str, Dict[str, float]] = {
    "5090": {"bf16": 209.5, "fp32": 104.8, "fp16": 209.5},
    "5080": {"bf16": 112.0, "fp32": 56.0,  "fp16": 112.0},
    "4090": {"bf16": 330.0, "fp32": 82.6,  "fp16": 330.0},
    "4080": {"bf16": 200.0, "fp32": 48.7,  "fp16": 200.0},
    "3090": {"bf16": 142.0, "fp32": 35.6,  "fp16": 142.0},
    "3080": {"bf16": 119.0, "fp32": 29.8,  "fp16": 119.0},
    "h100": {"bf16": 989.0, "fp32": 67.0,  "fp16": 989.0},
    "h200": {"bf16": 989.0, "fp32": 67.0,  "fp16": 989.0},
    "a100": {"bf16": 312.0, "fp32": 19.5,  "fp16": 312.0},
    "a6000": {"bf16": 155.0, "fp32": 38.7, "fp16": 155.0},
    "a40":  {"bf16": 150.0, "fp32": 37.4,  "fp16": 150.0},
    "l40s": {"bf16": 362.0, "fp32": 91.6,  "fp16": 362.0},
    "l40":  {"bf16": 181.0, "fp32": 90.5,  "fp16": 181.0},
    "a5000": {"bf16": 65.6, "fp32": 27.8,  "fp16": 65.6},
    "v100": {"bf16": 125.0, "fp32": 15.7,  "fp16": 125.0},
}


def _get_gpu_peak_tflops(device_index: int = 0, dtype_str: str = "bf16") -> float:
    """Theoretical peak Tensor-Core TFLOPS for ``dtype_str`` (without sparsity).

    ``dtype_str`` should be ``"bf16"``, ``"fp16"``, or ``"fp32"`` to match
    the compute dtype used in forward/backward.
    Falls back to 50 TFLOPS if the GPU is unrecognised.
    """
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(device_index).lower()
    for key, peaks in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return peaks.get(dtype_str, peaks.get("bf16", 50.0))
    return 50.0


def _resolve_compute_dtype_str(mp_policy: Optional[MixedPrecisionPolicy]) -> str:
    """Determine the dtype string (``"bf16"``/``"fp16"``/``"fp32"``) that
    matrix multiplies actually run in, given the FSDP mixed-precision policy.

    If ``mp_policy`` is ``None`` or has no ``param_dtype``, the model was loaded
    in BF16 (see ``AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)``)
    so the compute dtype defaults to ``"bf16"``.
    """
    if mp_policy is None:
        return "bf16"
    dt = getattr(mp_policy, "param_dtype", None)
    if dt is None:
        return "bf16"
    _MAP = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return _MAP.get(dt, "bf16")


class MFUTracker:
    """Track MFU and HFU per GRPO training step.

    Per sample with sequence length T, P = all params, P_t = trainable (LoRA):

    MFU (theoretical minimum FLOPs — no recomputation):
        ref forward  (no grad, LoRA off) : 2 · P · T
        policy forward    (with grad)    : 2 · P · T
        backward: activation grads       : 2 · P · T
        backward: weight grads (LoRA)    : 2 · P_t · T
        ─────────────────────────────────────────
        MFU total ≈ 6·P·T   (since P_t << P)

    HFU (actual hardware FLOPs — includes gradient-checkpoint recomputation):
        Same as above, plus the recomputed forward pass during backward
        triggered by gradient_checkpointing_enable():
        checkpoint recomputation         : +2 · P · T
        ─────────────────────────────────────────
        HFU total ≈ 8·P·T

    MFU = MFU_flops / (elapsed · peak)
    HFU = HFU_flops / (elapsed · peak)
    """

    def __init__(self, model, device_index: int = 0,
                 mp_policy: Optional[MixedPrecisionPolicy] = None,
                 gradient_checkpointing: bool = True):
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.gradient_checkpointing = gradient_checkpointing
        dtype_str = _resolve_compute_dtype_str(mp_policy)
        self.dtype_str = dtype_str
        self.gpu_peak_tflops = _get_gpu_peak_tflops(device_index, dtype_str)
        self.gpu_peak_flops = self.gpu_peak_tflops * 1e12
        self._history: List[Dict[str, float]] = []

    def compute(self, total_tokens: int, elapsed_sec: float,
                num_fwd_per_sample: int = 2) -> Dict[str, float]:
        """Return MFU and HFU metrics for one train_step.

        Args:
            total_tokens: sum of (query + response) lengths across every
                          sample in the batch.
            elapsed_sec:  wall-clock seconds for the train step
                          (cuda-synchronised on both ends).
            num_fwd_per_sample: forward passes per sample (2 = ref + policy).
        """
        if elapsed_sec <= 0 or total_tokens <= 0:
            return {}

        P, P_t, T = self.total_params, self.trainable_params, total_tokens

        fwd_flops = num_fwd_per_sample * 2 * P * T
        bwd_act_grad_flops = 2 * P * T
        bwd_weight_grad_flops = 2 * P_t * T
        mfu_flops = fwd_flops + bwd_act_grad_flops + bwd_weight_grad_flops

        recompute_flops = (2 * P * T) if self.gradient_checkpointing else 0
        hfu_flops = mfu_flops + recompute_flops

        mfu_achieved = (mfu_flops / elapsed_sec) / 1e12
        hfu_achieved = (hfu_flops / elapsed_sec) / 1e12
        mfu = (mfu_flops / elapsed_sec) / self.gpu_peak_flops if self.gpu_peak_flops > 0 else 0.0
        hfu = (hfu_flops / elapsed_sec) / self.gpu_peak_flops if self.gpu_peak_flops > 0 else 0.0

        stats: Dict[str, float] = {
            "mfu/mfu": mfu,
            "mfu/hfu": hfu,
            "mfu/mfu_achieved_tflops": mfu_achieved,
            "mfu/hfu_achieved_tflops": hfu_achieved,
            "mfu/peak_tflops": self.gpu_peak_tflops,
            "mfu/dtype": self.dtype_str,
            "mfu/total_tokens_step": total_tokens,
            "mfu/step_time_sec": elapsed_sec,
        }
        self._history.append(stats)
        return stats

    def summary(self) -> Dict[str, float]:
        """Aggregate MFU and HFU across all recorded steps."""
        if not self._history:
            return {}
        mfus = [h["mfu/mfu"] for h in self._history]
        hfus = [h["mfu/hfu"] for h in self._history]
        m_tfs = [h["mfu/mfu_achieved_tflops"] for h in self._history]
        h_tfs = [h["mfu/hfu_achieved_tflops"] for h in self._history]
        return {
            "mfu/lifetime_mean_mfu": float(np.mean(mfus)),
            "mfu/lifetime_median_mfu": float(np.median(mfus)),
            "mfu/lifetime_mean_hfu": float(np.mean(hfus)),
            "mfu/lifetime_median_hfu": float(np.median(hfus)),
            "mfu/lifetime_mean_mfu_tflops": float(np.mean(m_tfs)),
            "mfu/lifetime_mean_hfu_tflops": float(np.mean(h_tfs)),
            "mfu/lifetime_steps_profiled": len(self._history),
        }


# ─── Lightweight GRPO trainer for online RL ───

class GRPOTrainerOnline:
    """
    Group Relative Policy Optimization for an online game-playing loop.

    Collects (query_ids, response_ids, reward) tuples across episodes, then
    does a single gradient step when the batch is full:

      1. Normalize rewards within the batch (group-relative baseline).
      2. Compute log-probs of each response under the *current* policy.
      3. (Optional) Compute log-probs under the *reference* policy by
         disabling LoRA adapters — no second model copy needed.
      4. Loss = -mean(advantage * response_log_prob) + beta * KL
    """

    def __init__(self, model, tokenizer, device, batch_size=8, lr=1e-5, beta=0.1, max_grad_norm=1.0,
                 mp_policy: Optional[MixedPrecisionPolicy] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        _grad_ckpt = getattr(model, "is_gradient_checkpointing", False)
        self.mfu_tracker = MFUTracker(
            model, device.index or 0, mp_policy=mp_policy,
            gradient_checkpointing=_grad_ckpt,
        )

        self.buffer = {"query_ids": [], "response_ids": [], "rewards": []}

    def add_to_buffer(self, query_ids, response_ids, reward):
        self.buffer["query_ids"].append(query_ids)
        self.buffer["response_ids"].append(response_ids)
        self.buffer["rewards"].append(reward)

    def buffer_size(self):
        return len(self.buffer["rewards"])

    def _compute_response_log_probs(self, query_ids, response_ids):
        """Forward pass on query+response; return sum of log-probs over response tokens."""
        input_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        logits = self.model(input_ids=input_ids).logits
        response_start = query_ids.size(0)
        response_logits = logits[0, response_start - 1 : -1, :]
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
        return token_log_probs.sum()

    def _broadcast_buffer_batch(self):
        """Rank 0 pops a batch from its buffer and broadcasts to all ranks.
        Returns (query_ids_batch, response_ids_batch, rewards_tensor) on every rank."""
        if rank == 0:
            query_ids_batch = self.buffer["query_ids"][: self.batch_size]
            response_ids_batch = self.buffer["response_ids"][: self.batch_size]
            rewards_batch = self.buffer["rewards"][: self.batch_size]
            self.buffer["query_ids"] = self.buffer["query_ids"][self.batch_size :]
            self.buffer["response_ids"] = self.buffer["response_ids"][self.batch_size :]
            self.buffer["rewards"] = self.buffer["rewards"][self.batch_size :]
            n = len(query_ids_batch)
        else:
            n = 0

        n = broadcast_int(n)

        synced_queries = []
        synced_responses = []
        for i in range(n):
            if rank == 0:
                q_len = query_ids_batch[i].numel()
                r_len = response_ids_batch[i].numel()
            else:
                q_len, r_len = 0, 0
            q_len = broadcast_int(q_len)
            r_len = broadcast_int(r_len)

            if rank == 0:
                q = query_ids_batch[i].long().cuda()
                r = response_ids_batch[i].long().cuda()
            else:
                q = torch.zeros(q_len, dtype=torch.long, device="cuda")
                r = torch.zeros(r_len, dtype=torch.long, device="cuda")
            dist.broadcast(q, src=0)
            dist.broadcast(r, src=0)
            synced_queries.append(q.cpu())
            synced_responses.append(r.cpu())

        if rank == 0:
            rewards_t = torch.tensor(rewards_batch, dtype=torch.float32).cuda()
        else:
            rewards_t = torch.zeros(n, dtype=torch.float32, device="cuda")
        dist.broadcast(rewards_t, src=0)
        rewards_t = rewards_t.cpu()

        return synced_queries, synced_responses, rewards_t

    def train_step(self):
        """GRPO update — all ranks must call this together so FSDP collectives stay in sync."""
        should_train = broadcast_bool(self.buffer_size() >= self.batch_size if rank == 0 else False)
        if not should_train:
            return {}

        query_ids_batch, response_ids_batch, rewards_t = self._broadcast_buffer_batch()

        reward_std_before_norm = float(rewards_t.std().item())
        if reward_std_before_norm > 1e-4:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std_before_norm + 1e-8)
        else:
            advantages = rewards_t - rewards_t.mean()

        total_tokens = sum(
            q.numel() + r.numel()
            for q, r in zip(query_ids_batch, response_ids_batch)
        )
        torch.cuda.synchronize()
        _mfu_t0 = time.perf_counter()

        self.model.train()
        self.optimizer.zero_grad()

        n = len(query_ids_batch)
        total_loss_val = 0.0
        total_kl_val = 0.0

        for i in range(n):
            q = query_ids_batch[i]
            r = response_ids_batch[i]
            adv = advantages[i].to(self.device)

            with torch.no_grad():
                self.model.disable_adapter_layers()
                ref_log_prob = self._compute_response_log_probs(q, r)
                self.model.enable_adapter_layers()
                torch.cuda.empty_cache()

            current_log_prob = self._compute_response_log_probs(q, r)

            kl = current_log_prob - ref_log_prob
            sample_loss = (-adv * current_log_prob + self.beta * kl) / n

            sample_loss.backward()

            total_loss_val += sample_loss.item()
            total_kl_val += kl.item()

            del current_log_prob, ref_log_prob, kl, sample_loss
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.max_grad_norm,
        )
        self.optimizer.step()

        torch.cuda.synchronize()
        _mfu_elapsed = time.perf_counter() - _mfu_t0
        mfu_stats = self.mfu_tracker.compute(
            total_tokens, _mfu_elapsed, num_fwd_per_sample=2,
        )

        torch.cuda.empty_cache()

        stats = {
            "grpo/loss": total_loss_val,
            "grpo/mean_advantage": advantages.mean().item(),
            "grpo/mean_kl": total_kl_val / n,
            "grpo/mean_reward": rewards_t.mean().item(),
            "grpo/batch_reward_std": reward_std_before_norm,
        }
        stats.update(mfu_stats)
        return stats


# ─── LLM Agent with GRPO ───

class Agent(ABC):
    def __init__(
        self, model, tokenizer, grpo_trainer, max_input_token, device, generate_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 16,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }

        self.model = model
        self.tokenizer = tokenizer
        self.grpo_trainer = grpo_trainer
        self.max_input_token = max_input_token
        self.device = device
        self.generate_config_dict = generate_config_dict

        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_llm_input = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_episode_rewards = []
        self.current_episode_shaped_rewards = []
        self.current_episode_turn_data = []
        self.last_enemy_move_desc: Optional[str] = None

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(
        self, observation: gym.core.ObsType, env: gym.Env = None, ally_turn_index: int = 0,
        enemy_move_desc: Optional[str] = None,
    ) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str, env: gym.Env) -> Tuple[bool, Any]:
        pass

    def llm(self, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Run model.generate() across all FSDP ranks.
        Rank 0 tokenizes *messages* and broadcasts input_ids/attention_mask.
        Other ranks pass messages=None and receive the broadcast."""
        if rank == 0:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            ids_bcast, mask_bcast = broadcast_input_ids(inputs.input_ids, inputs.attention_mask)
        else:
            ids_bcast, mask_bcast = broadcast_input_ids(None, None)

        context_len = mask_bcast.size(1)
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=ids_bcast,
                attention_mask=mask_bcast,
                **{
                    key.split("/")[-1]: value
                    for key, value in self.generate_config_dict.items()
                }
            )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        generate_ids = generate_ids[:, context_len:]
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs[0]

    def act(self, observation, episode, round, env, ally_turn_index: int = 0):
        """Rank 0 formats the prompt and extracts the action from the response.
        All ranks participate in model.generate() via self.llm().
        Non-rank-0 callers should pass observation=None, env=None."""

        if rank == 0:
            message = self.format_observation(
                observation, env, ally_turn_index=ally_turn_index,
                enemy_move_desc=self.last_enemy_move_desc,
            )
            self.last_enemy_move_desc = None
            self.current_episode_messages += [{"role": "user", "content": message}]
            self.current_llm_input += [{"role": "user", "content": message}]

            prompt = self.tokenizer.apply_chat_template(
                self.current_llm_input, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_token_length = len(inputs.input_ids[0])

            print(f"[Ep {episode} Rd {round}] LLM Prompt: {prompt}")

            if (input_token_length + 200) >= self.max_input_token:
                training_log_token_truncate(
                    "chinese_chess_token_truncate_log",
                    num_episode=episode,
                    round=round,
                    input_token_length=input_token_length,
                )
                while len(self.current_llm_input) > 2:
                    self.current_llm_input.pop(1)
                    if (
                        len(self.current_llm_input) > 1
                        and self.current_llm_input[1]["role"] == "assistant"
                    ):
                        self.current_llm_input.pop(1)
                    prompt = self.tokenizer.apply_chat_template(
                        self.current_llm_input, tokenize=False, add_generation_prompt=True
                    )
                    check = self.tokenizer(prompt, return_tensors="pt")
                    if len(check.input_ids[0]) + 200 < self.max_input_token:
                        break
                inputs = self.tokenizer(prompt, return_tensors="pt")

        # All ranks call generate together (rank 0 broadcasts input_ids inside llm())
        if rank == 0:
            response = self.llm(self.current_llm_input)
        else:
            response = self.llm(None)

        # Only rank 0 extracts the action and manages episode state
        if rank == 0:
            print(f"[Ep {episode} Rd {round}] LLM Response: {response}")

            try:
                is_random, action = self.extract_action(response, env)
            except Exception:
                return None, None, response

            if is_random:
                piece_id, start, end = action_space_to_move(action)
                corrected_response = f"Action: {piece_id}, ({start[0]}, {start[1]}), ({end[0]}, {end[1]})"
                assistant_msg = f"(system override: random move) {corrected_response}"
            else:
                corrected_response = response
                assistant_msg = corrected_response

            query_ids = inputs.input_ids[0].cpu()
            max_train_ctx = 4096
            if len(query_ids) > max_train_ctx:
                query_ids = query_ids[-max_train_ctx:]
            response_ids = self.tokenizer(corrected_response, return_tensors="pt").input_ids[0]
            self.current_episode_turn_data.append((query_ids, response_ids, is_random))

            self.current_episode_messages += [{"role": "assistant", "content": assistant_msg}]
            self.current_llm_input += [{"role": "assistant", "content": assistant_msg}]
            return is_random, action, response

        # Non-rank-0: return dummy values (caller should not use them)
        return None, None, None

    def set_enemy_move_desc(self, desc: str):
        self.last_enemy_move_desc = desc

    def assign_reward(self, reward: float, shaped_reward: Optional[float] = None):
        self.current_episode_rewards.append(reward)
        self.current_episode_shaped_rewards.append(
            shaped_reward if shaped_reward is not None else reward
        )

    def terminate_episode(self, train=True, gamma: float = 0.99):
        """End an episode. Only rank 0 manages the buffer; all ranks call train_step together."""
        if rank == 0 and train and self.current_episode_turn_data:
            turns = self.current_episode_turn_data
            shaped = self.current_episode_shaped_rewards[:len(turns)]
            raw = self.current_episode_rewards

            terminal_reward = raw[-1] if raw else 0.0
            terminal_bonus = distribute_terminal_bonus(terminal_reward, len(turns), gamma)

            for i, ((q, r, was_random), bonus) in enumerate(zip(turns, terminal_bonus)):
                if not was_random:
                    per_turn = shaped[i] if i < len(shaped) else 0.0
                    combined = per_turn + bonus
                    self.grpo_trainer.add_to_buffer(q, r, combined)

        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_llm_input = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_episode_rewards = []
        self.current_episode_shaped_rewards = []
        self.current_episode_turn_data = []
        self.last_enemy_move_desc = None

        if train:
            all_stats = {}
            batch_reward_stds = []
            mfu_steps = []
            hfu_steps = []
            while True:
                step_stats = self.grpo_trainer.train_step()
                if not step_stats:
                    break
                all_stats = step_stats
                br = step_stats.get("grpo/batch_reward_std")
                if br is not None and np.isfinite(br):
                    batch_reward_stds.append(br)
                mfu_val = step_stats.get("mfu/mfu")
                if mfu_val is not None:
                    mfu_steps.append(mfu_val)
                hfu_val = step_stats.get("mfu/hfu")
                if hfu_val is not None:
                    hfu_steps.append(hfu_val)
            if batch_reward_stds:
                all_stats["grpo/batch_reward_std_mean"] = float(np.mean(batch_reward_stds))
                all_stats["grpo/grpo_train_steps"] = len(batch_reward_stds)
            if mfu_steps:
                all_stats["mfu/mean_mfu_episode"] = float(np.mean(mfu_steps))
            if hfu_steps:
                all_stats["mfu/mean_hfu_episode"] = float(np.mean(hfu_steps))
            return all_stats

        return {}


class ChineseChessAgent(Agent):
    _ACTION_RE = re.compile(
        r".*Action:\s*(\d+)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)"
    )

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Chinese Chess (Xiangqi) player. Your goal is to capture the enemy General.\n\n"
            "BOARD: A 10x9 numpy array where positive integers are your pieces, "
            "negative integers are enemy pieces, and 0 is empty.\n\n"
            "PIECE IDS (your pieces):\n"
            "  1=General, 2=Advisor_1, 3=Advisor_2, 4=Elephant_1, 5=Elephant_2,\n"
            "  6=Horse_1, 7=Horse_2, 8=Chariot_1, 9=Chariot_2, 10=Cannon_1, 11=Cannon_2,\n"
            "  12=Soldier_1, 13=Soldier_2, 14=Soldier_3, 15=Soldier_4, 16=Soldier_5\n\n"
            "Each turn you will see the board and a list of your legal moves. "
            "Pick one and respond with ONLY the action in this exact format:\n"
            "Action: piece_id, (start_row, start_col), (end_row, end_col)\n\n"
            "Example: Action: 8, (9, 0), (5, 0)\n"
            "This moves Chariot_1 (piece 8) from row 9 col 0 to row 5 col 0.\n\n"
            "GENERAL (piece 1): In the opening and early middlegame, do NOT shuffle the General "
            "for small steps or vague 'safety'. Keep it in the palace unless you must escape check, "
            "block an immediate mating threat, or no developing move (chariot, horse, cannon, soldier) "
            "is reasonable.\n\n"
            "STRATEGY TIPS: Protect your General without moving it unnecessarily; control the center; "
            "develop Chariots, Horses, and Cannons early; look for captures with those pieces."
        )

    # First N ally moves in an episode: extra nudge + list non-General moves before General moves.
    EARLY_GAME_ALLY_MOVES = 50

    def format_observation(
        self, observation: gym.core.ObsType, env: gym.Env = None, ally_turn_index: int = 0,
        enemy_move_desc: Optional[str] = None,
    ) -> str:
        message = ""
        if enemy_move_desc:
            message += f"{enemy_move_desc}\n\n"
        message += f"The current board looks like this:\n{observation}\n"
        if env is not None:
            legal_actions = np.where(env.ally_actions == 1)[0]
            lines_in_env_order = []
            non_general_lines = []
            general_lines = []
            for a in legal_actions:
                pid, start, end = action_space_to_move(a)
                name = PIECE_ID_TO_NAME[pid] if pid < len(PIECE_ID_TO_NAME) else 'UNKNOWN'
                line = (
                    f"  {pid} ({name}): ({start[0]}, {start[1]}) -> ({end[0]}, {end[1]})"
                )
                lines_in_env_order.append(line)
                if pid == 1:
                    general_lines.append(line)
                else:
                    non_general_lines.append(line)
            # Early game: list non-General moves first so the model sees developing options first.
            if ally_turn_index < self.EARLY_GAME_ALLY_MOVES:
                ordered = non_general_lines + general_lines
            else:
                ordered = lines_in_env_order
            message += "Your legal moves:\n" + "\n".join(ordered)
            if (
                ally_turn_index < self.EARLY_GAME_ALLY_MOVES
                and general_lines
                and non_general_lines
            ):
                message += (
                    "\n\n(Early game — prefer a move that is NOT piece 1 General unless you are in "
                    "check or must block an immediate threat.)"
                )
        return message

    def extract_action(self, response: str, env: gym.Env) -> Tuple[bool, Any]:
        match = self._ACTION_RE.search(response)
        if match:
            piece_id = int(match.group(1))
            from_row = max(0, min(9, int(match.group(2))))
            from_col = max(0, min(8, int(match.group(3))))
            to_row = max(0, min(9, int(match.group(4))))
            to_col = max(0, min(8, int(match.group(5))))
            action = move_to_action_space(
                piece_id, (from_row, from_col), (to_row, to_col)
            )
            legal = np.where(env.ally_actions == 1)[0]
            if action in legal:
                return False, action

        legal_moves = np.where(env.ally_actions == 1)[0]
        return True, np.random.choice(legal_moves)


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

max_input_token = 8192

hyperparams = {
    "model_name": "Qwen/Qwen2.5-14B-Instruct",
    "env": "gym_xiangqi:xiangqi-v0",
    "lora/r": 32,
    "lora/lora_alpha": 64,
    "lora/lora_dropout": 0.05,
    "lora/bias": "none",
    "lora/task_type": "CAUSAL_LM",
    "load_in_8bit": False,
    "grpo/batch_size": 8,
    "grpo/lr": 1e-6,
    "grpo/beta": 0.1,
    "seed": 42069,
    "episodes": 500,
    "record_video": False,
    "record_video_dir": "./video",
    "record_video_every_n": 10,  # wrapper episode index: record when id % N == 0
    "generate/max_new_tokens": 50,
    "generate/do_sample": True,
    "generate/top_p": 0.6,
    "generate/top_k": 0,
    "generate/temperature": 0.9,
    # LoRA adapter checkpoints (base weights not saved — reload base_model + this dir in PeftModel.from_pretrained)
    "checkpoint/dir": "./checkpoints/xiangqi_grpo_lora",
    "checkpoint/every_n_episodes": 25,  # 0 = disable periodic saves (final / interrupted still apply)
    # Set to a saved adapter dir (e.g. ./checkpoints/xiangqi_grpo_lora/final) to resume training; empty = fresh LoRA
    "checkpoint/load_adapter_path": "",
    # True: wipe episode metrics CSV when this script starts (one file per run, no appended old runs)
    "metrics/clear_csv_on_start": True,
    # Reward shaping
    "reward/material_scale": 0.1,  # multiplier on per-turn material delta
    "reward/terminal_gamma": 0.99,  # discount factor for distributing terminal reward
    # Inference-only: load saved LoRA, play N games with RecordVideo, then exit (no training)
    "inference/record_only": False,
    "inference/adapter_path": "",  # e.g. ./checkpoints/xiangqi_grpo_lora/final (falls back to checkpoint/load_adapter_path)
    "inference/num_games": 1,
    "inference/video_dir": "./video_inference",
    "inference/enemy": "greedy",
}

_infer_only = bool(hyperparams.get("inference/record_only", False))
if rank == 0:
    _wandb_init_kw: Dict[str, Any] = {
        "project": os.environ.get("WANDB_PROJECT"),
        "config": hyperparams,
    }
    if _infer_only:
        _wandb_init_kw["mode"] = "disabled"
    wandb.init(**_wandb_init_kw)
else:
    wandb.init(mode="disabled")

# ─── Load model on CPU (FSDP will handle sharding across devices) ───
# Rank 0 downloads/caches the model first; other ranks wait then load from cache.

if rank == 0:
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model_name"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
dist.barrier()
if rank != 0:
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model_name"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
dist.barrier()
device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

# ─── Apply LoRA (new) or load saved adapter (resume) ───

lora_config = LoraConfig(
    **{key.split("/")[-1]: value for key, value in hyperparams.items() if key.startswith("lora/")}
)

_infer_adapter = (hyperparams.get("inference/adapter_path") or "").strip()
_train_adapter = (hyperparams.get("checkpoint/load_adapter_path") or "").strip()
if _infer_only:
    _adapter_dir = _infer_adapter or _train_adapter
    if not _adapter_dir:
        raise ValueError(
            "inference/record_only requires inference/adapter_path or checkpoint/load_adapter_path "
            "to point at a saved LoRA directory."
        )
    _peft_trainable = False
else:
    _adapter_dir = _train_adapter
    _peft_trainable = True

if _adapter_dir:
    if not os.path.isdir(_adapter_dir):
        raise FileNotFoundError(f"LoRA adapter path is not a directory: {_adapter_dir!r}")
    model = PeftModel.from_pretrained(
        model, _adapter_dir, is_trainable=_peft_trainable
    )
    _tok_src = (
        _adapter_dir
        if os.path.isfile(os.path.join(_adapter_dir, "tokenizer_config.json"))
        else hyperparams["model_name"]
    )
    tokenizer = AutoTokenizer.from_pretrained(_tok_src)
    print(
        f"[checkpoint] Loaded LoRA adapter from {_adapter_dir!r} "
        f"(trainable={_peft_trainable})\n"
    )
else:
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))

for layer in model.base_model.model.model.layers:
    fully_shard(layer, mesh=mesh, **fsdp_kwargs)
fully_shard(model.base_model.model, mesh=mesh, **fsdp_kwargs)

# ─── (Optional) Load a previously-saved full state dict into the sharded model ───
# Uncomment if you have a model_state_dict.pt to restore. rank 0 reads the file
# and broadcasts shards to all other ranks.
#
# _sd_path = "model_state_dict.pt"
# if os.path.isfile(_sd_path):
#     full_sd = torch.load(_sd_path, mmap=True, weights_only=True, map_location="cpu")
#     set_model_state_dict(
#         model=model,
#         model_state_dict=full_sd,
#         options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
#     )
#     del full_sd
#     if rank == 0:
#         print(f"[checkpoint] Loaded state dict from {_sd_path!r}")


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total else 0
    return trainable, total, pct


_t, _all, _pct = count_trainable_params(model)
if rank == 0:
    print(f"GRPO trainable params: {_t:,} / {_all:,} ({_pct:.2f}%)")
    _gpu_name = torch.cuda.get_device_name(local_rank)
    _mfu_dtype = _resolve_compute_dtype_str(fsdp_kwargs.get("mp_policy"))
    _peak_tf = _get_gpu_peak_tflops(local_rank, _mfu_dtype)
    _grad_ckpt_on = getattr(model, "is_gradient_checkpointing", False)
    print(
        f"MFU profiling: GPU={_gpu_name} | compute_dtype={_mfu_dtype} | "
        f"peak={_peak_tf:.1f} TFLOPS | total_params={_all:,} | "
        f"trainable_params={_t:,} | grad_checkpointing={_grad_ckpt_on}\n"
    )


def save_lora_checkpoint(checkpoint_path: str, episode: int, label: str = "") -> None:
    """Save PEFT adapter + tokenizer + small JSON meta.
    Under FSDP the params are DTensor shards — gather them first via get_model_state_dict,
    then only rank 0 writes files.
    """
    full_sd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path, state_dict=full_sd)
        tokenizer.save_pretrained(checkpoint_path)
        meta = {
            "episode": episode,
            "label": label or None,
            "base_model": hyperparams["model_name"],
            "lora": {k: v for k, v in hyperparams.items() if k.startswith("lora/")},
        }
        with open(os.path.join(checkpoint_path, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[checkpoint] Saved LoRA adapter to {checkpoint_path!r}")
    dist.barrier()


_ckpt_root = hyperparams.get("checkpoint/dir") or "./checkpoints/xiangqi_grpo_lora"
_ckpt_every = int(hyperparams.get("checkpoint/every_n_episodes", 0) or 0)

# ─── Initialize GRPO trainer ───

_active_mp_policy = fsdp_kwargs.get("mp_policy")
grpo_trainer = GRPOTrainerOnline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size=hyperparams["grpo/batch_size"],
    lr=hyperparams["grpo/lr"],
    beta=hyperparams["grpo/beta"],
    mp_policy=_active_mp_policy,
)

# ─── Initialize Environment and Agents ───

gap_size = 10

env = None
if rank == 0 and not _infer_only:
    _record_video = bool(hyperparams.get("record_video", False))
    _record_dir = hyperparams.get("record_video_dir", "./video")
    _record_every = int(hyperparams.get("record_video_every_n", 10))

    _make_kwargs = {}
    if _record_video:
        _make_kwargs["render_mode"] = "rgb_array"

    if _make_kwargs:
        try:
            env = gym.make(hyperparams["env"], **_make_kwargs)
        except Exception as e:
            env = gym.make(hyperparams["env"])
            if _record_video:
                print(
                    f"record_video: could not use render_mode='rgb_array' ({e!r}); "
                    "recording disabled. Check gym_xiangqi / gym version."
                )
                _record_video = False
    else:
        env = gym.make(hyperparams["env"])

    if _record_video:
        if RecordVideo is None:
            print("RecordVideo not available (install gym>=0.26 or gymnasium); skipping video.")
        else:
            os.makedirs(_record_dir, exist_ok=True)
            env = RecordVideo(
                env,
                _record_dir,
                episode_trigger=lambda episode_id: episode_id % _record_every == 0,
                name_prefix="xiangqi",
            )
            print(
                f"Recording video every {_record_every} wrapper episode(s) -> {_record_dir!r} "
                "(episode_trigger uses the wrapper's internal episode index, not your training loop counter)."
            )

enemy_agent = GreedyEnemyAgent() if rank == 0 else None
_material_scale = float(hyperparams.get("reward/material_scale", 0.1))
_terminal_gamma = float(hyperparams.get("reward/terminal_gamma", 0.99))
ally_agent = ChineseChessAgent(
    model,
    tokenizer,
    grpo_trainer,
    max_input_token,
    device,
    {k.split("/")[-1]: v for k, v in hyperparams.items() if k.startswith("generate/")},
)


def _env_turn(e) -> int:
    inner = getattr(e, "unwrapped", None)
    if inner is not None and hasattr(inner, "turn"):
        return int(inner.turn)
    cur = e
    for _ in range(32):
        if hasattr(cur, "turn"):
            return int(cur.turn)
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    raise RuntimeError("Could not find .turn on env (RecordVideo / Xiangqi)")


_INFER_SIGNAL_ALLY_TURN = 1
_INFER_SIGNAL_GAME_DONE = 2
_INFER_SIGNAL_ALL_DONE = 3


def run_inference_record_videos(
    infer_ally_agent: ChineseChessAgent,
    infer_enemy: GreedyEnemyAgent,
) -> None:
    """Inference with RecordVideo. Rank 0 drives the env; all ranks participate in generate()."""
    num_games = max(1, int(hyperparams.get("inference/num_games", 1)))
    video_dir = hyperparams.get("inference/video_dir", "./video_inference")

    infer_env = None
    if rank == 0:
        if RecordVideo is None:
            raise RuntimeError(
                "inference/record_only requires RecordVideo (gym>=0.26 or gymnasium with RecordVideo)."
            )
        os.makedirs(video_dir, exist_ok=True)
        try:
            infer_env = gym.make(hyperparams["env"], render_mode="rgb_array")
        except Exception as e:
            raise RuntimeError(
                f"inference recording needs render_mode='rgb_array'. ({e!r})"
            ) from e
        infer_env = RecordVideo(
            infer_env, video_dir,
            episode_trigger=lambda episode_id, cap=num_games: episode_id < cap,
            name_prefix="xiangqi_infer",
        )
        print(
            f"[inference] Recording up to {num_games} game(s), opponent=GreedyEnemy -> {video_dir!r}"
        )

    model.eval()

    for game in range(1, num_games + 1):
        if rank == 0:
            raw_obs = infer_env.reset()
            observation = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs
            round_idx, done = 1, False
            ally_reward = enemy_reward = 0.0
            print(f"[inference game {game}/{num_games}] Opponent: GreedyEnemy")
            episode_ally_turns = episode_random_fallback = episode_act_failures = 0
            current_ally_rewards = current_enemy_rewards = 0.0

        while True:
            if rank == 0:
                while not done and _env_turn(infer_env) != ALLY:
                    enemy_action = infer_enemy.move(infer_env)
                    step_out = infer_env.step(enemy_action)
                    if len(step_out) == 5:
                        observation, enemy_reward, terminated, truncated, _ = step_out
                        done = bool(terminated or truncated)
                    else:
                        observation, enemy_reward, done, _ = step_out
                    current_enemy_rewards += enemy_reward
                    _log_env_step(game, round_idx, "Enemy (GreedyEnemy)", enemy_action, enemy_reward)
                    round_idx += 1
                    if round_idx >= 200:
                        print(f"[inference g{game} Rd {round_idx}] hit 200-round cap.")
                        done = True

                signal = _INFER_SIGNAL_GAME_DONE if done else _INFER_SIGNAL_ALLY_TURN
            else:
                signal = 0

            signal = broadcast_int(signal)
            if signal == _INFER_SIGNAL_GAME_DONE:
                break

            if rank == 0:
                used_random_fallback, ally_action, llm_output = infer_ally_agent.act(
                    observation, game, round_idx, infer_env, ally_turn_index=episode_ally_turns,
                )
            else:
                infer_ally_agent.act(None, game, 0, None, ally_turn_index=0)

            if rank == 0:
                if ally_action is None:
                    episode_act_failures += 1
                    print(f"[infer g{game} Rd {round_idx}] WARNING act() returned None")
                else:
                    episode_ally_turns += 1
                    if used_random_fallback:
                        episode_random_fallback += 1
                step_out = infer_env.step(ally_action)
                if len(step_out) == 5:
                    observation, ally_reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    observation, ally_reward, done, _ = step_out
                current_ally_rewards += ally_reward
                infer_ally_agent.assign_reward(ally_reward)
                if ally_action is not None:
                    _log_env_step(game, round_idx, "Ally action", ally_action, ally_reward,
                                  random_fallback=used_random_fallback)
                round_idx += 1
                if round_idx >= 200 and not done:
                    print(f"[inference g{game} Rd {round_idx}] hit 200-round cap.")
                    done = True

        if rank == 0:
            outcome = (
                "enemy_win" if enemy_reward == 100
                else "ally_win" if ally_reward == 100
                else "other/trunc"
            )
            print(
                f"[inference game {game}/{num_games}] done: {outcome} "
                f"(ally_ret={current_ally_rewards:.1f}, enemy_ret={current_enemy_rewards:.1f}, rounds={round_idx})"
            )
        infer_ally_agent.terminate_episode(train=False)

    if rank == 0:
        infer_env.close()
        print(f"[inference] Finished. Videos under {video_dir!r}")


def _log_env_step(ep, rd, side, action_idx, step_reward, **extra):
    pid, start, end = action_space_to_move(action_idx)
    piece = PIECE_ID_TO_NAME[pid]
    parts = [
        f"[Ep {ep} Rd {rd}] {side}",
        f"{piece} {start} -> {end}",
        f"step_reward={step_reward}",
    ]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    print(" | ".join(parts))
    print("================")


# ─── Training loop ───
# Rank 0 drives the gym environment. Before every model.generate() call,
# rank 0 broadcasts a signal so ALL ranks participate in the FSDP collective.
# Enemy turns, env stepping, logging, and metrics are rank-0 only.

_SIGNAL_ALLY_TURN = 1
_SIGNAL_EPISODE_DONE = 2

if _infer_only:
    run_inference_record_videos(ally_agent, enemy_agent)

if not _infer_only:
    ally_wins = enemy_wins = truncated_game = 0
    episode_lengths, ally_rewards, enemy_rewards, winning_rewards = [], [], [], []
    series_ally_win_rate_cumulative = []
    series_batch_reward_std = []
    series_trained_turn_fraction = []
    lifetime_ally_turns = lifetime_random_fallback = lifetime_act_failures = 0

    if rank == 0 and hyperparams.get("metrics/clear_csv_on_start", True):
        reset_episode_metrics_csv(EPISODE_METRICS_CSV)
        print(f"[metrics] Cleared {EPISODE_METRICS_CSV!r} for this run.\n")

    try:
        for episode in (trange(1, hyperparams["episodes"] + 1) if rank == 0 else range(1, hyperparams["episodes"] + 1)):

            if rank == 0:
                observation = env.reset()
                round_idx, done = 1, False
                current_ally_rewards = current_enemy_rewards = 0.0
                episode_ally_turns = episode_random_fallback = episode_act_failures = 0
                episode_hit_round_cap = False
                ally_reward = enemy_reward = 0.0
                print(f"[Ep {episode}] Opponent: GreedyEnemy")

            while True:
                if rank == 0:
                    while not done and env.turn != ALLY:
                        board_before_enemy = env.state.copy()
                        enemy_action = enemy_agent.move(env)
                        observation, enemy_reward, done, _ = env.step(enemy_action)
                        current_enemy_rewards += enemy_reward

                        enemy_desc = describe_enemy_move(
                            board_before_enemy, env.state, enemy_action,
                        )
                        ally_agent.set_enemy_move_desc(enemy_desc)

                        _log_env_step(
                            episode, round_idx, "Enemy (GreedyEnemy)",
                            enemy_action, enemy_reward,
                        )
                        round_idx += 1
                        if round_idx >= 200:
                            print(
                                f"[Ep {episode} Rd {round_idx}] "
                                f"Episode reached 200 rounds, stopping."
                            )
                            done = True
                            truncated_game += 1
                            episode_hit_round_cap = True

                    if done:
                        signal = _SIGNAL_EPISODE_DONE
                    else:
                        signal = _SIGNAL_ALLY_TURN
                else:
                    signal = 0

                signal = broadcast_int(signal)

                if signal == _SIGNAL_EPISODE_DONE:
                    break

                if rank == 0:
                    board_before_ally = env.state.copy()
                    used_random_fallback, ally_action, llm_output = ally_agent.act(
                        observation, episode, round_idx, env,
                        ally_turn_index=episode_ally_turns,
                    )
                else:
                    ally_agent.act(None, episode, 0, None, ally_turn_index=0)

                if rank == 0:
                    if ally_action is None:
                        episode_act_failures += 1
                        lifetime_act_failures += 1
                        print(f"[Ep {episode} Rd {round_idx}] WARNING act() returned None")
                    else:
                        episode_ally_turns += 1
                        if used_random_fallback:
                            episode_random_fallback += 1
                    _, ally_reward, done, _ = env.step(ally_action)
                    current_ally_rewards += ally_reward

                    shaped = compute_shaped_reward(
                        board_before_ally, ally_action, ally_reward,
                        scale=_material_scale,
                    )
                    ally_agent.assign_reward(ally_reward, shaped_reward=shaped)

                    if ally_action is not None:
                        _log_env_step(
                            episode, round_idx, "Ally action",
                            ally_action, ally_reward,
                            random_fallback=used_random_fallback,
                            shaped_reward=round(shaped, 4),
                        )
                    if episode % gap_size == 0 and ally_action is not None:
                        training_log_llm_output(
                            "chinese_chess_llm_output_log", episode, round_idx, llm_output
                        )
                    round_idx += 1
                    if round_idx >= 200 and not done:
                        print(
                            f"[Ep {episode} Rd {round_idx}] "
                            f"Episode reached 200 rounds, stopping."
                        )
                        done = True
                        truncated_game += 1
                        episode_hit_round_cap = True

            train_stats = ally_agent.terminate_episode(gamma=_terminal_gamma)

            # --- Rank 0: bookkeeping, logging, checkpointing ---
            if rank == 0:
                episode_lengths.append(round_idx)
                if enemy_reward == 100:
                    enemy_wins += 1
                    winning_rewards.append(current_enemy_rewards)
                elif ally_reward == 100:
                    ally_wins += 1
                    winning_rewards.append(current_ally_rewards)

                ally_rewards.append(current_ally_rewards)
                enemy_rewards.append(current_enemy_rewards)

                lifetime_ally_turns += episode_ally_turns
                lifetime_random_fallback += episode_random_fallback
                random_rate_episode = (
                    100.0 * episode_random_fallback / episode_ally_turns if episode_ally_turns else 0.0
                )
                random_rate_lifetime = (
                    100.0 * lifetime_random_fallback / lifetime_ally_turns
                    if lifetime_ally_turns
                    else 0.0
                )
                trained_turn_fraction = (
                    (episode_ally_turns - episode_random_fallback) / episode_ally_turns
                    if episode_ally_turns
                    else 0.0
                )

                episode_stats = {
                    "episode": episode,
                    "episode_length": round_idx,
                    "total_return": current_ally_rewards + current_enemy_rewards,
                    "ally_return": current_ally_rewards,
                    "enemy_return": current_enemy_rewards,
                    "message_ct": len(ally_agent.current_episode_messages),
                    "episode_messages": ally_agent.current_episode_messages,
                    "ally_wins": ally_wins,
                    "enemy_wins": enemy_wins,
                    "truncated_games": truncated_game,
                    "ally_win_rate": calculate_win_rate(ally_wins, episode),
                    "enemy_win_rate": calculate_win_rate(enemy_wins, episode),
                    "truncated_rate": calculate_win_rate(truncated_game, episode),
                    "ally_turns_episode": episode_ally_turns,
                    "random_fallback_episode": episode_random_fallback,
                    "random_move_rate_episode": random_rate_episode,
                    "random_move_rate_lifetime": random_rate_lifetime,
                    "act_failures_episode": episode_act_failures,
                    "act_failures_lifetime": lifetime_act_failures,
                    "trained_turn_fraction": trained_turn_fraction,
                    "enemy_policy": "greedy",
                }
                episode_stats.update(train_stats)

                append_episode_metrics_csv(
                    EPISODE_METRICS_CSV,
                    episode=episode,
                    rounds=round_idx,
                    ally_return=current_ally_rewards,
                    enemy_return=current_enemy_rewards,
                    ally_reward_term=ally_reward,
                    enemy_reward_term=enemy_reward,
                    episode_hit_round_cap=episode_hit_round_cap,
                    trained_turn_fraction=trained_turn_fraction,
                    random_move_pct_episode=random_rate_episode,
                    train_stats=train_stats,
                    enemy_policy="greedy",
                )

                series_ally_win_rate_cumulative.append(episode_stats["ally_win_rate"])
                series_trained_turn_fraction.append(trained_turn_fraction)
                br_std_ep = train_stats.get("grpo/batch_reward_std_mean")
                if br_std_ep is None:
                    br_std_ep = train_stats.get("grpo/batch_reward_std")
                series_batch_reward_std.append(
                    br_std_ep if br_std_ep is not None else float("nan")
                )

                wandb.log(episode_stats)

            if _ckpt_every > 0 and episode % _ckpt_every == 0:
                save_lora_checkpoint(
                    os.path.join(_ckpt_root, f"ep_{episode}"),
                    episode,
                    label=f"every_{_ckpt_every}",
                )

            if rank == 0 and episode % gap_size == 0:
                wr = calculate_win_rate(ally_wins, episode)
                ewr = calculate_win_rate(enemy_wins, episode)
                tr = calculate_win_rate(truncated_game, episode)
                br = train_stats.get(
                    "grpo/batch_reward_std_mean", train_stats.get("grpo/batch_reward_std", "n/a")
                )
                print(f"\n[Ep {episode}] --- periodic summary (every {gap_size} eps) ---")
                print("Opponent: GreedyEnemy")
                print(
                    f"Wins: ally {ally_wins} ({wr:.1f}%) | enemy {enemy_wins} ({ewr:.1f}%) | "
                    f"trunc {truncated_game} ({tr:.1f}%)"
                )
                print(
                    f"Last ep random fallback: {episode_random_fallback}/{episode_ally_turns} "
                    f"({random_rate_episode:.1f}%) | lifetime random %: {random_rate_lifetime:.1f}%"
                )
                print(
                    f"trained_turn_fraction={trained_turn_fraction:.3f} | "
                    f"GRPO batch σ mean (ep)={br} | buffer={ally_agent.grpo_trainer.buffer_size()}"
                )
                print(
                    f"GRPO last step: loss={train_stats.get('grpo/loss', 'n/a')} | "
                    f"mean_adv={train_stats.get('grpo/mean_advantage', 'n/a')} | "
                    f"mean_kl={train_stats.get('grpo/mean_kl', 'n/a')} | "
                    f"mean_rew={train_stats.get('grpo/mean_reward', 'n/a')}"
                )
                _mfu_val = train_stats.get("mfu/mfu")
                _hfu_val = train_stats.get("mfu/hfu")
                if _mfu_val is not None:
                    _step_sec = train_stats.get("mfu/step_time_sec", 0)
                    _mfu_ep_val = train_stats.get("mfu/mean_mfu_episode")
                    _hfu_ep_val = train_stats.get("mfu/mean_hfu_episode")
                    _ep_parts = []
                    if _mfu_ep_val is not None:
                        _ep_parts.append(f"ep_mean_mfu={_mfu_ep_val:.4f}")
                    if _hfu_ep_val is not None:
                        _ep_parts.append(f"ep_mean_hfu={_hfu_ep_val:.4f}")
                    _ep_str = (" | " + " | ".join(_ep_parts)) if _ep_parts else ""
                    _hfu_str = f" | HFU: {_hfu_val:.4f} ({_hfu_val*100:.2f}%)" if _hfu_val is not None else ""
                    print(
                        f"MFU: {_mfu_val:.4f} ({_mfu_val*100:.2f}%){_hfu_str} | "
                        f"step_time={_step_sec:.2f}s{_ep_str}"
                    )
                print("-" * 60)

        if rank == 0 and episode_lengths:
            save_lora_checkpoint(
                os.path.join(_ckpt_root, "final"),
                len(episode_lengths),
                label="normal_completion",
            )

    except Exception:
        if rank == 0:
            print("\n" + "=" * 60)
            print("TRAINING CRASHED — traceback below")
            print("=" * 60)
            traceback.print_exc()
            print("=" * 60)
        try:
            _crash_ep = episode  # type: ignore[name-defined]
        except NameError:
            _crash_ep = None
        if _crash_ep is not None and _crash_ep >= 1:
            try:
                save_lora_checkpoint(
                    os.path.join(_ckpt_root, f"interrupted_ep{_crash_ep}"),
                    _crash_ep,
                    label="interrupted",
                )
            except Exception as save_err:
                if rank == 0:
                    print(f"[checkpoint] Could not save interrupted checkpoint: {save_err}")

if env is not None:
    env.close()

if not _infer_only and rank == 0:
    print(f"\nFinal score after {episode} episodes -> Ally Wins: {ally_wins}, Enemy Wins: {enemy_wins}, Truncated Game: {truncated_game}")
    print(f"Ally Win Rate: {calculate_win_rate(ally_wins, episode)}%")
    print(f"Enemy Win Rate: {calculate_win_rate(enemy_wins, episode)}%")
    print(f"Truncated Rate: {calculate_win_rate(truncated_game, episode)}%")
    print(f"Average Winning Reward per Episode: {average_reward(winning_rewards)}")
    print(f"Average Episode Length: {average_episode_length(episode_lengths)}")
    print(f"Ally Reward Variability: {reward_variability(ally_rewards)}")
    print(f"Enemy Reward Variability: {reward_variability(enemy_rewards)}")
    _mfu_summary = grpo_trainer.mfu_tracker.summary()
    if _mfu_summary:
        print(
            f"\nMFU lifetime: "
            f"mean={_mfu_summary['mfu/lifetime_mean_mfu']:.4f} "
            f"({_mfu_summary['mfu/lifetime_mean_mfu']*100:.2f}%) | "
            f"median={_mfu_summary['mfu/lifetime_median_mfu']:.4f} | "
            f"tflops={_mfu_summary['mfu/lifetime_mean_mfu_tflops']:.2f}"
        )
        print(
            f"HFU lifetime: "
            f"mean={_mfu_summary['mfu/lifetime_mean_hfu']:.4f} "
            f"({_mfu_summary['mfu/lifetime_mean_hfu']*100:.2f}%) | "
            f"median={_mfu_summary['mfu/lifetime_median_hfu']:.4f} | "
            f"tflops={_mfu_summary['mfu/lifetime_mean_hfu_tflops']:.2f}"
        )
        print(f"Steps profiled: {_mfu_summary['mfu/lifetime_steps_profiled']}")
    plot_episode_lengths(episode_lengths, ally="LLM-Agent", enemy="Random")
    plot_episode_rewards(ally_rewards, enemy_rewards, ally="LLM-Agent", enemy="Random")
    if series_ally_win_rate_cumulative:
        plot_grpo_diagnostics(
            series_ally_win_rate_cumulative,
            series_batch_reward_std,
            series_trained_turn_fraction,
        )

dist.destroy_process_group()
    
