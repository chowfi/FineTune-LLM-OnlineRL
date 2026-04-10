import csv
import json
import os
import random
import re
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
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gym_xiangqi.agents import RandomAgent
from gym_xiangqi.constants import ALLY, PIECE_ID_TO_NAME, PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move, move_to_action_space


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

    def __init__(self, model, tokenizer, device, batch_size=8, lr=1e-5, beta=0.1, max_grad_norm=1.0):
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
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = self.model(input_ids=input_ids).logits
        response_start = query_ids.size(0)
        response_logits = logits[0, response_start - 1 : -1, :]
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
        return token_log_probs.sum()

    def train_step(self):
        if self.buffer_size() < self.batch_size:
            return {}

        query_ids_batch = self.buffer["query_ids"][: self.batch_size]
        response_ids_batch = self.buffer["response_ids"][: self.batch_size]
        rewards_batch = self.buffer["rewards"][: self.batch_size]

        self.buffer["query_ids"] = self.buffer["query_ids"][self.batch_size :]
        self.buffer["response_ids"] = self.buffer["response_ids"][self.batch_size :]
        self.buffer["rewards"] = self.buffer["rewards"][self.batch_size :]

        rewards_t = torch.tensor(rewards_batch, dtype=torch.float32)
        reward_std_before_norm = float(rewards_t.std().item())
        if reward_std_before_norm > 1e-4:
            advantages = (rewards_t - rewards_t.mean()) / (reward_std_before_norm + 1e-8)
        else:
            advantages = rewards_t - rewards_t.mean()

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

        torch.cuda.empty_cache()

        stats = {
            "grpo/loss": total_loss_val,
            "grpo/mean_advantage": advantages.mean().item(),
            "grpo/mean_kl": total_kl_val / n,
            "grpo/mean_reward": rewards_t.mean().item(),
            "grpo/batch_reward_std": reward_std_before_norm,
        }
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
        self.current_episode_turn_data = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(
        self, observation: gym.core.ObsType, env: gym.Env = None, ally_turn_index: int = 0
    ) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str, env: gym.Env) -> Tuple[bool, Any]:
        pass

    def llm(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        context_len = inputs['attention_mask'].size(1)
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids,
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
        message = self.format_observation(observation, env, ally_turn_index=ally_turn_index)
        self.current_episode_messages += [{"role": "user", "content": message}]
        self.current_llm_input += [{"role": "user", "content": message}]

        prompt = self.tokenizer.apply_chat_template(
            self.current_llm_input, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        response = self.llm(self.current_llm_input)
        print(f"[Ep {episode} Rd {round}] LLM Response: {response}")

        try:
            is_random, action = self.extract_action(response, env)
        except Exception as e:
            return None, None, response

        if is_random:
            piece_id, start, end = action_space_to_move(action)
            corrected_response = f"Action: {piece_id}, ({start[0]}, {start[1]}), ({end[0]}, {end[1]})"
        else:
            corrected_response = response

        query_ids = inputs.input_ids[0].cpu()
        max_train_ctx = 4096
        if len(query_ids) > max_train_ctx:
            query_ids = query_ids[-max_train_ctx:]
        response_ids = self.tokenizer(corrected_response, return_tensors="pt").input_ids[0]
        self.current_episode_turn_data.append((query_ids, response_ids, is_random))

        self.current_episode_messages += [{"role": "assistant", "content": corrected_response}]
        self.current_llm_input += [{"role": "assistant", "content": corrected_response}]
        return is_random, action, response

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def terminate_episode(self, train=True):
        if train and self.current_episode_turn_data:
            rewards = self.current_episode_rewards
            turns = self.current_episode_turn_data

            if all(r == 0 for r in rewards[:-1]) and len(rewards) > 0:
                per_turn = rewards[-1] / max(len(turns), 1)
                distributed_rewards = [per_turn] * len(turns)
            else:
                distributed_rewards = rewards[:len(turns)]

            for (q, r, was_random), rew in zip(turns, distributed_rewards):
                if not was_random:
                    self.grpo_trainer.add_to_buffer(q, r, rew)

        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_llm_input = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_episode_rewards = []
        self.current_episode_turn_data = []

        if train:
            all_stats = {}
            batch_reward_stds = []
            while self.grpo_trainer.buffer_size() >= self.grpo_trainer.batch_size:
                step_stats = self.grpo_trainer.train_step()
                all_stats = step_stats
                br = step_stats.get("grpo/batch_reward_std")
                if br is not None and np.isfinite(br):
                    batch_reward_stds.append(br)
            if batch_reward_stds:
                all_stats["grpo/batch_reward_std_mean"] = float(np.mean(batch_reward_stds))
                all_stats["grpo/grpo_train_steps"] = len(batch_reward_stds)
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
        self, observation: gym.core.ObsType, env: gym.Env = None, ally_turn_index: int = 0
    ) -> str:
        message = f"The current board looks like this:\n{observation}\n"
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

max_input_token = 14000

hyperparams = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "env": "gym_xiangqi:xiangqi-v0",
    "lora/r": 16,
    "lora/lora_alpha": 32,
    "lora/lora_dropout": 0.05,
    "lora/bias": "none",
    "lora/task_type": "CAUSAL_LM",
    # False: FP16 on GPU (more VRAM, avoids bitsandbytes CUDA paths / timeouts)
    "load_in_8bit": False,
    "grpo/batch_size": 8,
    "grpo/lr": 2e-6,
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
    # From this episode index onward, each episode picks all-greedy or all-random enemy (Bernoulli)
    "enemy/mix_greedy_start_episode": 80,
    "enemy/greedy_episode_probability": 0.5,  # P(greedy episode); 1-P is random-opponent episode
    # Inference-only: load saved LoRA, play N games with RecordVideo, then exit (no training)
    "inference/record_only": False,
    "inference/adapter_path": "",  # e.g. ./checkpoints/xiangqi_grpo_lora/final (falls back to checkpoint/load_adapter_path)
    "inference/num_games": 1,
    "inference/video_dir": "./video_inference",
    "inference/enemy": "random",  # "random" or "greedy" for the whole recorded run
}

_infer_only = bool(hyperparams.get("inference/record_only", False))
_wandb_init_kw: Dict[str, Any] = {
    "project": os.environ.get("WANDB_PROJECT"),
    "config": hyperparams,
}
if _infer_only:
    _wandb_init_kw["mode"] = "disabled"
wandb.init(**_wandb_init_kw)

# ─── Load model ───

load_in_8bit = bool(hyperparams.get("load_in_8bit", False))
if load_in_8bit:
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    hyperparams["model_name"],
    quantization_config=quantization_config,
    device_map="auto" if load_in_8bit else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

if not load_in_8bit:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
    device = next(model.parameters()).device
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
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total else 0
    return trainable, total, pct


_t, _all, _pct = count_trainable_params(model)
print(f"GRPO trainable params: {_t:,} / {_all:,} ({_pct:.2f}%)\n")


def save_lora_checkpoint(checkpoint_path: str, episode: int, label: str = "") -> None:
    """Save PEFT adapter + tokenizer + small JSON meta (not full base model weights)."""
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
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


_ckpt_root = hyperparams.get("checkpoint/dir") or "./checkpoints/xiangqi_grpo_lora"
_ckpt_every = int(hyperparams.get("checkpoint/every_n_episodes", 0) or 0)

# ─── Initialize GRPO trainer ───

grpo_trainer = GRPOTrainerOnline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size=hyperparams["grpo/batch_size"],
    lr=hyperparams["grpo/lr"],
    beta=hyperparams["grpo/beta"],
)

# ─── Initialize Environment and Agents ───

gap_size = 10

env = None
if not _infer_only:
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

enemy_agent_random = RandomAgent()
enemy_agent_greedy = GreedyEnemyAgent()
_greedy_start = int(hyperparams.get("enemy/mix_greedy_start_episode", 80))
_greedy_p = float(
    hyperparams.get(
        "enemy/greedy_episode_probability",
        hyperparams.get("enemy/greedy_move_probability", 0.5),
    )
)
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


def run_inference_record_videos(
    infer_ally_agent: ChineseChessAgent,
    infer_enemy_random: RandomAgent,
    infer_enemy_greedy: GreedyEnemyAgent,
) -> None:
    """Load is already done; play N games with RecordVideo (no GRPO updates)."""
    num_games = max(1, int(hyperparams.get("inference/num_games", 1)))
    video_dir = hyperparams.get("inference/video_dir", "./video_inference")
    enemy_kind = (hyperparams.get("inference/enemy", "random") or "random").strip().lower()
    if enemy_kind not in ("random", "greedy"):
        print(f"[inference] Unknown inference/enemy={enemy_kind!r}; using random")
        enemy_kind = "random"

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
        infer_env,
        video_dir,
        episode_trigger=lambda episode_id, cap=num_games: episode_id < cap,
        name_prefix="xiangqi_infer",
    )

    print(
        f"[inference] Recording up to {num_games} game(s), opponent={enemy_kind!r} -> {video_dir!r}"
    )
    model.eval()

    for game in range(1, num_games + 1):
        raw_obs = infer_env.reset()
        observation = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs
        round_idx, done = 1, False
        ally_reward = enemy_reward = 0.0
        use_greedy = enemy_kind == "greedy"
        _ep_opponent = "GreedyEnemy" if use_greedy else "RandomEnemy"
        print(f"[inference game {game}/{num_games}] Opponent: {_ep_opponent}")

        episode_ally_turns = episode_random_fallback = episode_act_failures = 0
        current_ally_rewards = current_enemy_rewards = 0.0

        while not done:
            if _env_turn(infer_env) == ALLY:
                used_random_fallback, ally_action, llm_output = infer_ally_agent.act(
                    observation,
                    game,
                    round_idx,
                    infer_env,
                    ally_turn_index=episode_ally_turns,
                )
                if ally_action is None:
                    episode_act_failures += 1
                    print(
                        f"[infer g{game} Rd {round_idx}] WARNING act() returned None"
                    )
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
                    _log_env_step(
                        game,
                        round_idx,
                        "Ally action",
                        ally_action,
                        ally_reward,
                        random_fallback=used_random_fallback,
                    )
            else:
                enemy_action = (
                    infer_enemy_greedy.move(infer_env)
                    if use_greedy
                    else infer_enemy_random.move(infer_env)
                )
                step_out = infer_env.step(enemy_action)
                if len(step_out) == 5:
                    observation, enemy_reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    observation, enemy_reward, done, _ = step_out
                current_enemy_rewards += enemy_reward
                _log_env_step(
                    game,
                    round_idx,
                    f"Enemy ({_ep_opponent})",
                    enemy_action,
                    enemy_reward,
                )

            round_idx += 1
            if round_idx >= 200:
                print(
                    f"[inference g{game} Rd {round_idx}] hit 200-round cap. Opponent={_ep_opponent}"
                )
                done = True

        outcome = (
            "enemy_win"
            if enemy_reward == 100
            else "ally_win"
            if ally_reward == 100
            else "other/trunc"
        )
        print(
            f"[inference game {game}/{num_games}] done: {outcome} "
            f"(ally_ret={current_ally_rewards:.1f}, enemy_ret={current_enemy_rewards:.1f}, rounds={round_idx})"
        )
        infer_ally_agent.terminate_episode(train=False)

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

if _infer_only:
    run_inference_record_videos(ally_agent, enemy_agent_random, enemy_agent_greedy)

if not _infer_only:
    ally_wins = enemy_wins = truncated_game = 0
    episode_lengths, ally_rewards, enemy_rewards, winning_rewards = [], [], [], []
    series_ally_win_rate_cumulative = []
    series_batch_reward_std = []
    series_trained_turn_fraction = []
    lifetime_ally_turns = lifetime_random_fallback = lifetime_act_failures = 0
    
    if hyperparams.get("metrics/clear_csv_on_start", True):
        reset_episode_metrics_csv(EPISODE_METRICS_CSV)
        print(f"[metrics] Cleared {EPISODE_METRICS_CSV!r} for this run.\n")
    
    try:
        for episode in trange(1, hyperparams["episodes"] + 1):
            observation = env.reset()
            round_idx, done = 1, False
            current_ally_rewards = current_enemy_rewards = 0.0
            episode_ally_turns = episode_random_fallback = episode_act_failures = 0
            episode_hit_round_cap = False
            episode_enemy_greedy_turns = episode_enemy_random_turns = 0
            if episode >= _greedy_start:
                use_greedy_for_episode = random.random() < _greedy_p
            else:
                use_greedy_for_episode = False
            _ep_opponent = "GreedyEnemy" if use_greedy_for_episode else "RandomEnemy"
            _ep_mix_note = (
                ""
                if episode >= _greedy_start
                else f" | greedy mix Random↔Greedy from ep {_greedy_start}"
            )
            print(f"[Ep {episode}] Opponent (full episode): {_ep_opponent}{_ep_mix_note}")
    
            while not done:
                if env.turn == ALLY:
                    used_random_fallback, ally_action, llm_output = ally_agent.act(
                        observation,
                        episode,
                        round_idx,
                        env,
                        ally_turn_index=episode_ally_turns,
                    )
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
                    ally_agent.assign_reward(ally_reward)
                    if ally_action is not None:
                        _log_env_step(
                            episode,
                            round_idx,
                            "Ally action",
                            ally_action,
                            ally_reward,
                            random_fallback=used_random_fallback,
                        )
                    if episode % gap_size == 0 and ally_action is not None:
                        training_log_llm_output(
                            "chinese_chess_llm_output_log", episode, round_idx, llm_output
                        )
                else:
                    if use_greedy_for_episode:
                        enemy_action = enemy_agent_greedy.move(env)
                        episode_enemy_greedy_turns += 1
                    else:
                        enemy_action = enemy_agent_random.move(env)
                        episode_enemy_random_turns += 1
                    observation, enemy_reward, done, _ = env.step(enemy_action)
                    current_enemy_rewards += enemy_reward
                    _log_env_step(
                        episode,
                        round_idx,
                        f"Enemy ({_ep_opponent})",
                        enemy_action,
                        enemy_reward,
                    )
    
                round_idx += 1
                if round_idx >= 200:
                    print(
                        f"[Ep {episode} Rd {round_idx}] "
                        f"Episode reached 200 rounds, stopping to prevent infinite loop. "
                        f"Opponent={_ep_opponent}"
                    )
                    done = True
                    truncated_game += 1
                    episode_hit_round_cap = True
    
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
                "enemy_greedy_turns": episode_enemy_greedy_turns,
                "enemy_random_turns": episode_enemy_random_turns,
                "enemy_policy": "greedy" if use_greedy_for_episode else "random",
            }
    
            train_stats = ally_agent.terminate_episode()
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
                enemy_policy="greedy" if use_greedy_for_episode else "random",
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
    
            if episode % gap_size == 0:
                wr = calculate_win_rate(ally_wins, episode)
                ewr = calculate_win_rate(enemy_wins, episode)
                tr = calculate_win_rate(truncated_game, episode)
                br = train_stats.get(
                    "grpo/batch_reward_std_mean", train_stats.get("grpo/batch_reward_std", "n/a")
                )
                print(f"\n[Ep {episode}] --- periodic summary (every {gap_size} eps) ---")
                print(f"Last episode opponent: {_ep_opponent}")
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
                print("-" * 60)
    
        if episode_lengths:
            save_lora_checkpoint(
                os.path.join(_ckpt_root, "final"),
                len(episode_lengths),
                label="normal_completion",
            )
    
    except Exception:
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
                print(f"[checkpoint] Could not save interrupted checkpoint: {save_err}")
        input("Press Enter to exit...")

if env is not None:
    env.close()

if not _infer_only:
    print(f"\nFinal score after {episode} episodes -> Ally Wins: {ally_wins}, Enemy Wins: {enemy_wins}, Truncated Game: {truncated_game}")
    print(f"Ally Win Rate: {calculate_win_rate(ally_wins, episode)}%")
    print(f"Enemy Win Rate: {calculate_win_rate(enemy_wins, episode)}%")
    print(f"Truncated Rate: {calculate_win_rate(truncated_game, episode)}%")
    print(f"Average Winning Reward per Episode: {average_reward(winning_rewards)}")
    print(f"Average Episode Length: {average_episode_length(episode_lengths)}")
    print(f"Ally Reward Variability: {reward_variability(ally_rewards)}")
    print(f"Enemy Reward Variability: {reward_variability(enemy_rewards)}")
    plot_episode_lengths(episode_lengths, ally="LLM-Agent", enemy="Random")
    plot_episode_rewards(ally_rewards, enemy_rewards, ally="LLM-Agent", enemy="Random")
    if series_ally_win_rate_cumulative:
        plot_grpo_diagnostics(
            series_ally_win_rate_cumulative,
            series_batch_reward_std,
            series_trained_turn_fraction,
        )
