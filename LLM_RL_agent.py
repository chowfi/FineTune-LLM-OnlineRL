import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gym
import matplotlib.pyplot as plt
import numpy as np
import wandb
from gym.wrappers import RecordVideo
from tqdm import trange
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from gym_xiangqi.constants import ALLY, PIECE_ID_TO_NAME, PIECE_POINTS
from gym_xiangqi.utils import action_space_to_move, move_to_action_space
from gym_xiangqi.agents import RandomAgent
# ─── Evaluation metrics ───

def calculate_win_rate(wins, total_games):
    return (wins / total_games) * 100 if total_games > 0 else 0

def average_reward(rewards):
    return sum(rewards) / len(rewards) if rewards else 0

def average_episode_length(rounds_list):
    return sum(rounds_list) / len(rounds_list) if rounds_list else 0

def reward_variability(rewards):
    return np.std(rewards)

# ─── Evaluation visualizations ───

def plot_episode_lengths(episode_lengths, window_size=3, ally="Greedy", enemy="Random"):
    plt.figure(figsize=(10, 5))
    moving_average = np.convolve(episode_lengths, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size, len(episode_lengths) + 1), moving_average, color='red', label='Moving Average')
    plt.title(f'Episode Lengths Over Games ({ally} vs {enemy})')
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Length')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_episode_rewards(ally_rewards, enemy_rewards, window_size=3, ally="Greedy", enemy="Random"):
    plt.figure(figsize=(10, 5))
    ally_ma = np.convolve(ally_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size, len(ally_rewards) + 1), ally_ma, color='cyan', label=f'{ally} Moving Average')
    enemy_ma = np.convolve(enemy_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size, len(enemy_rewards) + 1), enemy_ma, color='lime', label=f'{enemy} Moving Average')
    plt.title(f'Episode Rewards Over Games ({ally} vs {enemy}) ')
    plt.xlabel('Episode Number')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_grpo_diagnostics(
    ally_win_rate_cumulative,
    batch_reward_std_per_episode,
    trained_turn_fraction,
    window_size=5,
):
    """Cumulative ally win %, GRPO batch reward std (pre-norm), fraction of ally turns with gradients."""
    episodes = np.arange(1, len(ally_win_rate_cumulative) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, ally_win_rate_cumulative, color="tab:blue", label="Cumulative ally win %")
    if len(ally_win_rate_cumulative) >= window_size:
        ma = np.convolve(
            ally_win_rate_cumulative, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0].plot(
            range(window_size, len(ally_win_rate_cumulative) + 1),
            ma,
            color="tab:orange",
            label=f"{window_size}-ep MA",
        )
    axes[0].set_ylabel("Ally win rate (%)")
    axes[0].set_title("Cumulative ally win rate")
    axes[0].legend()
    axes[0].grid(True)

    std_arr = np.asarray(batch_reward_std_per_episode, dtype=float)
    std_masked = np.ma.masked_invalid(std_arr)
    axes[1].plot(episodes, std_masked, color="tab:green", alpha=0.85, label="Mean σ if multi-step ep")
    axes[1].set_ylabel("Batch reward σ (pre-norm)")
    axes[1].set_title("GRPO: std of raw rewards in each update batch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(episodes, trained_turn_fraction, color="tab:purple", label="Trained / ally turns")
    if len(trained_turn_fraction) >= window_size:
        ma_t = np.convolve(
            trained_turn_fraction, np.ones(window_size) / window_size, mode="valid"
        )
        axes[2].plot(
            range(window_size, len(trained_turn_fraction) + 1),
            ma_t,
            color="tab:brown",
            label=f"{window_size}-ep MA",
        )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Non-random (trained) ally turns / total ally turns")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# ─── Log helpers ───

def training_log_token_truncate(file, num_episode, round, input_token_length=2048):
  with open(file, 'a', encoding='utf-8') as f:
    f.write(f">>Episode {num_episode} round {round}, the input token length got to {input_token_length}\n")

def training_log_round_info(file, num_episode, round, turn, piece, move, reward):
  with open(file, 'a', encoding='utf-8') as f:
    f.write(f"Episode {num_episode} round: {round}\n{turn} made the move {piece} from {move[1]} to {move[2]}.\nReward: {reward}\n====================\n")

def training_log_episode_info(file, num_episode, round, ally_wins, enemy_wins, truncated_game):
  with open(file, 'a', encoding='utf-8') as f:
    f.write(f"Episode {num_episode} ended with {round} rounds. Ally Wins: {ally_wins}, Enemy Wins: {enemy_wins}, Truncated Game: {truncated_game}\n")

def training_log_llm_output(file, num_episode, round, llm_output):
  with open(file, 'a', encoding='utf-8') as f:
    f.write(f"Episode {num_episode} round {round}: response: {llm_output}\n")


# ─── FEN-like board representation ───

PIECE_ID_TO_FEN = {
    1: 'K', 2: 'A', 3: 'A', 4: 'E', 5: 'E',
    6: 'H', 7: 'H', 8: 'R', 9: 'R',
    10: 'C', 11: 'C',
    12: 'P', 13: 'P', 14: 'P', 15: 'P', 16: 'P',
}

def board_to_fen(state):
    """Convert 10x9 numpy board to a FEN-like string.
    Uppercase = ally pieces, lowercase = enemy pieces, digits = empty squares.
    Rows separated by '/'.
    """
    rows = []
    for row in state:
        fen_row = ''
        empty = 0
        for cell in row:
            cell = int(cell)
            if cell == 0:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                piece_id = abs(cell)
                char = PIECE_ID_TO_FEN.get(piece_id, '?')
                if cell < 0:
                    char = char.lower()
                fen_row += char
        if empty > 0:
            fen_row += str(empty)
        rows.append(fen_row)
    return '/'.join(rows)


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


# ─── Greedy baseline opponent ───

class GreedyAgent:
    def __init__(self):
        pass

    def move(self, env):
        actions = (env.ally_actions if env.turn == ALLY
                   else env.enemy_actions)
        legal_moves = np.where(actions == 1)[0]

        best_value = 0
        best_move = None

        for move in legal_moves:
            _, _, end = action_space_to_move(move)
            target_piece_id = env.state[end[0]][end[1]]
            piece_value = 0

            if target_piece_id != 0:
                piece_value = PIECE_POINTS[abs(target_piece_id)]

            if piece_value > best_value:
                best_value = piece_value
                best_move = move

        if best_move is None and legal_moves.size > 0:
            best_move = np.random.choice(legal_moves)

        return best_move


# ─── LLM Agent with GRPO ───

class Agent(ABC):
    def __init__(
        self, model, tokenizer, grpo_trainer, max_input_token, device, generate_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
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
    def extract_action(self, response: str) -> gym.core.ActType:
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

        print(f"Round {round} LLM Prompt: {prompt}")

        if (input_token_length + 200) >= self.max_input_token:
          training_log_token_truncate('chinese_chess_token_truncate_log', num_episode=episode, round=round, input_token_length=input_token_length)
          while len(self.current_llm_input) > 2:
              self.current_llm_input.pop(1)
              if (len(self.current_llm_input) > 1
                      and self.current_llm_input[1]["role"] == "assistant"):
                  self.current_llm_input.pop(1)
              prompt = self.tokenizer.apply_chat_template(
                  self.current_llm_input, tokenize=False, add_generation_prompt=True
              )
              check = self.tokenizer(prompt, return_tensors="pt")
              if len(check.input_ids[0]) + 200 < self.max_input_token:
                  break
          inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        response = self.llm(self.current_llm_input)
        print(f"Round {round} LLM Response: {response}")

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
    EARLY_GAME_ALLY_MOVES = 25

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

    def extract_action(self, response: str, env: gym.Env) -> gym.core.ActType:
        match = re.compile(r".*Action:\s*(\d+)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)").search(response)
        if match:
            piece_id = int(match.group(1))
            from_row, from_col, to_row, to_col = int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
            if from_row not in range(0,10):
                from_row = 0
            if to_row not in range(0,10):
                to_row = 0
            if from_col not in range(0,9):
                from_col = 0
            if to_col not in range(0,9):
                to_col = 0

            action = move_to_action_space(piece_id, (from_row, from_col), (to_row, to_col))
            if action in np.where(env.ally_actions == 1)[0]:
              return False, action

        legal_moves = np.where(env.ally_actions == 1)[0]
        return True, np.random.choice(legal_moves)


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

max_input_token = 16384

hyperparams = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "env": "gym_xiangqi:xiangqi-v0",
    "lora/r": 16,
    "lora/lora_alpha": 32,
    "lora/lora_dropout": 0.05,
    "lora/bias": "none",
    "lora/task_type": "CAUSAL_LM",
    "load_in_8bit": True,
    "grpo/batch_size": 8,
    "grpo/lr": 2e-6,
    "grpo/beta": 0.1,
    "seed": 42069,
    "episodes": 500,
    "generate/max_new_tokens": 50,
    "generate/do_sample": True,
    "generate/top_p": 0.6,
    "generate/top_k": 0,
    "generate/temperature": 0.9,
}

wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
HF_TOKEN = os.environ.get("HF_TOKEN")

# ─── Load model ───

load_in_8bit = bool(hyperparams.get("load_in_8bit", False))
quantization_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

model = AutoModelForCausalLM.from_pretrained(
    hyperparams["model_name"],
    quantization_config=quantization_config,
    device_map="auto" if load_in_8bit else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

if not load_in_8bit:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# ─── Apply LoRA ───

lora_config = LoraConfig(
    **{key.split("/")[-1]: value for key, value in hyperparams.items() if key.startswith("lora/")}
)
model = get_peft_model(model, lora_config)

device = next(model.parameters()).device

tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(f'GRPO model parameters to be updated:\n{print_number_of_trainable_model_parameters(model)}\n')

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
env = gym.make(hyperparams["env"])
enemy_agent = RandomAgent()
ally_agent = ChineseChessAgent(
    model,
    tokenizer,
    grpo_trainer,
    max_input_token,
    device,
    {
        key: value
        for key, value in hyperparams.items()
        if key.startswith("generate/")
    },
)

# ─── Training loop ───

ally_wins, enemy_wins, truncated_game = 0, 0, 0
episode_lengths, ally_rewards, enemy_rewards, winning_rewards = [], [], [], []
series_ally_win_rate_cumulative = []
series_batch_reward_std = []
series_trained_turn_fraction = []
training_errors = []
# Random fallback = model output did not match a legal move; env uses random legal move.
lifetime_ally_turns = 0
lifetime_random_fallback = 0
lifetime_act_failures = 0

try:
  for episode in trange(1, hyperparams["episodes"] + 1):
    observation = env.reset()
    round, done = 1, False
    current_ally_rewards, current_enemy_rewards = 0, 0
    episode_ally_turns = 0
    episode_random_fallback = 0
    episode_act_failures = 0

    while not done:
        if env.turn == ALLY:
          used_random_fallback, ally_action, llm_output = ally_agent.act(
              observation, episode, round, env, ally_turn_index=episode_ally_turns
          )
          if ally_action is None:
            episode_act_failures += 1
            lifetime_act_failures += 1
            print(f"WARNING Episode {episode} Round {round}: act() returned None")
          else:
            episode_ally_turns += 1
            if used_random_fallback:
              episode_random_fallback += 1
          _, ally_reward, done, info = env.step(ally_action)
          current_ally_rewards += ally_reward
          ally_agent.assign_reward(ally_reward)

          move = action_space_to_move(ally_action)
          piece = PIECE_ID_TO_NAME[move[0]]
          print(f"Episode: {episode} Round: {round}")
          print(f"Ally made the move {piece} from {move[1]} to {move[2]}. Reward: {ally_reward}. Random {used_random_fallback}")
          print("================")

          if episode % gap_size == 0:
            print(f"\nEpisode {episode}, Round {round}")
            move = action_space_to_move(ally_action)
            piece = PIECE_ID_TO_NAME[move[0]]
            print(f"Ally made the move {piece} from {move[1]} to {move[2]}. Reward: {ally_reward}. Is Ally's move random: {used_random_fallback}")
            print("================")
            training_log_llm_output("chinese_chess_llm_output_log", episode, round, llm_output)

        else:
          enemy_action = enemy_agent.move(env)
          observation, enemy_reward, done, info = env.step(enemy_action)
          current_enemy_rewards += enemy_reward

          move = action_space_to_move(enemy_action)
          piece = PIECE_ID_TO_NAME[move[0]]
          print(f"Episode: {episode} Round: {round}")
          print(f"Enemy made the move {piece} from {move[1]} to {move[2]}. Reward: {enemy_reward} ")
          print("================")

          if episode % gap_size == 0:
            print(f"\nEpisode {episode}, Round {round}")
            move = action_space_to_move(enemy_action)
            piece = PIECE_ID_TO_NAME[move[0]]
            print(f"Enemy made the move {piece} from {move[1]} to {move[2]}. Reward: {enemy_reward}")
            print("================")

        round += 1
        if round >= 200:
            print("Episode reached 200 rounds, stopping to prevent infinite loop.")
            done = True
            truncated_game += 1

    episode_lengths.append(round)
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
        100.0 * lifetime_random_fallback / lifetime_ally_turns if lifetime_ally_turns else 0.0
    )

    trained_turn_fraction = (
        (episode_ally_turns - episode_random_fallback) / episode_ally_turns
        if episode_ally_turns
        else 0.0
    )

    episode_stats = {
            "episode": episode,
            "episode_length": round,
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
        }

    train_stats = ally_agent.terminate_episode()
    episode_stats.update(train_stats)

    series_ally_win_rate_cumulative.append(episode_stats["ally_win_rate"])
    series_trained_turn_fraction.append(trained_turn_fraction)
    br_std_ep = train_stats.get("grpo/batch_reward_std_mean")
    if br_std_ep is None:
        br_std_ep = train_stats.get("grpo/batch_reward_std")
    series_batch_reward_std.append(br_std_ep if br_std_ep is not None else float("nan"))

    wandb.log(episode_stats)

    if episode % gap_size == 0:
      win_rate = calculate_win_rate(ally_wins, episode)
      enemy_win_rate = calculate_win_rate(enemy_wins, episode)
      trunc_rate = calculate_win_rate(truncated_game, episode)
      print(f"\n--- Episode {episode} Summary ---")
      print(f"Ally Wins: {ally_wins} ({win_rate:.1f}%) | Enemy Wins: {enemy_wins} ({enemy_win_rate:.1f}%) | Truncated: {truncated_game} ({trunc_rate:.1f}%)")
      print(f"Random fallback (last ep): {episode_random_fallback}/{episode_ally_turns} ({random_rate_episode:.1f}%) | Lifetime: {random_rate_lifetime:.1f}%")
      print(
          f"Trained-turn fraction: {trained_turn_fraction:.3f} | "
          f"GRPO batch reward σ (pre-norm, mean this ep): "
          f"{train_stats.get('grpo/batch_reward_std_mean', train_stats.get('grpo/batch_reward_std', 'n/a'))}"
      )
      print(f"Buffer size: {ally_agent.grpo_trainer.buffer_size()}")
      print(f"----------------------------")

except Exception as e:
  import traceback
  print("\n" + "=" * 60)
  print("TRAINING CRASHED — traceback below")
  print("=" * 60)
  traceback.print_exc()
  print("=" * 60)
  input("Press Enter to exit...")

env.close()

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
