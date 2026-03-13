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

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)

# functions for all the evaluation metrics

def calculate_win_rate(wins, total_games):
    return (wins / total_games) * 100 if total_games > 0 else 0

def average_reward(rewards):
    """
    Average reward of winning agent
    """
    return sum(rewards) / len(rewards) if rewards else 0

def average_episode_length(rounds_list):
    return sum(rounds_list) / len(rounds_list) if rounds_list else 0

def reward_variability(rewards):
    return np.std(rewards)

# functions for evaluation visualizations

def plot_episode_lengths(episode_lengths, window_size=3, ally="Greedy", enemy="Random"):
    """
    Plots the lengths of each episode and includes a moving average line to highlight trends.

    """
    plt.figure(figsize=(10, 5))
    episodes = range(1, len(episode_lengths) + 1)

    moving_average = np.convolve(episode_lengths, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size, len(episode_lengths) + 1), moving_average, color='red', label='Moving Average')

    plt.title(f'Episode Lengths Over Games ({ally} vs {enemy})')
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Length')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_episode_rewards(ally_rewards, enemy_rewards, window_size=3, ally="Greedy", enemy="Random"):
    """
    Plots the cumulative rewards for ally and enemy agents over episodes, including moving averages.

    Parameters:
    - ally_rewards (list): A list of floats representing the cumulative rewards for each episode by the ally agent.
    - enemy_rewards (list): A list of floats representing the cumulative rewards for each episode by the enemy agent.
    - window_size (int): The size of the window used for the moving average calculation.
    """
    plt.figure(figsize=(10, 5))

    episodes = range(1, len(enemy_rewards) + 1)


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


# Some log functions that will help save intermediate results
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

class Agent(ABC):
    def __init__(
        self, model, tokenizer, max_input_token, device, generate_config_dict=None, ppo_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if ppo_config_dict is None:
            ppo_config_dict = {"batch_size": 16, "mini_batch_size": 16}

        self.model = model
        self.tokenizer = tokenizer
        self.max_input_token = max_input_token
        self.device = device
        self.generate_config_dict = generate_config_dict
        self.model_ref = create_reference_model(model)
        self.ppo_config = PPOConfig(**ppo_config_dict)
        self.ppo_trainer = PPOTrainer(self.ppo_config, model, self.model_ref, tokenizer)

        self.current_batch = {"queries": [], "responses": [], "rewards": []}

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_llm_input = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(self, observation: gym.core.ObsType) -> str:
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
        generate_ids = self.model.generate(
            inputs=inputs.input_ids,
            **{
                key.split("/")[-1]: value
                for key, value in self.generate_config_dict.items()
            }
        )
        generate_ids = generate_ids[:, context_len:]
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = outputs[0]

        return response

    def act(self, observation, episode, round, env):
        message = self.format_observation(observation)
        self.current_episode_messages += [{"role": "user", "content": message}]
        self.current_llm_input+=[{"role": "user", "content": message}]

        # Check if the prompt exceeds the maximum token length
        prompt = self.tokenizer.apply_chat_template(
            self.current_llm_input, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_length=len(inputs.input_ids[0])

        # Log the prompt for this round
        print(f"Round {round} LLM Prompt: {prompt}") #toggle this for print out of system prompt

        # Truncate if the length is longer than 2048-200 (give some buffer)
        if (input_token_length+200)>=self.max_input_token:
          # if it exceeds, reset self.current_llm_input
          training_log_token_truncate('chinese_chess_token_truncate_log', num_episode=episode, round=round, input_token_length=input_token_length)
          self.current_llm_input=[
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
          ]
          self.current_llm_input += [{"role": "user", "content": message}]


        response = self.llm(self.current_llm_input)

        # Log the response for this round
        print(f"Round {round} LLM Response: {response}") #toggle this for print out of llm response

        try:
            random, action = self.extract_action(response, env)
        except Exception as e:
            return None, None, response

        self.current_episode_messages += [{"role": "assistant", "content": response}]
        self.current_llm_input += [{"role": "assistant", "content": response}]
        return random, action, response

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_ppo(self, messages, rewards):
        queries, responses = [], []
        for i in range(2, len(messages), 2):
            prompt = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            conversation_chunks = prompt.split("</s>")[:-1]
            query = "</s>".join(conversation_chunks[:-1]) + "</s>"
            response = conversation_chunks[-1]

            # if len(query)>4500:
            #   query=query[:4500]

            query = self.tokenizer(query, return_tensors="pt").input_ids[0]
            response = self.tokenizer(response, return_tensors="pt").input_ids[0]

            # if torch.all(query >= 0) and torch.all(query < 50265):
            #   print(" ")
            # else:
            #   print("Invalid indices detected in query.")

            # if torch.all(response >= 0) and torch.all(response < 50265):
            #   print(" ")
            # else:
            #   print("Invalid indices detected in response.")
            query = torch.clamp(query, 0, 50265 - 1)
            response = torch.clamp(response, 0, 50265 - 1)

            queries.append(query)
            responses.append(response)

        if all(reward == 0 for reward in rewards[:-1]):
            # if sparse rewards, give equal reward to all conversation turns
            per_turn_reward = rewards[-1] / (len(messages) / 2)
            rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(
                queries
            )
        else:
            rewards = [torch.tensor(reward, dtype=torch.float16) for reward in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_ppo(
                self.current_episode_messages, self.current_episode_rewards
            )

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_llm_input = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []

        if train:
            self.current_batch["queries"].extend(queries)
            self.current_batch["responses"].extend(responses)
            self.current_batch["rewards"].extend(rewards)

            if len(self.current_batch["queries"]) >= self.ppo_config.batch_size:
                train_stats = self.train_batch(
                    self.current_batch["queries"],
                    self.current_batch["responses"],
                    self.current_batch["rewards"],
                )
                return train_stats

        return {}

    def train_batch(self, batch_queries, batch_responses, batch_rewards):
        if len(batch_queries) > self.ppo_config.batch_size:
            queries = batch_queries[: self.ppo_config.batch_size]
            responses = batch_responses[: self.ppo_config.batch_size]
            rewards = batch_rewards[: self.ppo_config.batch_size]

            # keep the remainder for the next batch
            self.current_batch["queries"] = batch_queries[self.ppo_config.batch_size :]
            self.current_batch["responses"] = batch_responses[
                self.ppo_config.batch_size :
            ]
            self.current_batch["rewards"] = batch_rewards[self.ppo_config.batch_size :]
        else:
            queries, responses, rewards = batch_queries, batch_responses, batch_rewards
            self.current_batch = {"queries": [], "responses": [], "rewards": []}

        train_stats = self.ppo_trainer.step(queries, responses, rewards)
        # print(train_stats)
        # print(f'LOG_STATS:{self.ppo_trainer.log_stats(train_stats, self.current_batch, rewards)}')

        torch.cuda.empty_cache()

        return train_stats

class ChineseChessAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert Chinese chess player. Your goal is to take your opponent's general piece. Every turn, you'll see the current layout on the board demonstrated by a 10 by 9 numpy array. Each coordinate in the numpy array corresponds to a single coordinate on the board with the value range from -16 to 16, which represents the pieces. Negative integers are enemy pieces, positive integers are your pieces, and 0 means it's empty. Decide what you want your move to be by writing "Action: [Piece_id, (Start_x Start_y), (End_x, End_y)]". Piece_id should be an integer from 1 to 16 which indicates the piece that you want to move. Start should be a tuple with two integers, where the first integer should be from 0 to 9 and the second integer should be from 0 to 8. Start tuple represents where Piece_id is at. End should be a tuple with two integers, where the first integer should be from 0 to 9 and the second integer should be from 0 to 8. End tuple represents where you want to move Piece_id to."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        message = f"The current board looks like this: {observation}."
        return message

    def extract_action(self, response: str, env: gym.Env) -> gym.core.ActType:
        # match = re.compile(r"Action:\s*\[\s*(\d+)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)\s*,\s*\((\d+)\s*,\s*(\d+)\s*\)\s*\]").search(response)
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

# Settings and config for the LLM before we start the training
max_input_token=2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf", #"Dream-org/Dream-v0-Instruct-7B", #"facebook/opt-125m", #"facebook/opt-125m", "meta-llama/Llama-2-7b-chat-hf"
        "env": "gym_xiangqi:xiangqi-v0",
        "lora/r": 16, #changed to 16 from 32 because gpu spikes
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": False,
        "batch_size": 1,
        "seed": 42069,
        "episodes": 25,
        "generate/max_new_tokens": 50,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
    }

wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
device = "cuda:0"
# device='cpu'
HF_TOKEN = os.environ.get("HF_TOKEN")

lora_config = LoraConfig(
    **{key.split("/")[-1]: value for key, value in hyperparams.items() if key.startswith("lora/")}
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    pretrained_model_name_or_path=hyperparams["model_name"],
    peft_config=lora_config,
    load_in_8bit=hyperparams["load_in_8bit"]
).to(device)

tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])
tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.pretrained_model.resize_token_embeddings(len(tokenizer))

# Save initial parameters for comparison later
initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(f'PPO model parameters to be updated:\n{print_number_of_trainable_model_parameters(model)}\n')

gap_size=10

# Initialize Environment and Agents
env = gym.make(hyperparams["env"])
enemy_agent = GreedyAgent()
ally_agent = ChineseChessAgent(
    model,
    tokenizer,
    max_input_token,
    device,
    {
        key: value
        for key, value in hyperparams.items()
        if key.startswith("generate/")
    },
    {
        "batch_size": hyperparams["batch_size"],
        "mini_batch_size": hyperparams["batch_size"],
    },
)

# Variables to track statistics
ally_wins, enemy_wins, truncated_game = 0, 0, 0
episode_lengths, ally_rewards, enemy_rewards, winning_rewards = [], [], [], []
training_errors = []


for episode in trange(1, hyperparams["episodes"]+1):
    observation = env.reset()
    round, done = 1, False
    current_ally_rewards, current_enemy_rewards = 0, 0

    while not done:
        if env.turn==ALLY:
          random, ally_action, llm_output = ally_agent.act(observation, episode, round, env)
          _, ally_reward, done, info = env.step(ally_action)
          current_ally_rewards += ally_reward
          ally_agent.assign_reward(ally_reward)

          # Logging move details
          move = action_space_to_move(ally_action)
          piece = PIECE_ID_TO_NAME[move[0]]
          print(f"Episode: {episode} Round: {round}")
          print(f"Ally made the move {piece} from {move[1]} to {move[2]}. Reward: {ally_reward}. Random {random}")
          print("================")

          if episode%gap_size==0:
            print(f"\nEpisode {episode}, Round {round}")
            move = action_space_to_move(ally_action)
            piece = PIECE_ID_TO_NAME[move[0]]
            print(f"Ally made the move {piece} from {move[1]} to {move[2]}. Reward: {ally_reward}. Is Ally's move random: {random}")
            print("================")
            training_log_llm_output("chinese_chess_llm_output_log", episode, round, llm_output)

        else:
          enemy_action = enemy_agent.move(env)
          observation, enemy_reward, done, info = env.step(enemy_action)
          current_enemy_rewards += enemy_reward

          # Logging move details
          move = action_space_to_move(enemy_action)
          piece = PIECE_ID_TO_NAME[move[0]]
          print(f"Episode: {episode} Round: {round}")
          print(f"Enemy made the move {piece} from {move[1]} to {move[2]}. Reward: {enemy_reward} ")
          print("================")

          if episode%gap_size==0:
            print(f"\nEpisode {episode}, Round {round}")
            move = action_space_to_move(enemy_action)
            piece = PIECE_ID_TO_NAME[move[0]]
            print(f"Enemy made the move {piece} from {move[1]} to {move[2]}. Reward: {enemy_reward}")
            print("================")

        round+=1
        if round >= 200:
            print("Episode reached 200 rounds, stopping to prevent infinite loop.")
            done = True
            truncated_game += 1

    episode_lengths.append(round)
    if enemy_reward==100:
      enemy_wins += 1
      winning_rewards.append(current_enemy_rewards)
    elif ally_reward==100:
      ally_wins += 1
      winning_rewards.append(current_ally_rewards)

    ally_rewards.append(current_ally_rewards)
    enemy_rewards.append(current_enemy_rewards)

    episode_stats = {
            "episode": episode,
            "total_return": current_ally_rewards + current_enemy_rewards,
            "ally_return": current_ally_rewards,
            "enemy_return": current_enemy_rewards,
            "message_ct": len(ally_agent.current_episode_messages),
            "episode_messages": ally_agent.current_episode_messages,
            # "loss":
        }

    train_stats = ally_agent.terminate_episode()
    episode_stats.update(train_stats)
    wandb.log(episode_stats)

    if episode%gap_size==0:
      print(f"Episode {episode} ended. Ally Wins: {ally_wins}, Enemy Wins: {enemy_wins}, Truncated Game: {truncated_game} ")
env.close()

print(f"\nFinal score after {episode} episodes -> Ally Wins: {ally_wins}, Enemy Wins: {enemy_wins}, Truncated Game: {truncated_game}")
print(f"Ally Win Rate: {calculate_win_rate(ally_wins, episode)}%")
print(f"Enemy Win Rate: {calculate_win_rate(enemy_wins, episode)}%")
print(f"Truncated Rate: {calculate_win_rate(truncated_game, episode)}%")
print(f"Average Winning Reward per Episode: {average_reward(winning_rewards)}")
print(f"Average Episode Length: {average_episode_length(episode_lengths)}")
print(f"Ally Reward Variability: {reward_variability(ally_rewards)}")
print(f"Enemy Reward Variability: {reward_variability(enemy_rewards)}")
plot_episode_lengths(episode_lengths, ally="LLM-Agent", enemy="Greedy")
plot_episode_rewards(ally_rewards, enemy_rewards, ally="LLM-Agent", enemy="Greedy")