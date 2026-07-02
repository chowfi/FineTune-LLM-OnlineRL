"""
Play 5 Xiangqi games: Claude (Greedy) as ally vs Random enemy.
Runs headless (no display window).
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
from gym_xiangqi.agents import RandomAgent
from gym_xiangqi.constants import ALLY, ENEMY
from gym_xiangqi.utils import action_space_to_move

# Piece-value table (abs of state cell value → point worth)
PIECE_POINTS = {1: 100, 2: 4, 3: 4, 4: 2, 5: 2, 6: 9, 7: 9, 8: 4, 9: 4,
                10: 4, 11: 4, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1}

MAX_ROUNDS = 500


class GreedyAgent:
    """Captures the highest-value piece reachable this turn; else moves randomly."""

    def move(self, env):
        uw = env.unwrapped
        valid = np.nonzero(uw.ally_actions if uw.turn == ALLY else uw.enemy_actions)[0]
        if len(valid) == 0:
            return env.action_space.sample()

        best_action = None
        best_value = -1

        for a in valid:
            _, from_pos, to_pos = action_space_to_move(a)
            row, col = to_pos
            target = uw.state[row][col]
            # Enemy pieces are negative on the board from ally's perspective
            if target < 0:
                value = PIECE_POINTS.get(abs(target), 0)
                if value > best_value:
                    best_value = value
                    best_action = a

        if best_action is None:
            best_action = int(np.random.choice(valid))

        return int(best_action)


def play_game(game_num, greedy, random_agent):
    env = gym.make("gym_xiangqi:xiangqi-v0")
    env.reset()
    done = False
    rounds = 0
    total_reward = 0.0
    result = "truncated"

    while not done and rounds < MAX_ROUNDS:
        uw = env.unwrapped
        if uw.turn == ALLY:
            action = greedy.move(env)
        else:
            action = random_agent.move(env)

        _, reward, done, _ = env.step(action)
        total_reward += reward
        rounds += 1

    if done:
        # Positive cumulative reward = ally (greedy) won more material / got the general
        if total_reward > 0:
            result = "ally (Claude/Greedy) wins"
        elif total_reward < 0:
            result = "enemy (Random) wins"
        else:
            result = "draw"

    env.close()
    return rounds, total_reward, result


def main():
    greedy = GreedyAgent()
    random_agent = RandomAgent()

    print("=" * 55)
    print("  5 Xiangqi Games: Claude (Greedy) vs Random Agent")
    print("=" * 55)

    results = []
    for i in range(1, 6):
        print(f"\nGame {i}: running...", end=" ", flush=True)
        rounds, reward, result = play_game(i, greedy, random_agent)
        results.append(result)
        print(f"done ({rounds} rounds)")
        print(f"  Net reward : {reward:+.1f}")
        print(f"  Result     : {result}")

    print("\n" + "=" * 55)
    ally_wins = sum(1 for r in results if "ally" in r)
    enemy_wins = sum(1 for r in results if "enemy" in r)
    trunc = sum(1 for r in results if "truncated" in r)
    print(f"  Final record: {ally_wins}W / {enemy_wins}L / {trunc} truncated")
    print("=" * 55)


if __name__ == "__main__":
    main()
