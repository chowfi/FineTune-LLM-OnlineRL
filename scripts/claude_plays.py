"""
Interactive Xiangqi: Claude (ally) vs GreedyAgent (enemy).
State persisted between calls by replaying action history.

Usage:
  python claude_plays.py new [N]     # start game N (default 1)
  python claude_plays.py             # show current board + valid moves
  python claude_plays.py <idx>       # play move at index from shown list
"""

import os
import sys
import json
import warnings

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
warnings.filterwarnings("ignore")

import gym
import numpy as np
from gym_xiangqi.utils import action_space_to_move

STATE_FILE = "/tmp/xiangqi_state.json"
MAX_ROUNDS = 300

# ── piece helpers ───────────────────────────────────────────────────────────

PIECE_SYMBOL = {
    1: "G",
    2: "A",
    3: "A",
    4: "E",
    5: "E",
    6: "H",
    7: "H",
    8: "R",
    9: "R",
    10: "C",
    11: "C",
    12: "S",
    13: "S",
    14: "S",
    15: "S",
    16: "S",
}
PIECE_NAME = {
    1: "General",
    2: "Advisor",
    3: "Advisor",
    4: "Elephant",
    5: "Elephant",
    6: "Horse",
    7: "Horse",
    8: "Chariot",
    9: "Chariot",
    10: "Cannon",
    11: "Cannon",
    12: "Soldier",
    13: "Soldier",
    14: "Soldier",
    15: "Soldier",
    16: "Soldier",
}
PIECE_VALUE = {
    1: 100,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 4,
    7: 4,
    8: 9,
    9: 9,
    10: 4,
    11: 4,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
}


def sym(cell):
    if cell == 0:
        return " . "
    pid = abs(cell)
    s = PIECE_SYMBOL.get(pid, "?")
    return f" {s} " if cell > 0 else f"-{s}-"  # ally=space-padded, enemy=dashes


def col_letter(c):
    return "abcdefghi"[c]


# ── board display ───────────────────────────────────────────────────────────


def display_board(state):
    print()
    print("     a   b   c   d   e   f   g   h   i")
    print("   +" + "---+" * 9)
    for r in range(10):
        row_str = f" {r} |"
        for c in range(9):
            row_str += sym(state[r][c]) + "|"
        if r == 4:
            row_str += "  ── river ──"
        print(row_str)
        print("   +" + "---+" * 9)
    print()
    print("  ally (Claude) = lowercase  |  enemy (Greedy) = -X-")
    print()


# ── greedy enemy agent ──────────────────────────────────────────────────────


def greedy_enemy_move(uw):
    """Enemy is negative pieces; look for highest-value ally piece to capture."""
    valid = np.nonzero(uw.enemy_actions)[0]
    if len(valid) == 0:
        return None
    best, best_val = None, -1
    for a in valid:
        _, _, to_pos = action_space_to_move(a)
        target = uw.state[to_pos[0]][to_pos[1]]
        if target > 0:  # ally piece
            v = PIECE_VALUE.get(target, 0)
            if v > best_val:
                best_val, best = v, a
    return int(best if best is not None else np.random.choice(valid))


# ── env helpers ─────────────────────────────────────────────────────────────


def make_env():
    env = gym.make("gym_xiangqi:xiangqi-v0")
    env.reset()
    return env


def replay(actions):
    env = make_env()
    total_reward = 0.0
    done = False
    for a in actions:
        if done:
            break
        _, r, done, _ = env.step(a)
        total_reward += r
    return env, total_reward, done


# ── move list display ────────────────────────────────────────────────────────


def list_ally_moves(uw):
    valid = np.nonzero(uw.ally_actions)[0]
    moves = []
    for a in valid:
        pid, from_pos, to_pos = action_space_to_move(a)
        target = uw.state[to_pos[0]][to_pos[1]]
        cap = ""
        if target < 0:
            cap = f"  ★ captures enemy {PIECE_NAME.get(abs(target), '?')} (val={PIECE_VALUE.get(abs(target), 0)})"
        moves.append((int(a), pid, from_pos, to_pos, cap))
    return moves


def display_moves(moves, uw):
    print(f"  {'#':>3}  {'Piece':<10} {'From':>4} → {'To':<4}  Notes")
    print("  " + "-" * 60)
    for i, (a, pid, fp, tp, cap) in enumerate(moves):
        pname = PIECE_NAME.get(abs(pid), f"#{pid}")
        fr = f"{fp[0]}{col_letter(fp[1])}"
        to = f"{tp[0]}{col_letter(tp[1])}"
        print(f"  {i:>3}  {pname:<10} {fr:>4} → {to:<4} {cap}")
    print()


# ── state persistence ────────────────────────────────────────────────────────


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return None


def save_state(d):
    with open(STATE_FILE, "w") as f:
        json.dump(d, f)


# ── main ─────────────────────────────────────────────────────────────────────


def show_position(st):
    actions = st["actions"]
    env, total_reward, done = replay(actions)
    uw = env.unwrapped

    game_num = st["game_num"]
    turn_num = (
        sum(1 for i, a in enumerate(actions) if i % 2 == 0) + 1
    )  # rough ally turn count

    print(f"\n{'=' * 60}")
    print(
        f"  Game {game_num}  |  Move {len(actions) + 1}  |  Net reward so far: {total_reward:+.0f}"
    )
    print(f"{'=' * 60}")
    display_board(uw.state)

    if done:
        if total_reward > 0:
            result = "Claude wins!"
        elif total_reward < 0:
            result = "Greedy wins."
        else:
            result = "Draw."
        print(f"  GAME OVER: {result}  (total reward {total_reward:+.0f})")
        st["done"] = True
        st["result"] = result
        save_state(st)
        env.close()
        return

    moves = list_ally_moves(uw)
    if not moves:
        print("  No valid moves — Claude loses.")
        st["done"] = True
        st["result"] = "No moves — Claude loses."
        save_state(st)
        env.close()
        return

    print(f"  Your valid moves ({len(moves)} options):")
    display_moves(moves, uw)
    print("  Run:  python claude_plays.py <idx>   to play a move")
    env.close()


def play_move(st, idx):
    actions = st["actions"]
    env, total_reward, done = replay(actions)
    if done:
        print("Game already over.")
        env.close()
        return

    uw = env.unwrapped
    moves = list_ally_moves(uw)
    if idx < 0 or idx >= len(moves):
        print(f"  Invalid index {idx}. Choose 0–{len(moves) - 1}.")
        env.close()
        return

    chosen_action, pid, fp, tp, cap = moves[idx]
    pname = PIECE_NAME.get(abs(pid), f"#{pid}")
    fr = f"{fp[0]}{col_letter(fp[1])}"
    to_ = f"{tp[0]}{col_letter(tp[1])}"
    print(f"\n  Claude plays: {pname} {fr} → {to_}{cap}")

    _, r, done, _ = env.step(chosen_action)
    total_reward += r
    actions.append(chosen_action)

    if not done:
        uw2 = env.unwrapped
        enemy_action = greedy_enemy_move(uw2)
        if enemy_action is not None:
            _, pid_e, fp_e, tp_e, _ = (
                list(
                    filter(
                        lambda m: m[0] == enemy_action,
                        [
                            (int(a), *action_space_to_move(a), "")
                            for a in np.nonzero(uw2.enemy_actions)[0]
                        ],
                    )
                )[0]
                if False
                else (enemy_action, *action_space_to_move(enemy_action), "")
            )
            pname_e = PIECE_NAME.get(abs(pid_e), f"#{pid_e}")
            fr_e = f"{fp_e[0]}{col_letter(fp_e[1])}"
            to_e = f"{tp_e[0]}{col_letter(tp_e[1])}"

            target_e = uw2.state[tp_e[0]][tp_e[1]]
            cap_e = (
                f"  ★ captures your {PIECE_NAME.get(target_e, '?')}"
                if target_e > 0
                else ""
            )
            print(f"  Greedy plays: {pname_e} {fr_e} → {to_e}{cap_e}")

            _, r2, done, _ = env.step(enemy_action)
            total_reward += r2
            actions.append(enemy_action)

    env.close()
    st["actions"] = actions
    save_state(st)

    # Now show the new position
    env2, _, done2 = replay(actions)
    uw3 = env2.unwrapped
    print(f"\n{'=' * 60}")
    print(
        f"  Game {st['game_num']}  |  Move {len(actions) + 1}  |  Net reward: {total_reward:+.0f}"
    )
    print(f"{'=' * 60}")
    display_board(uw3.state)

    if done or done2:
        if total_reward > 0:
            result = "Claude wins!"
        elif total_reward < 0:
            result = "Greedy wins."
        else:
            result = "Draw."
        print(f"  GAME OVER: {result}")
        st["done"] = True
        st["result"] = result
        save_state(st)
        env2.close()
        return

    moves2 = list_ally_moves(uw3)
    print(f"  Your valid moves ({len(moves2)} options):")
    display_moves(moves2, uw3)
    print("  Run:  python claude_plays.py <idx>   to play a move")
    env2.close()


def main():
    args = sys.argv[1:]

    if args and args[0] == "new":
        game_num = int(args[1]) if len(args) > 1 else 1
        st = {"game_num": game_num, "actions": [], "done": False, "result": None}
        save_state(st)
        print(f"\n  Starting Game {game_num}: Claude vs Greedy")
        show_position(st)
        return

    st = load_state()
    if st is None:
        print("  No game in progress. Run:  python claude_plays.py new")
        return

    if not args:
        show_position(st)
        return

    try:
        idx = int(args[0])
    except ValueError:
        print(f"  Unknown argument: {args[0]}")
        return

    play_move(st, idx)


if __name__ == "__main__":
    main()
