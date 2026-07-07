"""All MuZero-Xiangqi hyperparameters. The only place numbers live."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MuZeroConfig:
    # Engine
    pikafish_bin: str = os.environ.get("PIKAFISH_BIN", "pikafish")
    pikafish_depth: int = 8
    pikafish_movetime_ms: int = 100
    pikafish_timeout_sec: float = 15.0

    # Encoding
    history_length: int = 8
    input_planes: int = 0  # derived: 14 * history_length + 2 (set in __post_init__)
    action_space: int = 8100  # 90 from-squares x 90 to-squares

    # Network
    channels: int = 192
    repr_blocks: int = 12
    dyn_blocks: int = 8
    value_bins: int = 601
    value_max: float = 3.0  # h-transformed units; n-step returns live in ~[-2,2]
    reward_bins: int = 21
    reward_max: float = 2.0
    moves_left_max: int = 200

    # MCTS
    num_simulations: int = 800
    interior_topk: int = 64  # non-root nodes expand top-k prior actions
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    temperature_moves: int = 30  # sample proportional to visits for first N plies

    # Self-play
    num_workers: int = 3
    games_per_worker: int = 28
    # "latest": stock MuZero — the newest network plays both sides, no enemy.
    # "frozen_enemy": learner vs frozen snapshot with streak promotion.
    self_play_mode: str = "latest"
    truncation_symmetric: bool = True  # derived from self_play_mode in __post_init__
    promote_after_consecutive_wins: int = 3
    max_game_plies: int = 300
    # Red first moves in ENGINE UCI (rank 0 = bottom): central/edge cannon,
    # horses, elephants, pawn advances.
    opening_book: tuple = (
        "h2e2",
        "b2e2",
        "h0g2",
        "b0c2",
        "c0e2",
        "g0e2",
        "c3c4",
        "g3g4",
        "a3a4",
        "i3i4",
    )

    # Env rewards / adjudication
    shaping_weight: float = 0.3
    shaping_cp_scale: float = 200.0
    blunder_cp_threshold: float = 200.0  # mover delta <= -this counts as a blunder
    repetition_penalty: float = -0.3
    repetition_cp_ok: float = -100.0  # penalize repeater whose cp >= this
    repetition_swing_cp: float = 50.0  # "no threat" = cp swing below this
    truncation_cp: float = -800.0
    truncation_consecutive: int = 6

    # Replay / training
    # ~18 iterations of self-play at 84 games/loop. The original 5000 (~60
    # iterations) left value targets bootstrapped from a ~28-iteration-stale
    # policy, which showed up as a steadily climbing loss/value.
    buffer_games: int = 1500
    games_per_train_loop: int = 512
    batch_size: int = 512
    unroll_steps: int = 8
    td_steps: int = 10
    discount: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    # (policy, value, reward, moves_left, material, consistency)
    loss_weights: tuple = (1.0, 0.25, 1.0, 0.2, 0.1, 2.0)
    per_alpha: float = 0.6
    truncated_tail_weight: float = 0.3

    # Warm start
    warmstart_plies: int = 2000
    warmstart_movetime_ms: int = 50
    warmstart_multipv: int = 4
    warmstart_train_batches: int = 200

    # Fixed-opponent gate
    gate_every_loops: int = 10
    gate_games: int = 20
    gate_movetime_ms: int = 10

    # Misc
    device: str = "cuda"
    seed: int = 0
    checkpoint_dir: str = "checkpoints/muzero_xiangqi"
    # Every N iterations, also save a permanent ally-weights snapshot to
    # checkpoint_dir/archive/iter_NNNN.pt for the Elo arena. 0 disables.
    checkpoint_archive_every: int = 20
    wandb_project: str = "muzero-xiangqi"

    def __post_init__(self):
        self.input_planes = 14 * self.history_length + 2
        if self.self_play_mode not in ("latest", "frozen_enemy"):
            raise ValueError(f"self_play_mode: {self.self_play_mode!r}")
        self.truncation_symmetric = self.self_play_mode == "latest"
