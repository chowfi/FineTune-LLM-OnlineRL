# MuZero for Xiangqi — Design Spec

**Date:** 2026-07-02
**Status:** Approved by user (design conversation, 2026-07-02)
**Target hardware:** 1× RTX 5090 (32 GB), single machine

## 1. Goal

Build a MuZero-inspired (EfficientZero-style) agent that learns Xiangqi through
self-play on a single GPU. The system is a learned world model + MCTS:
representation, dynamics, and prediction networks searched with pUCT. It is a
new subsystem, independent of the existing LLM/GRPO pipeline; it reuses
`src/pikafish_eval.py` (legality + evaluation) and `gym_xiangqi` (board
mechanics only).

## 2. Confirmed decisions

| Decision | Choice |
|---|---|
| Reward role of Pikafish | **Hybrid**: terminal outcome (+1/0/−1) is the primary value target; Pikafish cp deltas give a small dense per-move shaping reward; cp also drives truncation, anti-repetition penalty, and all metrics |
| "MCTS of 8, 800 sims" | Training unroll depth **K=8**; **800 simulations** per move |
| Parallel games | **3 actor workers × 28 games = 84** concurrent games, batched GPU inference |
| Replay buffer | **5,000 games** (FIFO), whole-game tensorized trajectories, PER priorities |
| Train loop size | Each loop consumes **~512 games' worth** of sampled positions |
| Cold start | **~2,000 plies** of Pikafish-vs-Pikafish games (low movetime) preloaded into the buffer with MultiPV soft policy targets + short supervised phase |
| Opening diversity | First ply sampled from a fixed book of **10 distinct opening moves** |
| Self-play opponent | Ally (latest weights) vs **frozen enemy** snapshot; enemy ← ally weights after **3 consecutive ally wins**; sides alternate each game |
| Legality | **Pikafish only** (`list_legal_moves` via `go perft 1`, cached); gym_xiangqi legality is not used; agent can only ever select a legal move |
| Code location | New `muzero/` package at repo root |

## 3. Package layout

```
muzero/
  config.py         # single dataclass holding every hyperparameter in this spec
  env.py            # Pikafish-legal env wrapper (repetition, truncation, adjudication)
  encoding.py       # board -> tensor, move <-> action-index (8100 flat)
  network.py        # representation / dynamics / prediction nets + aux heads + SimSiam
  mcts.py           # pUCT search, Dirichlet noise, virtual loss, batched leaves
  selfplay.py       # actor workers, GPU inference server, ally/enemy promotion
  replay_buffer.py  # 5000-game FIFO, PER sampling, K=8 unroll batches
  train.py          # entrypoint: loop orchestration + combined loss
  warmstart.py      # Pikafish game generation + supervised pretrain
  metrics.py        # wandb + CSV metric aggregation
  tests/            # unit + smoke tests (see §10)
```

Each module has one purpose and a narrow interface; `config.py` is the only
place numbers live.

## 4. Environment wrapper (`env.py`)

- Wraps `gym_xiangqi` for board state transitions only.
- **Legality:** every turn, legal moves come exclusively from
  `PikafishEvaluator.list_legal_moves(fen)`. MCTS root priors and the final
  move selection are masked to this set.
- **Repetition-with-no-threat → draw:** position-hash counting detects 3-fold
  repetition; "no threat" means neither side's Pikafish eval swings materially
  across the repeat. Adjudicated as a **draw**; a **−0.3 reward penalty** is
  applied to the repeating side whose eval was ≥ −100 cp (it repeated while not
  losing). This both ends the game (env-enforced draw) and punishes the agent.
- **Hopeless-game truncation (asymmetric):** if the learning agent's Pikafish
  eval ≤ **−800 cp for 6 consecutive own turns**, truncate the episode and
  score it as a loss (value target −1). Tail positions of truncated games are
  down-weighted in buffer sampling so hopeless grinding does not erode learned
  gains. Winning saturated positions always play out (practice converting
  endgames).

## 5. Tensor encoding (`encoding.py`)

- **Input:** `115 × 10 × 9` float tensor.
  - 14 binary piece planes (7 piece types × 2 colors) × last **8** positions = 112.
  - 3 broadcast planes: side-to-move, repetition count, no-progress (plies
    since last capture).
- **Action space:** flat **90 × 90 = 8,100** (from-square × to-square), always
  masked to Pikafish-legal moves. Bidirectional `move_to_index` /
  `index_to_move` (UCI ↔ index) helpers.
- All buffer storage, batching, and network I/O are tensors end to end.

## 6. Networks (`network.py`), sized for one 5090

- **Representation:** conv stem + 12 residual blocks × 192 channels →
  `192 × 10 × 9` latent.
- **Dynamics:** latent + action plane → 8 residual blocks → next latent +
  reward head.
- **Prediction heads:** policy (8,100 logits), value (categorical, 601 bins over [−300, 300] with MuZero invertible scaling),
  **moves-left** (categorical over remaining plies, capped at 200), **material-balance**
  (regression, Red-minus-Black piece values computed from the board — no
  engine call), and a **SimSiam projector/predictor** for the consistency loss.
- ~25–35 M parameters; bf16; inference batch up to 512.

## 7. MCTS + self-play (`mcts.py`, `selfplay.py`)

- 800 simulations/move, pUCT, Dirichlet root noise, temperature decay with
  move number, virtual loss for batched leaf evaluation.
- 3 actor processes × 28 games submit leaf batches to a shared GPU inference
  server (batching queue), interleaved with training on the same device.
- Search module is isolated so **Gumbel MuZero** (32–64 sims) can be swapped in
  later if throughput is insufficient.
- **Ally/enemy:** ally uses latest weights; enemy is a frozen snapshot. After
  3 consecutive ally wins, enemy ← ally. Sides alternate so the ally trains as
  both Red and Black.
- **Opening book:** first ply drawn from 10 fixed distinct openings (central
  cannon, elephant, horse, pawn advances, etc.).

## 8. Replay buffer + training (`replay_buffer.py`, `train.py`)

- Buffer: 5,000 whole games, FIFO eviction, per-position PER priorities
  (|value target − predicted value|). Truncated-game tails down-weighted.
- Train loop: sample `512 × (current mean game length in plies)` positions per
  loop, in batches of 512, unrolling **K=8** dynamics steps per sample.
- **Loss:**
  `L = λ_p·policy CE + λ_v·value CE + λ_r·reward CE + λ_m·moves-left CE + λ_b·material MSE + λ_c·SimSiam consistency`
  with defaults **λ = (1, 0.25, 1, 0.2, 0.1, 2)** (consistency weight per
  EfficientZero).
- **Value targets:** n-step bootstrap from MCTS root values + hybrid rewards
  (shaped Pikafish cp deltas, tanh-squashed) + terminal outcome.

## 9. Warm start (`warmstart.py`)

Before self-play: generate ~2,000 plies of Pikafish-vs-Pikafish games at low
movetime, with MultiPV-derived soft policy targets; load into the buffer and
run a short supervised phase so early MCTS iterations are not random.

## 10. Metrics (wandb, `metrics.py`)

**Requested:** win/draw/loss rates (per enemy era + rolling), **Pikafish points
won** (mean final cp, mean cp-AUC per game — engine-based, not gym rewards),
draw-by-repetition rate, truncation rate.

**Additional:** enemy-promotion count + games-per-promotion, mean game length,
MCTS root value vs Pikafish eval correlation (value-head honesty), root policy
entropy, fraction of moves matching Pikafish best, per-loss-component curves,
buffer age of sampled positions, GPU inference batch utilization, and a
periodic **fixed-opponent gate** (20 games vs Pikafish at fixed weak movetime
every 10 train loops, configurable) as an absolute strength anchor.

## 11. Testing

- Unit: FEN↔tensor and move↔index round-trips; legality mask equals Pikafish's
  legal set; repetition and truncation adjudication cases.
- Smoke: 2 tiny games end to end (8 sims, tiny net) → buffer → one train step.

## 12. Alternatives considered

- **Adapt `muzero-general`/EfficientZero repos:** rejected — Atari-centric,
  Ray-based, weakly maintained; adapting to Xiangqi + Pikafish legality +
  custom self-play rules costs more than building directly.
- **Gumbel MuZero from the start:** deferred — kept as a drop-in option behind
  the isolated search interface if 800 sims proves too slow on one GPU.
