# Experiment Log: Greedy on-policy action selection for the legal-move sampler

**Date:** 2026-05-14  
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

The legal-move sampler in `LLM_RL_agent_FSDP_v2.py` builds a 32-move GRPO group by sampling **without replacement** from the policy's distribution over legal moves, then plays `choice_pool[0]` — the **first weighted draw**. That is statistically equivalent to a single temperature-1 sample of the policy, which is noisy: the agent regularly plays moves it itself considers low-probability, even when a clearly-better legal move was visible in the group.

**Hypothesis:** Replacing the action-selection rule with **argmax over the per-legal-move policy log-probs** (greedy on-policy) keeps training on-policy (no Pikafish oracle), matches inference-time greedy decoding, and improves expected env return per turn without breaking GRPO. The GRPO group is unchanged — only which member of the group gets played in the env.

## 2. Configuration Changes

### Code (`LLM_RL_agent_FSDP_v2.py`)

- New hyperparam in the `hyperparams` dict:
  - `"grpo/legal_move_action_selection": "greedy"` (default).
  - Allowed values: `"greedy"` | `"first_sample"`.
- New `XiangqiAgent.__init__` arg `legal_move_action_selection: str = "greedy"`; validated (raises `ValueError` on other values).
- `_generate_policy_sampled_legal_candidates` now, when greedy mode is active and `score_arr` is finite:
  - Computes `greedy_idx = argmax(score_arr)` over **all** legal moves (not just the sampled ones).
  - If `greedy_idx` is already in `selected_idx`, it is moved to position `0`.
  - If not, the lowest-policy-prob slot is dropped and `greedy_idx` is inserted at position `0` (preserves group budget `k`).
  - GRPO group composition is otherwise unchanged.
- Stdout sampler line now appends `played_greedy=<uci>@p=<float>` when greedy mode is on, so logs are self-documenting.
- Per-ally-turn **per-legal-move policy log-prob block** in `output.log` (sorted desc):
  - Header: `[Ep .. Rd ..] Legal-move policy log-prob sums (response tokens only, n=<N>):`
  - One row per legal move: `<uci>\t<sum log p>`. Highest sum = most probable `Move: <uci>` under the current policy.
- **Engine-best agreement tracking** (new for the 2026-05-15 follow-up run):
  - One cached call per FEN to `pikafish_evaluator.bestmove_root_cached(fen)` to get Pikafish's overall best legal move regardless of the 32 sampled.
  - New per-turn metrics surfaced in `candidate_metrics` (and to W&B):
    - `game/engine_best_known` — 1.0 if Pikafish returned a `bestmove`.
    - `game/engine_best_in_group` — 1.0 if Pikafish's best overall is among the 32 distinct legal moves.
    - `game/chosen_is_engine_argmax_in_group` — 1.0 if the played move == argmax-`engine_reward` move inside the 32.
    - `game/chosen_is_engine_best_overall` — 1.0 if played move == Pikafish best overall.
    - `game/chosen_engine_rank_in_group` — 1 = best-in-group, len(group) = worst.
    - `game/chosen_minus_argmax_cp_delta` — chosen cp_delta minus the group-argmax cp_delta (negative = blunder vs best-in-group).
  - Episode aggregates (W&B):
    - `game/engine_best_known_rate`, `game/engine_best_in_group_rate`,
      `game/chosen_is_engine_argmax_in_group_rate`,
      `game/chosen_is_engine_best_overall_rate` (all in **%**).
    - `game/mean_chosen_engine_rank_in_group`,
      `game/median_chosen_engine_rank_in_group`,
      `game/mean_chosen_minus_argmax_cp_delta`.
  - New stdout line per ally turn in the board-sync block:
    `Engine-best comparison: engine_best_overall=<uci> in_group=<0|1> argmax_in_group=<uci> argmax_engine_reward=… argmax_cp_delta=… chosen=<uci> chosen_is_argmax_in_group=<0|1> chosen_is_engine_best_overall=<0|1> chosen_rank_in_group=<int> chosen_minus_argmax_cp_delta=…`

### Defaults rationale
- `"greedy"` is the natural deployment-time decoding strategy (temperature=0 / argmax). Training under the same rule keeps train/eval distributions aligned.
- `"first_sample"` is kept for backwards compatibility and ablation runs.

### What did **not** change

- GRPO group composition is still (≈) proportional to policy with `legal_move_sample_temperature=1.0` and `legal_move_sample_epsilon=0.05`.
- Reward path is unchanged (Pikafish scores all 32; combined reward unchanged).
- `game/play_best_candidate=False` still applies: Pikafish is never used as an action-selection oracle.

## 3. Run Command

Same as before. No CLI change needed — the new hyperparam picks up the default automatically.

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
```

To ablate, override the hyperparam in the dict (e.g. for a control run):

```python
"grpo/legal_move_action_selection": "first_sample",
```

## 4. Quantitative Results

**Run in progress** (W&B `output.log` ~17.5k lines, 4 episodes complete):

- Scoreboard: `all_episodes ally=17.5 enemy=149` over Ep 1–4 (still 0% ally win, expected this early; ally is just landing small captures).
- Greedy verified: `played_greedy=…@p=…` is on every ally turn.
- Spot-check (Ep 1 Rd 1): chosen `a9a8` (engine=3.68, cp_delta=-107) vs best-in-group `g6g5` / `b7e7` (engine=5.52, cp_delta≈0). The policy's argmax was clearly **not** Pikafish's best of the 32, confirming the need for the new engine-best agreement metrics added below.

After the next run, expected W&B targets to track:

- `game/chosen_is_engine_argmax_in_group_rate` — climbs as the policy learns to put more mass on the engine-best move *inside* the group.
- `game/engine_best_in_group_rate` — measures whether sampling 32 moves under the policy is wide enough to *cover* Pikafish's preferred move; if this stays low (≪50%), the group is missing the best option and GRPO can't learn it.
- `game/chosen_is_engine_best_overall_rate` — joint condition (covers in group **and** picks it). Hard ceiling = `engine_best_in_group_rate`.
- `game/mean_chosen_engine_rank_in_group` — 1.0 = perfect agreement, len/2 = random within group.
- `game/mean_chosen_minus_argmax_cp_delta` — negative magnitude = avg cp the policy leaves on the table by not picking the in-group best.
- Per-turn cp_delta (mean / median across episode), `game/mean_ally_cp_after_move_red` / EMA: expect improvement vs `first_sample` baseline.
- **Stdout sanity:** the `Policy-sampled ... played_greedy=<uci>@p=<f>` line and the new `Engine-best comparison: …` line should both appear each ally turn.

## 5. Conclusion & Next Steps

- This is a strict superset of the previous behavior: `"first_sample"` recovers it exactly.
- Greedy action selection has no impact on GRPO's gradient (all 32 candidates still contribute via group-relative advantages); it only changes which move is committed to the env.
- **Next:** after a run, check whether `game/ally_cp_after_move_red_ema` drift improves and whether `chinese_chess_episode_metrics_v2.csv` shows better legal-move quality per turn. If not, fall back to `"first_sample"` and re-investigate the policy entropy.
- **Followups (backlog):** consider a low-temperature `"low_temp"` mode (Option B in our design notes) and a `"group_argmax"` mode (Option C: argmax within the sampled group only).
