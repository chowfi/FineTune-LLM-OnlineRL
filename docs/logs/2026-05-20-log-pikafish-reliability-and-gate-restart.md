# Experiment Log: Pikafish reliability + gate-only restart

**Date:** 2026-05-20
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

The May-16 run (episodes 12-18) showed the clearest per-move learning trend:
engine-best agreement rose and the chosen move's engine rank improved under the
gate reward with the stronger GRPO optimizer settings. The later combined
`gate + r_best/r_good` reward added extra Pikafish calls and noise while the
engine client was frequently restarting on poison FENs.

Hypothesis: keep the May-16 optimizer/reward/truncation shape, add the
GreedyEnemy epsilon curriculum, and first stabilize Pikafish. If learning
returns, then the combined bonus can be reintroduced as a separate ablation.

## 2. Configuration Changes

- `pikafish_eval.py`: set explicit Pikafish `Threads=2` and `Hash=128` after
  UCI startup so the engine does not compete with the LLM loop for all CPU
  cores.
- `pikafish_eval.py`: add a short-lived FEN-level poison cache. If Pikafish
  emits a critical error on `perft`, `bestmove`, or `evaluate_cp`, subsequent
  legal/eval calls for that FEN return `None` for 90 seconds instead of
  restarting once per candidate move.
- `reward/combine_gate_with_r_best`: `True` -> `False` for the next controlled
  restart. This returns to gate-only reward while isolating the Pikafish fix.
- `game/cp_saturation_threshold`: `6000.0` -> `4000.0`.
- `game/cp_saturation_consecutive`: `5` -> `3`.
- Kept May-16 GRPO optimizer settings unchanged:
  `grpo/lr=1e-5`, `grpo/beta=0.01`, `grpo/ppo_epochs=4`,
  `grpo/max_grad_norm=0.5`, `clip_eps_low=0.2`, `clip_eps_high=0.28`.
- Kept GreedyEnemy epsilon curriculum enabled.

## 3. Run Command

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_18 \
  --start-episode 19
```

Note: `ep_18` predates optimizer-state checkpointing, so this restart will not
load Adam moments. The first checkpoint saved by this run will include
`optimizer.pt`.

## 4. Quantitative Results

Pending next run. Watch:

- `game/engine_eval_success_rate`
- Pikafish restart count in `output.log`
- `game/chosen_is_engine_argmax_in_group_rate`
- `game/chosen_is_engine_best_overall_rate`
- `game/mean_chosen_engine_rank_in_group`
- `game/mean_chosen_minus_argmax_cp_delta`
- `game/cp_saturation_truncation_rate`

## 5. Conclusion & Next Steps

If engine reliability improves and May-16 alignment trends resume, continue the
gate-only curriculum for several checkpoints before reintroducing the combined
`r_best/r_good` bonus. If alignment still fails, investigate policy update
strength and action-selection confidence rather than reward-shape complexity.
