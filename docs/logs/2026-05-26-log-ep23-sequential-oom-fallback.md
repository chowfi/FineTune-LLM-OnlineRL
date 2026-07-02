# Experiment Log: Ep 23 Sequential OOM Fallback

**Date:** 2026-05-26
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal

The Ep 23 adapter-only resume failed during the first GRPO update even after
entering the sequential OOM fallback. Goal: identify why the fallback still
retained too much GPU memory and make the next resume attempt survive this path.

## 2. Configuration Changes

- Updated `LLM_RL_agent_FSDP_v2.py` so the sequential GRPO fallback backprops
  each sample immediately.
- The OOM fallback now skips engine-policy alignment for that turn, because
  alignment needs all candidate logits in one differentiable group and would
  retain one graph per candidate.
- Added a stdout marker:
  `[GRPO] Sequential fallback skips engine-policy alignment to avoid retaining per-sample graphs.`

## 3. Run Command

Failed command:

```bash
mv checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt checkpoints/xiangqi_grpo_v2/ep_22/optimizer.pt.bak
export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b \
  --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_22 \
  --start-episode 23
```

## 4. Quantitative Results

- Failure happened on Ep 23 Rd 1 after sampling 32 distinct legal moves.
- Batched PPO forward OOMed and entered the sequential fallback.
- Samples 24-32 OOMed and were skipped.
- Crash still occurred at `seq_group_loss.backward()` with only ~102 MiB free:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 238.00 MiB`.
- No optimizer step completed (`global_train_step=0` in the interrupted checkpoint).

Validation after code change:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run python -m py_compile LLM_RL_agent_FSDP_v2.py
```

## 5. Conclusion & Next Steps

- Root cause: the sequential fallback was not fully sequential when
  `grpo/engine_policy_align_coef > 0`; it deferred backward to combine GRPO and
  alignment losses, retaining many per-sample graphs and causing another OOM.
- Next run should retry Ep 23 from `checkpoints/xiangqi_grpo_v2/ep_22` with the
  same adapter-only resume command. If the fallback is reached, expect the new
  alignment-skip marker and either a reduced-sample GRPO step or skipped samples
  without a hard crash.
