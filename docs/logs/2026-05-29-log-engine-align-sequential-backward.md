# Experiment Log: Engine-align sequential backward (OOM fix)

**Date:** 2026-05-29
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal
Engine-policy alignment was enabled (`grpo/engine_policy_align_coef=0.05`, `logprob_micro_batch=1`) but every ally turn logged `CUDA OOM during engine-policy alignment`. Hypothesis: the alignment pass kept too many candidate forward graphs alive (batched re-forward over cached micro-batches). Fix by (1) `empty_cache` before/ between phases and (2) two-phase sequential backward using detached softmax coefficients.

## 2. Configuration Changes
- **Code (`LLM_RL_agent_FSDP_v2.py`):** `_backward_engine_policy_alignment` rewritten:
  - Phase 1: no-grad sequential logits → KL metrics + `(policy - target)` coeffs.
  - Phase 2: one grad forward per valid candidate; `(coef_i * z_i).backward()` accumulates into existing GRPO grads.
  - `torch.cuda.synchronize()` + `empty_cache()` before phase 1 and between phases.
  - Call sites now always pass `query_ids_batch` / `response_ids_batch` (no cache re-forward).
- **Hyperparams:** unchanged (`logprob_micro_batch=1` already set in run config).

## 3. Run Command
Restart training after pulling this patch (same command as current v2 run).

## 4. Quantitative Results
*(Pending re-run.)* Success criteria:
- No `[GRPO] CUDA OOM during engine-policy alignment` on typical 32-candidate turns.
- W&B / CSV: nonzero `grpo/engine_align_kl` on most ally turns.

## 5. Conclusion & Next Steps
- If OOM persists: reduce `num_generations` or alignment temperature; last resort `engine_policy_align_coef=0`.
- If alignment runs: watch `game_chosen_is_engine_best_overall_rate` over ~10 episodes vs prior flat 0%.
