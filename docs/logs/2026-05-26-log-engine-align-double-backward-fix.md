# Experiment Log: Engine-Align Double-Backward Fix

**Date:** 2026-05-26
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal
Fix the Ep 23 resume crash where GRPO + engine-policy alignment in the same turn triggered `RuntimeError: Trying to backward through the graph a second time` at `align_loss.backward()`.

## 2. Configuration Changes
- **Code:** `LLM_RL_agent_FSDP_v2.py` `GRPOTrainer.train_group` — when `engine_scores_t` is present (`align_active`), defer GRPO micro-batch backward until after the alignment KL is built, then call a single `(epoch_group_loss + align_loss).backward()`. Same pattern in the sequential OOM fallback path.
- **Hyperparams:** unchanged.

## 3. Run Command
```bash
# Resume that previously crashed on Ep 23 Rd 1
python LLM_RL_agent_FSDP_v2.py --resume-from checkpoints/xiangqi_grpo_v2/ep_22
```

## 4. Quantitative Results
- **Crash site:** Ep 23, round 1, first turn with both GRPO PG term and engine alignment active (44 legal candidates logged).
- **Root cause:** `group_loss.backward()` per micro-batch freed autograd graphs while `align_logits_epoch` still referenced the same `cur_tok_lp` tensors.

## 5. Conclusion & Next Steps
- Fix applied; re-run Ep 23 resume to confirm training proceeds and `grpo/engine_align_*` metrics log on mixed GRPO+alignment turns.
- Monitor peak VRAM on alignment turns (deferred backward keeps all micro-batch graphs alive until the combined backward).
