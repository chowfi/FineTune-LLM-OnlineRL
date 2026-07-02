# Experiment Log: Frozen SFT KL reference + OOM-safe engine alignment

**Date:** 2026-05-29
**Agent/Author:** Cursor Agent

## 1. Hypothesis / Goal

Anchor GRPO KL to the **frozen SFT adapter** (not raw base Qwen) so RL preserves
SFT legality/format while tuning tactics. Re-enable **engine-policy alignment**
without the prior fused-backward CUDA OOM.

## 2. Configuration Changes

**Code (`LLM_RL_agent_FSDP_v2.py`):**

- New frozen LoRA slot `sft_ref` loaded from `checkpoint/sft_ref_adapter_path`
  (default `checkpoints/xiangqi_sft`); never trained.
- `grpo/kl_reference`: `"sft_ref"` (fallback to base if slot missing).
- KL reference forwards use `set_adapter("sft_ref")` instead of
  `disable_adapter()`.
- `grpo/engine_policy_align_coef`: `0.0` → **`0.05`**.
- Engine alignment runs as a **second forward + backward** after per-micro GRPO
  backwards (no deferred fused graph).
- Sequential fallback also runs alignment via the separate path.
- Checkpoints save **`default` adapter only** (`selected_adapters=["default"]`).

**New hyperparams:**

| Key | Default |
|-----|---------|
| `checkpoint/sft_ref_adapter_path` | `checkpoints/xiangqi_sft` |
| `grpo/kl_reference` | `sft_ref` |
| `grpo/engine_policy_align_coef` | `0.05` |

## 3. Run Command

```bash
export PIKAFISH_BIN=/home/fchow/bin/pikafish
uv run torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py \
  --model-size 7b --mixed-precision \
  --resume-from checkpoints/xiangqi_grpo_v2/ep_40 \
  --start-episode 41
```

Expect stdout: `[kl-ref] Loaded frozen sft_ref adapter from 'checkpoints/xiangqi_sft'`.

## 4. Quantitative Results

Not run yet. Watch:

- Offline `good@1` / `legal@1` vs SFT on holdout (should stop regressing).
- `grpo/engine_align_loss`, `grpo/engine_align_kl` > 0 when Pikafish scores valid.
- `grpo/mean_kl_move` vs prior run (may shift with sft_ref anchor).
- OOM: alignment skip message should be rare; no fused-backward OOM.

## 5. Conclusion & Next Steps

- **Resume:** adapter `default` = latest RL; `sft_ref` always = original SFT.
- If alignment still OOMs, lower `grpo/logprob_micro_batch` or
  `grpo/engine_policy_align_coef` to `0.03`.
- Re-run holdout eval every ~10 episodes to confirm `good@1` trends above SFT.
