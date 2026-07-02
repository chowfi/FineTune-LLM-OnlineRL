# Experiment Log: GRPO first-step backward crash after loading SFT adapter

**Date:** 2026-05-14  
**Agent/Author:** Cursor agent

## 1. Hypothesis / Goal

Document why online GRPO (`LLM_RL_agent_FSDP_v2.py`) could fail on the **first** `train_group` backward with:

`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

after loading a LoRA adapter from SFT (`PeftModel.from_pretrained(..., is_trainable=True)`), and what code changes fix it.

## 2. Is this “because of the new SFT”?

**No.** SFT itself does not introduce this bug.

- **SFT** (`scripts/train_sft_xiangqi.py`) uses Unsloth / TRL training, which already sets up training correctly (including patterns equivalent to “inputs need grad” for checkpointing when needed).
- **GRPO** uses a **custom** trainer inside `LLM_RL_agent_FSDP_v2.py` (`GRPOTrainerOnline`): it runs its own forwards for log-probs, enables gradient checkpointing on the unwrapped model, optionally uses the **legal-move policy sampler** (which calls `.eval()` on the policy for scoring), and uses PEFT’s **`disable_adapter()`** context for the reference policy.

The crash showed up when you **first** combined: loaded SFT adapter + GRPO + legal-move sampler + `torchrun` single-GPU. Earlier runs without that exact combo (or without hitting this path) would not have surfaced the same failure.

## 3. What was wrong (simple)

Three separate issues stacked:

1. **Gradient checkpointing + frozen base + LoRA**  
   The GRPO trainer turns on `gradient_checkpointing_enable()` but did **not** call `enable_input_require_grads()`. For many HF + PEFT setups, without that hook the graph from logits back to LoRA weights can be missing, so `log_probs` have no `grad_fn` → `backward()` errors.

2. **Legal-move sampler left the policy in `eval()`**  
   The sampler scores every legal move under `model.eval()` for inference. Training must run under `model.train()` again afterward. Unsloth’s patched blocks also tie checkpointing behavior to `self.training`, so staying in `eval()` is especially risky right before a training forward.

3. **Reference policy toggle was a no-op on current PEFT**  
   `PeftModel` exposes `disable_adapter()` (context manager), not `disable_adapter_layers`. The old `_toggle_adapters` path did nothing on modern `peft`, so ref vs adapter log-probs were not separated as intended (KL still wrong until fixed; not always the direct cause of the backward error, but corrected alongside).

## 4. Configuration / code changes

- `LLM_RL_agent_FSDP_v2.py`
  - After `gradient_checkpointing_enable()`, call `enable_input_require_grads()` when available.
  - Add `restore_policy_train_mode(model)` (`model.train()` on outer + unwrapped module) and call it at the start of `train_group` and in a `finally` after legal-move scoring.
  - Precompute + sequential GRPO: use `with unwrapped.disable_adapter():` for reference log-probs when supported; keep legacy `_toggle_adapters` only for older APIs.

## 5. Run command (sanity)

Same as before (example):

```bash
torchrun --nproc_per_node 1 LLM_RL_agent_FSDP_v2.py --model-size 7b --mixed-precision
```

With `checkpoint/load_adapter_path` pointing at the SFT adapter directory (e.g. `checkpoints/xiangqi_sft`).

## 6. Conclusion & next steps

- The failure is **GRPO integration** (custom trainer + sampler + checkpointing + PEFT API), not a defect in the SFT dataset or adapter files.
- After the fix, re-run GRPO from the SFT init and confirm the first ally `train_group` completes and W&B shows non-empty GRPO stats.
- Related pipeline context: [2026-05-13 Xiangqi-R1 SFT rebuild log](2026-05-13-log-xiangqi-r1-sft-rebuild.md).
