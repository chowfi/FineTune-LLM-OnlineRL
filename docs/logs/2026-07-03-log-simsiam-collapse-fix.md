# SimSiam consistency collapse confirmed at iteration 30; BatchNorm projector fix

**Date:** 2026-07-03
**Agent/Author:** Claude Code (Fable 5)

## 1. Hypothesis / Goal
`loss/consistency` sat at exactly −1.000 from iteration ~3 of the first real
MuZero run. Hypothesis: projector collapse (trivial matching), not a perfect
world model. Build a diagnostic, confirm on the live checkpoint, fix.

## 2. Configuration Changes
- New `muzero/diagnose_consistency.py` — measures pairwise cosine among
  projections of different samples (collapse ⇒ ≈1.0) vs trunk diversity.
- `muzero/network.py` projector: `LayerNorm(1024)` → `BatchNorm1d(1024)`,
  plus SimSiam-style affine-free output BN; predictor gains `BatchNorm1d(512)`.
  (SimSiam requires batch-wise normalization in the projector: LayerNorm
  normalizes per-sample and cannot prevent all samples mapping to one point.)
- New regression test `test_projector_uses_batchnorm_not_layernorm`.

## 3. Run Command
```bash
uv run python -m muzero.diagnose_consistency --ckpt checkpoints/muzero_xiangqi/latest.pt  # on GPU box
uv run python -m muzero.diagnose_consistency --tiny --train-steps 60          # local pressure test
uv run pytest muzero/tests -q                                                  # 45 passed, 5 skipped
```

## 4. Quantitative Results
- **Live checkpoint (iteration 30): collapsed.** Pairwise cosine of target
  projections = 1.0000, per-dim std of normalized projections = 0.0001, while
  trunk latents stayed diverse (pairwise 0.61). Collapse fully contained in
  projector/predictor; policy/value trunk unharmed.
- Local pressure test (unlearnable synthetic data, 60 train steps):
  LayerNorm projector → loss −0.998, pairwise 0.999 (reproduces the run).
  BatchNorm projector → loss dips to −0.72 then **recovers to −0.18**,
  pairwise 0.376. BN structurally blocks the constant-output solution.

## 5. Qualitative Outcome
The −1.0 consistency loss was contributing a constant −2.0 to `loss/total`
(λ=2.0) with zero learning signal. All other heads were learning normally
(value-cp correlation up to ~0.2–0.3, promotions at iter 14), so ~30
iterations are cheap to discard.

## 6. Repo / Handoff Updates
- `docs/AGENT_TODO.md`: unchanged (Active task still "first run"; run must be
  restarted with the new architecture).
- **Checkpoint compatibility:** the BN layers change the state dict —
  `checkpoints/muzero_xiangqi/latest.pt` from the collapsed run cannot be
  resumed. Restart from scratch (warm start ≈ 15 min post-perft-fix).

## 7. Conclusion & Next Steps
- On the GPU box: stop the run, `git pull`, delete/ignore the old checkpoint,
  `uv run python -m muzero.train` fresh.
- Watch `loss/consistency`: healthy is a value that moves and stays clear of
  −1.0 (typically drifting negative slowly as the world model improves). If it
  pins at −1.0 again, re-run the diagnostic against the new checkpoint.
