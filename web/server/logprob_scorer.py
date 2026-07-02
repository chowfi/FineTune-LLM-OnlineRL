"""Batched response log-probability scoring for legal-move selection."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class MoveLogprobScorer:
    """Score ``Move: <uci>`` candidates under a causal LM (inference only)."""

    def __init__(self, model, tokenizer, device: torch.device, micro_batch: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.micro_batch = max(1, int(micro_batch))

    @torch.inference_mode()
    def score_moves(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
    ) -> List[float]:
        if not queries or not responses:
            return []
        scores: List[float] = []
        chunk_micro = self.micro_batch
        oom_retries_left = 3
        while oom_retries_left >= 0:
            try:
                for start in range(0, len(queries), chunk_micro):
                    end = min(start + chunk_micro, len(queries))
                    qs = queries[start:end]
                    rs = responses[start:end]
                    out = self._compute_response_log_probs_batch(qs, rs)
                    token_lp = out["token_lp"]
                    resp_mask = out["resp_mask"]
                    batch_scores = (token_lp * resp_mask).sum(dim=1).tolist()
                    scores.extend(float(x) for x in batch_scores)
                return scores
            except torch.cuda.OutOfMemoryError:
                scores.clear()
                if self.device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if chunk_micro <= 1:
                    raise
                chunk_micro = max(1, chunk_micro // 2)
                print(
                    f"[xiangqi-play] CUDA OOM scoring moves; retry micro_batch={chunk_micro}.",
                    flush=True,
                )
                oom_retries_left -= 1
        raise RuntimeError(
            "CUDA OOM scoring legal moves after repeated micro_batch shrink"
        )

    def _compute_response_log_probs_batch(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
    ) -> Dict[str, Any]:
        G = len(queries)
        q_lens = [int(q.numel()) for q in queries]
        r_lens = [int(r.numel()) for r in responses]
        max_total = max(q + r for q, r in zip(q_lens, r_lens))
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = torch.full(
            (G, max_total), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (G, max_total), dtype=torch.long, device=self.device
        )
        resp_tokens_padded = torch.zeros(
            (G, max_total), dtype=torch.long, device=self.device
        )
        resp_lens = torch.tensor(r_lens, dtype=torch.long, device=self.device)

        for i, (q, r) in enumerate(zip(queries, responses)):
            seq = torch.cat([q, r]).to(self.device, dtype=torch.long)
            length = seq.numel()
            input_ids[i, max_total - length :] = seq
            attention_mask[i, max_total - length :] = 1
            resp_tokens_padded[i, max_total - r.numel() :] = r.to(
                self.device, dtype=torch.long
            )

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted = logits[:, :-1, :]
        target = resp_tokens_padded[:, 1:]
        # Target log-probs only (avoid materializing full-vocab log_softmax tensor).
        shifted_f = shifted.float()
        log_norm = torch.logsumexp(shifted_f, dim=-1)
        target_logits = shifted_f.gather(2, target.unsqueeze(-1)).squeeze(-1)
        token_lp = target_logits - log_norm
        del logits, shifted, shifted_f, log_norm, target_logits

        pos = torch.arange(max_total - 1, device=self.device).unsqueeze(0)
        resp_pred_start = (max_total - resp_lens - 1).unsqueeze(1)
        response_mask = (pos >= resp_pred_start).to(token_lp.dtype)
        token_lp = token_lp * response_mask

        return {
            "token_lp": token_lp,
            "resp_mask": response_mask,
            "resp_lens": resp_lens,
            "max_total": max_total,
        }
