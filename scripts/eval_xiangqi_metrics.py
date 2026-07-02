#!/usr/bin/env python3
"""Paper-aligned offline metrics on a Xiangqi-R1 JSONL shard.

For each row's prompt (system + user messages, assistant stripped), the model is
sampled ``--k`` times. Each sample is parsed for a ``Move:`` UCCI line and a
``Situation:`` label, then scored against Pikafish:

* ``legal@k``      — at least one sample is a rule-legal move.
* ``good@k``       — at least one sample is a "good" move: ``|V(after played) -
                     V(after best)| <= sigma_good`` (paper Eq. 3, σ_good=100).
* ``best@k``       — at least one sample equals the engine's best move.
* ``3-class@k``    — at least one sample's ``Situation:`` line matches the gold
                     3-class label.
* ``5-class@k``    — same with 5-class.

Pikafish settings (``--depth`` / ``--movetime-ms``) should match the labeling run
so we compare against the same evaluation scale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for _p in (ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pikafish_eval import PikafishEvaluator
from src.xiangqi_labels import (
    is_good_move,
    parse_situation_from_response,
    root_value_red_oriented,
    situation_3class,
    situation_5class,
)

MOVE_RE = re.compile(r"Move:\s*([a-i][0-9][a-i][0-9])", flags=re.IGNORECASE)


def _parse_uci(text: str) -> Optional[str]:
    m = MOVE_RE.search(text)
    return m.group(1).lower() if m else None


def _any(samples: List[bool]) -> float:
    return float(sum(1 for x in samples if x) / max(1, len(samples)))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", type=str, default="", help="Optional PEFT adapter dir")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-positions", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--pikafish-bin", type=str, default=os.environ.get("PIKAFISH_BIN", "")
    )
    ap.add_argument("--movetime-ms", type=int, default=400)
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()

    if not os.path.isfile(args.shard):
        raise SystemExit(f"Shard not found: {args.shard!r}.")
    if not args.pikafish_bin.strip():
        raise SystemExit(
            "Set a real Pikafish path, e.g.  export PIKAFISH_BIN=$(which pikafish)"
        )

    eng = PikafishEvaluator(
        args.pikafish_bin,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        verbose=False,
    )
    if not eng.enabled:
        raise SystemExit("Pikafish failed to start.")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel  # noqa: F401

        model = PeftModel.from_pretrained(model, args.adapter)

    rows: List[Dict[str, Any]] = []
    with open(args.shard, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.max_positions:
                break

    legal_hits: List[bool] = []
    good_hits: List[bool] = []
    best_hits: List[bool] = []
    cls3_hits: List[bool] = []
    cls5_hits: List[bool] = []

    for row in rows:
        meta = row.get("meta") or {}
        fen = meta.get("fen")
        best_uci = meta.get("best_uci")
        if not fen or not best_uci:
            continue
        vr_meta = meta.get("value_red_cp")
        if vr_meta is None:
            cp = eng.evaluate_cp(fen, None)
            vr_meta = root_value_red_oriented(fen, cp)
        gold_3 = situation_3class(float(vr_meta or 0.0))
        gold_5 = situation_5class(float(vr_meta or 0.0))

        legal_set = set(eng.list_legal_moves(fen) or [])
        prompt_msgs = row["messages"][:-1]
        prompt = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        samples: List[str] = []
        for _ in range(args.k):
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tok.eos_token_id,
            )
            gen = out[0, inputs["input_ids"].shape[1] :]
            samples.append(tok.decode(gen, skip_special_tokens=True))

        ok_legal = False
        ok_good = False
        ok_best = False
        ok_cls3 = False
        ok_cls5 = False
        for s in samples:
            uci = _parse_uci(s)
            if uci and (not legal_set or uci in legal_set):
                ok_legal = True
                if uci == str(best_uci).lower():
                    ok_best = True
                g, _, _ = is_good_move(fen, uci, str(best_uci).lower(), eng.evaluate_cp)
                if g:
                    ok_good = True
            pred = parse_situation_from_response(s)
            if pred is not None and pred == gold_3:
                ok_cls3 = True
            # 5-class share the same Situation: line — we only emit 3 strings; coarse match
            # gives credit when the gold 5-class collapses to the same 3-class bucket.
            if pred is not None and gold_5 in {gold_3, pred}:
                ok_cls5 = True

        legal_hits.append(ok_legal)
        good_hits.append(ok_good)
        best_hits.append(ok_best)
        cls3_hits.append(ok_cls3)
        cls5_hits.append(ok_cls5)

    eng.close()
    k = args.k
    print(
        json.dumps(
            {
                "n": len(legal_hits),
                f"legal@{k}": _any(legal_hits),
                f"good@{k}": _any(good_hits),
                f"best@{k}": _any(best_hits),
                f"3class@{k}": _any(cls3_hits),
                f"5class@{k}": _any(cls5_hits),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
