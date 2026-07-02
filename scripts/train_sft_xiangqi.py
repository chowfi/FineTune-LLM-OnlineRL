#!/usr/bin/env python3
"""LoRA SFT on the Xiangqi-R1 JSONL produced by ``build_xiangqi_sft_dataset.py``.

Each input row has ``messages: [system, user, assistant]`` formatted as a Qwen chat
template; we just pass them through Unsloth + TRL's ``SFTTrainer``. The default
target is Qwen2.5-7B-Instruct (matches the GRPO v2 backbone), with a small LoRA so
this can run on a single 24 GB GPU.

Example::

    export PIKAFISH_BIN=$(which pikafish)
    uv run python scripts/build_xiangqi_sft_dataset.py --samples 50000
    uv run python scripts/train_sft_xiangqi.py \\
        --dataset data/xiangqi_sft/xiangqi_sft_train.jsonl \\
        --output-dir checkpoints/xiangqi_sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for _p in (ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Unsloth must load before trl/transformers for full patches (library warning).
import unsloth  # noqa: F401
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="data/xiangqi_sft/xiangqi_sft_train.jsonl",
        help="JSONL from build_xiangqi_sft_dataset.py",
    )
    ap.add_argument("--output-dir", type=str, default="checkpoints/xiangqi_sft")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(
            f"Dataset not found: {args.dataset!r}. Build it first with "
            "scripts/build_xiangqi_sft_dataset.py (and download_xiangqi_pgn.py)."
        )

    raw = _load_jsonl(args.dataset)
    if not raw:
        raise SystemExit(f"Empty dataset: {args.dataset}")
    ds = Dataset.from_list([{"messages": r["messages"]} for r in raw])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    def formatting_func(example: Dict[str, Any]) -> list[str]:
        """Apply the model's chat template. Handles both single-row and batched calls."""
        convs = example["messages"]
        if not convs:
            return []
        if isinstance(convs[0], dict):
            text = tokenizer.apply_chat_template(
                convs, tokenize=False, add_generation_prompt=False
            )
            return [text]
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in convs
        ]

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        logging_steps=5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        max_length=args.max_seq_length,
        seed=args.seed,
        report_to="none",
    )
    setattr(train_args, "max_seq_length", int(args.max_seq_length))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=train_args,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
