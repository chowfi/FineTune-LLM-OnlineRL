"""Plain-HF inference wrapper for the base Qwen2.5-7B-Instruct.

Loads the base model (no LoRA, no Unsloth - keeps the SFT job's
``unsloth_compiled_cache/`` untouched), runs the chat template through it,
and parses out ``Move: <uci>`` with a permissive regex that accepts both
chess (4 or 5 chars, with optional promotion letter) and xiangqi (4 chars)
UCI moves. Illegal / unparseable outputs fall back to a uniformly random
legal move (mirrors :class:`XiangqiAgent` in v2), counted as ``format_fail``.

Telemetry returned per move::

    {
        "move": "e2e4",             # the chosen move in LLM dialect
        "parse_ok": True,           # regex matched a UCI token
        "legal_ok": True,           # parsed move was in legal_moves list
        "format_fail": False,       # True if we fell back to random
        "raw_text": "...",          # generated text (post-prompt)
        "gen_tokens": 42,           # number of generated tokens
        "wall_sec": 1.23,           # generate() wall time
    }
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Permissive UCI: chess "e2e4[q]" OR xiangqi "b7b4" (digits 0-9 for xiangqi,
# 1-8 for chess - we accept any digit and let the legality check filter).
MOVE_RE = re.compile(
    r"Move\s*:\s*([a-h][0-9][a-h][0-9][qrbn]?|[a-i][0-9][a-i][0-9])",
    flags=re.IGNORECASE,
)


@dataclass
class LLMMoveResult:
    move: str
    parse_ok: bool
    legal_ok: bool
    format_fail: bool
    raw_text: str
    gen_tokens: int
    wall_sec: float


def _extract_move(text: str) -> Optional[str]:
    if not text:
        return None
    m = MOVE_RE.search(text)
    return m.group(1).lower() if m else None


class LLMPlayer:
    """Stateless LLM move generator. One instance per benchmark run."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        dtype: str = "bfloat16",
        device: str = "auto",
        max_new_tokens: int = 192,
        temperature: float = 0.6,
        top_p: float = 0.9,
        do_sample: bool = True,
        seed: int = 0,
        attn_implementation: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self._rng = random.Random(seed)
        self._gen_seed = int(seed)

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device not in {"auto"}:
            self.model = self.model.to(device)
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def _build_prompt_ids(self, messages: List[Dict[str, str]]):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        return encoded, prompt

    def generate_move(
        self,
        messages: List[Dict[str, str]],
        legal_moves: List[str],
    ) -> LLMMoveResult:
        """Generate a single move. On parse/legality failure, fall back to
        a uniformly random legal move so the game never stalls."""
        encoded, _prompt = self._build_prompt_ids(messages)
        prompt_len = encoded["input_ids"].shape[1]

        if torch_cuda_available(self._torch):
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self._torch.no_grad():
            output_ids = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch_cuda_available(self._torch):
            self._torch.cuda.synchronize()
        wall_sec = time.perf_counter() - t0

        new_tokens = output_ids[0, prompt_len:]
        gen_tokens = int(new_tokens.shape[0])
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = _extract_move(raw_text)
        parse_ok = parsed is not None

        # Normalise legal-move list for membership checking (lowercase).
        legals_lower = {m.lower() for m in legal_moves}

        legal_ok = parse_ok and parsed in legals_lower
        if legal_ok:
            chosen = parsed
            format_fail = False
        else:
            # Random legal fallback. Game never stalls.
            chosen = self._rng.choice(sorted(legals_lower)) if legals_lower else ""
            format_fail = True

        return LLMMoveResult(
            move=chosen,
            parse_ok=parse_ok,
            legal_ok=legal_ok,
            format_fail=format_fail,
            raw_text=raw_text,
            gen_tokens=gen_tokens,
            wall_sec=wall_sec,
        )


def torch_cuda_available(torch_module) -> bool:
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:
        return False
