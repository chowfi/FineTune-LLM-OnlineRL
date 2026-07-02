"""Load ep LoRA and pick Black moves via greedy legal-move logprob scoring."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gym import Env
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.benchmark.xiangqi_prompt import format_xiangqi_turn_messages
from web.server.flip_utils import (
    action_to_algebraic,
    flip_move,
    get_flipped_enemy_legal_actions,
)
from web.server.logprob_scorer import MoveLogprobScorer
from src.xiangqi_board import board_to_graphic, board_to_fen


def resolve_play_device(device: str) -> Tuple[torch.device, Dict[str, Any]]:
    """Map CLI/env device string to torch device + ``from_pretrained`` kwargs."""
    name = (device or "cuda").strip().lower()
    if name == "cpu":
        return torch.device("cpu"), {
            "dtype": torch.float32,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }
    if name in ("cuda", "gpu", "auto"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA/GPU requested for the play engine but torch.cuda.is_available() "
                "is False. Use --device cpu or free the GPU."
            )
        torch.cuda.empty_cache()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda:0"), {
            "dtype": dtype,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
        }
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} not available")
        torch.cuda.empty_cache()
        idx = dev.index if dev.index is not None else 0
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dev, {
            "dtype": dtype,
            "device_map": {"": idx},
            "low_cpu_mem_usage": True,
        }
    return dev, {
        "dtype": torch.float32,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }


class EnginePlayer:
    def __init__(
        self,
        *,
        adapter_path: str,
        base_model: str = "unsloth/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_prompt_length: int = 768,
        logprob_micro_batch: int = 4,
    ):
        self.torch_device, load_kwargs = resolve_play_device(device)
        env_max = os.environ.get("XIANGQI_PLAY_MAX_PROMPT_LENGTH", "").strip()
        if env_max:
            max_prompt_length = int(env_max)
        self.max_prompt_length = int(max_prompt_length)
        micro = logprob_micro_batch
        env_micro = os.environ.get("XIANGQI_PLAY_LOGPROB_MICRO_BATCH", "").strip()
        if env_micro:
            micro = int(env_micro)
        if self.torch_device.type == "cpu":
            micro = min(micro, 2)

        print(
            f"[xiangqi-play] torch device={self.torch_device} dtype={load_kwargs.get('dtype')}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True, **load_kwargs
        )
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # Keep adapter weights on the same device map as the base (avoids implicit CUDA load).
        adapter_map = load_kwargs.get("device_map")
        self.model = PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=False,
            device_map=adapter_map,
        )
        self.model.eval()
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize()

        self.scorer = MoveLogprobScorer(
            self.model,
            self.tokenizer,
            self.torch_device,
            micro_batch=micro,
        )

    @torch.inference_mode()
    def choose_black_move(
        self,
        env: Env,
        last_human_move: Optional[str],
    ) -> Tuple[str, int]:
        """Return (algebraic move on real board, action id)."""
        flipped_actions, flipped_to_original = get_flipped_enemy_legal_actions(env)
        if not flipped_actions:
            raise RuntimeError("No legal Black moves")

        flipped_board = -env.state[::-1, :]
        flipped_enemy_desc = flip_move(last_human_move) if last_human_move else None
        hint_actions = list(flipped_actions)
        random.shuffle(hint_actions)
        legal_hints = [action_to_algebraic(int(a)) for a in hint_actions]

        messages = format_xiangqi_turn_messages(
            fen=board_to_fen(flipped_board),
            graphic=board_to_graphic(flipped_board),
            enemy_move_desc=flipped_enemy_desc,
            legal_moves_hint=legal_hints,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        if encoded.input_ids.size(1) > self.max_prompt_length:
            encoded.input_ids = encoded.input_ids[:, -self.max_prompt_length :]
            if encoded.attention_mask is not None:
                encoded.attention_mask = encoded.attention_mask[
                    :, -self.max_prompt_length :
                ]
        query_ids = encoded.input_ids[0].to(self.torch_device)

        move_probe_texts = [
            f"Move: {action_to_algebraic(action)}" for action in flipped_actions
        ]
        response_ids_batch = [
            self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .to(self.torch_device)
            for text in move_probe_texts
        ]
        query_ids_batch = [query_ids for _ in response_ids_batch]

        scores = self.scorer.score_moves(query_ids_batch, response_ids_batch)
        if not scores or len(scores) != len(flipped_actions):
            chosen_orig = int(random.choice(list(flipped_to_original.values())))
        else:
            best_idx = int(np.argmax(np.array(scores, dtype=np.float64)))
            chosen_flipped = flipped_actions[best_idx]
            chosen_orig = flipped_to_original[chosen_flipped]

        move_str = action_to_algebraic(chosen_orig)
        return move_str, chosen_orig
