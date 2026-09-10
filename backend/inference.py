"""
inference.py — Llama-cpp-python generation engine for GGUF models.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama

from config import (
    GGUF_MODEL_PATH,
    GGUF_N_CTX,
    GGUF_N_THREADS,
    GGUF_N_GPU_LAYERS,
    GGUF_FLASH_ATTN,
    GGUF_TYPE_K,
    GGUF_TYPE_V,
)

logger = logging.getLogger("dheeren's_chat.inference")


@dataclass
class GenerationRequest:
    messages:           list
    system:             str   = ""
    max_tokens:         int   = 256
    temperature:        float = 0.7
    top_p:              float = 0.9
    top_k:              int   = 40
    repetition_penalty: float = 1.15
    range_epsilon:      float = 0.1


class InferenceEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm: Optional[Llama] = None
        self._ready = False
        self._device = "cpu" if GGUF_N_GPU_LAYERS == 0 else f"cuda ({GGUF_N_GPU_LAYERS} layers)"

    def load_model(self) -> None:
        if not os.path.exists(GGUF_MODEL_PATH):
            raise FileNotFoundError(f"GGUF model not found at '{GGUF_MODEL_PATH}'")

        logger.info("[inference] Loading GGUF model from %s...", GGUF_MODEL_PATH)
        t0 = time.perf_counter()

        self._llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=GGUF_N_CTX,
            n_threads=GGUF_N_THREADS,
            n_gpu_layers=GGUF_N_GPU_LAYERS,
            flash_attn=GGUF_FLASH_ATTN,
            type_k=GGUF_TYPE_K,
            type_v=GGUF_TYPE_V,
            verbose=False,
        )
        self._ready = True
        logger.info("✅ GGUF Engine loaded in %.2fs — ready for requests", time.perf_counter() - t0)

    def _fit_messages_to_context(self, messages: list, max_new_tokens: int) -> list:
        """
        Reimplements original context fitting logic: drops oldest user/assistant pairs
        so rendered prompt tokens + max_new_tokens fit within GGUF_N_CTX.
        """
        if not messages or self._llm is None:
            return messages

        messages = list(messages)
        has_system = messages[0]["role"] == "system"
        floor = 2 if has_system else 1

        while len(messages) > floor:
            # Estimate token length of the conversation turns
            text_block = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            token_count = len(self._llm.tokenize(text_block.encode("utf-8")))

            if token_count + max_new_tokens < GGUF_N_CTX:
                break

            # Drop oldest user/assistant pair after system prompt
            drop_at = 1 if has_system else 0
            del messages[drop_at:drop_at + 2]

        return messages

    def _prepare_messages(self, req: GenerationRequest) -> list:
        msgs = []
        if req.system and req.system.strip():
            msgs.append({"role": "system", "content": req.system.strip()})
        msgs.extend(req.messages)
        return self._fit_messages_to_context(msgs, req.max_tokens)

    def generate(self, req: GenerationRequest) -> dict:
        if not self._ready or self._llm is None:
            raise RuntimeError("Model is not loaded yet.")

        messages = self._prepare_messages(req)
        t0 = time.perf_counter()

        with self._lock:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repeat_penalty=req.repetition_penalty,
                stream=False,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        reply_text = response["choices"][0]["message"].get("content", "")
        tokens_generated = response.get("usage", {}).get("completion_tokens", len(reply_text.split()))

        return {
            "response":         reply_text,
            "tokens_generated": tokens_generated,
            "elapsed_ms":       round(elapsed_ms, 2),
            "device":           self._device,
        }

    def generate_stream(
        self,
        req: GenerationRequest,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if not self._ready or self._llm is None:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", "Model not loaded"))
            return

        messages = self._prepare_messages(req)

        with self._lock:
            try:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repeat_penalty=req.repetition_penalty,
                    stream=True,
                )

                tokens_generated = 0
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        tokens_generated += 1
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", content))

                loop.call_soon_threadsafe(queue.put_nowait, ("done", tokens_generated))

            except Exception as exc:
                logger.error("Streaming error: %s", exc, exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device


engine = InferenceEngine()