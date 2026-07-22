"""
inference.py — Generation engine for the backend.
Now aligned with the SFT stack: tiktoken chat tokenizer (sft_tokenizer.py),
KV-cache decoding (one token per step, not a full recompute), fp32-only,
RangeFlow unchanged. Checkpoint loading strips optimizer state the same way
standalone_inference.py does.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from config import CHECKPOINT_PATH, FORCE_FP32, ModelConfig
from model import GPT
from sft_tokenizer import (
    build_chat_tokenizer,
    render_prompt_for_generation,
    decode_response,
    ROLE_TOKEN_ID,
    END_ID,
    MIN_VOCAB_SIZE_REQUIRED,
)

logger = logging.getLogger("dheeren's_chat.inference")

DEFAULT_SYSTEM = (
    "You are a helpful and knowledgeable AI assistant. "
    "Answer clearly and concisely."
)


@dataclass
class GenerationRequest:
    prompt:             str
    system:             str   = DEFAULT_SYSTEM
    max_tokens:         int   = 256
    temperature:        float = 0.7
    top_p:              float = 0.9
    top_k:              int   = 40
    repetition_penalty: float = 1.15
    range_epsilon:      float = 0.1


# ---------------------------------------------------------------------------
# Checkpoint loader — strip optimizer state, rebuild ModelConfig from the
# checkpoint itself (same "never guess the architecture" rule sft_train.py
# and standalone_inference.py both follow).
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: str, device: str) -> tuple[GPT, ModelConfig]:
    import os
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at '{ckpt_path}'.")

    logger.info("[checkpoint] Loading %s", ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in raw or "config" not in raw:
        raise RuntimeError(
            f"'{ckpt_path}' doesn't look like a checkpoint.py-produced file "
            f"(missing 'model'/'config'). Found keys: {list(raw.keys())}"
        )

    model_cfg = ModelConfig(**raw["config"])
    if model_cfg.vocab_size < MIN_VOCAB_SIZE_REQUIRED:
        raise RuntimeError(
            f"model_cfg.vocab_size={model_cfg.vocab_size} < {MIN_VOCAB_SIZE_REQUIRED} "
            f"required for the chat special tokens — this checkpoint predates SFT."
        )

    model = GPT(model_cfg)
    model.load_state_dict(raw["model"], strict=True)
    del raw

    dtype = torch.float32 if FORCE_FP32 else torch.float32  # fp32 only, always
    model = model.to(device=device, dtype=dtype)
    model.eval()

    step     = None
    val_loss = None
    logger.info(
        "[checkpoint] loaded — d_model=%d n_layers=%d n_heads=%d max_seq_len=%d  params=%.1fM",
        model_cfg.d_model, model_cfg.n_layers, model_cfg.n_heads, model_cfg.max_seq_len,
        model.count_params(non_embedding=True) / 1e6,
    )
    return model, model_cfg


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(logits, recent_ids, penalty, window):
    if penalty == 1.0 or not recent_ids:
        return logits
    ids = torch.tensor(list(set(recent_ids[-window:])), dtype=torch.long, device=logits.device)
    ids = ids[ids < logits.size(0)]
    if ids.numel() == 0:
        return logits
    logits = logits.clone()
    scores = logits[ids]
    logits[ids] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def _top_k_filter(logits, top_k):
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits, top_p):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    remove = cum_probs - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(logits, recent_ids, temperature, top_k, top_p, repetition_penalty):
    logits = logits.clone()
    logits[MIN_VOCAB_SIZE_REQUIRED:] = float("-inf")   # never-trained padding rows

    logits = _apply_repetition_penalty(logits, recent_ids, repetition_penalty, window=64)
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        logits = _top_k_filter(logits.unsqueeze(0), top_k).squeeze(0)
    if top_p < 1.0:
        logits = _top_p_filter(logits.unsqueeze(0), top_p).squeeze(0)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._device = self._pick_device()
        self._model:  Optional[GPT]         = None
        self._cfg:    Optional[ModelConfig] = None
        self._enc     = None
        self._ready   = False
        logger.info("InferenceEngine created — device: %s", self._device.upper())

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Heavy lifting — call this off the event loop (see main.py's
        lifespan) so server startup itself stays sub-second."""
        t0 = time.perf_counter()

        # tiktoken load is essentially free (no network, no HF download) —
        # this alone removes most of the old startup latency vs AutoTokenizer.
        self._enc = build_chat_tokenizer()

        model, cfg = _load_checkpoint(CHECKPOINT_PATH, self._device)
        self._model = model
        self._cfg   = cfg
        self._ready = True

        logger.info("✅ Engine ready in %.2fs — accepting requests", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: GenerationRequest) -> dict:
        if not self._ready:
            raise RuntimeError("Model is not loaded yet.")
        with self._lock:
            return self._run_generation(req, stream_queue=None, loop=None)

    def generate_stream(self, req: GenerationRequest, queue: asyncio.Queue,
                         loop: asyncio.AbstractEventLoop) -> None:
        if not self._ready:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", "Model not loaded"))
            return
        with self._lock:
            self._run_generation(req, stream_queue=queue, loop=loop)

    # ------------------------------------------------------------------
    # Core generation — prefill once (capture), then decode ONE token
    # per step via the KV cache (guard). This replaces the old approach
    # of recomputing the full sequence every step.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _run_generation(self, req: GenerationRequest,
                         stream_queue: Optional[asyncio.Queue],
                         loop: Optional[asyncio.AbstractEventLoop]) -> dict:
        streaming = stream_queue is not None
        t_start = time.perf_counter()
        model, cfg, device = self._model, self._cfg, self._device

        messages = [{"role": "system", "content": req.system},
                    {"role": "user", "content": req.prompt}]
        prompt_ids = render_prompt_for_generation(messages, self._enc)

        if len(prompt_ids) >= cfg.max_seq_len:
            raise RuntimeError(
                f"Prompt ({len(prompt_ids)} tokens) doesn't fit in "
                f"max_seq_len={cfg.max_seq_len}."
            )

        max_new = min(req.max_tokens, cfg.max_seq_len - len(prompt_ids) - 1)

        kv_cache = model.new_kv_cache(batch_size=1, device=device, dtype=torch.float32)

        # ── Phase A: prefill + capture ─────────────────────────────────
        model.clear_anchors()
        model.set_epsilon(req.range_epsilon)
        model.set_range_mode("capture")

        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, _ = model(x, kv_cache=kv_cache, start_pos=0)
        next_logits = logits[0, -1, :].float()

        # ── Phase B: guard — one new token per forward pass ────────────
        model.set_range_mode("guard")

        generated_ids: list[int] = []
        for step in range(max_new):
            token_id = sample_next_token(
                next_logits, prompt_ids + generated_ids,
                req.temperature, req.top_k, req.top_p, req.repetition_penalty,
            )

            if token_id == END_ID:
                break

            generated_ids.append(token_id)

            if streaming:
                current_text = decode_response(generated_ids, self._enc)
                prev_text    = decode_response(generated_ids[:-1], self._enc) if len(generated_ids) > 1 else ""
                new_chars    = current_text[len(prev_text):]
                if new_chars:
                    loop.call_soon_threadsafe(stream_queue.put_nowait, ("token", new_chars))

            if kv_cache.capacity_left() < 1:
                break

            x = torch.tensor([[token_id]], dtype=torch.long, device=device)
            logits, _ = model(x, kv_cache=kv_cache, start_pos=kv_cache.length)
            next_logits = logits[0, -1, :].float()

        model.set_range_mode("standard")
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if streaming:
            loop.call_soon_threadsafe(stream_queue.put_nowait, ("done", len(generated_ids)))
            return {}

        response_text = decode_response(generated_ids, self._enc)
        return {
            "response":         response_text,
            "tokens_generated": len(generated_ids),
            "elapsed_ms":       round(elapsed_ms, 2),
            "device":           device,
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device


engine = InferenceEngine()