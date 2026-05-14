"""
inference.py — Generation engine for the Noir Whisper backend.
Updated to use the Senku model (HF tokenizer, RoPE, RMSNorm, SwiGLU).

Two generation methods:
  • generate()        — blocking, returns the full text at once.
  • generate_stream() — pushes each decoded token into an asyncio.Queue
                        as it is produced, enabling SSE streaming.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    ACTIVE_CONFIG,
    CHECKPOINT_PATH,
    MODEL_PRESETS,
    SPECIAL_TOKENS,
    TOKENIZER_NAME,
    TOKENIZER_REAL_VOCAB,
    ModelConfig,
)
from model import GPT

logger = logging.getLogger("noir_whisper.inference")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are Senku, a helpful and knowledgeable AI assistant. "
    "Answer clearly and concisely."
)


def _build_chat_prompt(user_message: str, system: str = DEFAULT_SYSTEM) -> str:
    """
    Wrap a user message in the ChatML format the SFT model was trained on:

        <|system|>
        {system}
        <|user|>
        {user_message}
        <|assistant|>

    The model generates from the open <|assistant|> tag onward.
    """
    return f"<|system|>\n{system}\n<|user|>\n{user_message}\n<|assistant|>\n"


@dataclass
class GenerationRequest:
    prompt:             str
    system:             str   = DEFAULT_SYSTEM   # overridable system prompt
    max_tokens:         int   = 512
    temperature:        float = 0.7
    top_p:              float = 0.9
    top_k:              int   = 50
    repetition_penalty: float = 1.1
    range_epsilon:      float = 0.1


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def _load_tokenizer(name: str):
    """
    Load HuggingFace tokenizer with Senku special tokens added.

    Returns (tokenizer, encode_fn, safe_decode_fn, stop_ids, real_vocab_size).

    safe_decode_fn  — never raises; silently drops ids outside the real vocab.
    stop_ids        — set of int ids that should halt generation.
    """
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    real_vocab = len(tok)
    eos_id     = tok.eos_token_id or 0
    end_id     = tok.convert_tokens_to_ids("<|end|>") or eos_id
    user_id    = tok.convert_tokens_to_ids("<|user|>") or eos_id
    stop_ids   = {eos_id, end_id, user_id}

    logger.info(
        "[tokenizer] HuggingFace '%s'  vocab=%d  eos=%d  stop_ids=%s",
        name, real_vocab, eos_id, sorted(stop_ids),
    )

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def safe_decode(ids: list[int]) -> str:
        try:
            return tok.decode([i for i in ids if i < real_vocab])
        except Exception:
            return ""

    return tok, encode, safe_decode, stop_ids, real_vocab


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _load_checkpoint(
    ckpt_path: str,
    device: str,
    preset_override: Optional[str] = None,
) -> tuple[GPT, ModelConfig, dict]:
    """
    Load a Senku GPT checkpoint.

    Config priority:
      1. preset_override (MODEL_PRESET env var / explicit arg)
      2. 'config' dict stored in the checkpoint
      3. 'train_cfg' fallback
      4. ACTIVE_CONFIG default from config.py
    """
    import os, dataclasses
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at '{ckpt_path}'.")

    logger.info("[checkpoint] Loading  %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Resolve ModelConfig ────────────────────────────────────────────
    if preset_override and preset_override in MODEL_PRESETS:
        cfg = MODEL_PRESETS[preset_override]
        logger.info("[checkpoint] Using preset override: %s", preset_override)

    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
        cfg = ModelConfig(**filtered)
        logger.info(
            "[checkpoint] Restored ModelConfig from checkpoint "
            "(filtered %d non-model keys).",
            len(ckpt["config"]) - len(filtered),
        )

    elif "train_cfg" in ckpt:
        tc = ckpt["train_cfg"]
        cfg = ModelConfig(
            vocab_size  = tc.get("vocab_size",  50304),
            d_model     = tc.get("d_model",     768),
            n_layers    = tc.get("n_layers",    12),
            n_heads     = tc.get("n_heads",     12),
            max_seq_len = tc.get("max_seq_len", 1024),
        )
        logger.info("[checkpoint] Inferred ModelConfig from train_cfg.")

    else:
        cfg = ACTIVE_CONFIG
        logger.info("[checkpoint] Using ACTIVE_CONFIG from config.py.")

    # ── Build & load model ─────────────────────────────────────────────
    model = GPT(cfg).to(device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    # Strip DDP / torch.compile prefixes if present
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s%s", len(missing), missing[:5],
                       "…" if len(missing) > 5 else "")
    if unexpected:
        logger.warning("Unexpected keys (%d): %s%s", len(unexpected), unexpected[:5],
                       "…" if len(unexpected) > 5 else "")

    model.eval()

    n_params    = model.parameter_count()
    n_params_ne = model.parameter_count(non_embedding=True)
    step        = ckpt.get("step", "?")
    val_loss    = ckpt.get("val_loss", None)
    tokens_seen = ckpt.get("tokens_seen", None)

    logger.info(
        "[checkpoint] step=%s  params=%.1fM  (non-embed=%.1fM)  "
        "val_loss=%s  tokens_seen=%s",
        step, n_params / 1e6, n_params_ne / 1e6,
        val_loss,
        f"{tokens_seen/1e9:.2f}B" if tokens_seen else "?",
    )

    return model, cfg, ckpt


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor,          # (vocab_size,) — 1-D, already on device
    recent_ids: list[int],
    penalty: float,
    window: int,
) -> torch.Tensor:
    """
    Vectorised repetition penalty (CTRL formulation).
    Avoids a Python loop + per-scalar device sync.
    """
    if penalty == 1.0 or not recent_ids:
        return logits

    ids = torch.tensor(
        list(set(recent_ids[-window:])),
        dtype=torch.long,
        device=logits.device,
    )
    ids = ids[ids < logits.size(0)]   # safety: ignore out-of-vocab ids
    if ids.numel() == 0:
        return logits

    logits = logits.clone()
    scores = logits[ids]
    logits[ids] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    k         = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs                 = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask               = cum_probs > top_p
    remove_mask[..., 1:]      = remove_mask[..., :-1].clone()
    remove_mask[..., 0]       = False
    sorted_logits[remove_mask] = float("-inf")
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Singleton-style inference engine.  Create once; call generate() or
    generate_stream() many times.

    A threading.Lock prevents concurrent requests from corrupting the
    RangeFlow anchor state shared across generation steps.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._device    = self._pick_device()
        self._model: GPT | None   = None
        self._cfg:   ModelConfig | None = None
        self._encode_fn = None
        self._decode_fn = None
        self._stop_ids: set[int] = set()
        self._real_vocab: int    = TOKENIZER_REAL_VOCAB
        self._pad_mask: torch.Tensor | None = None
        logger.info("InferenceEngine created — device: %s", self._device.upper())

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Build model, load checkpoint, and set up tokenizer.  Call once at startup."""
        logger.info("=" * 60)
        logger.info("🚀  RangeFlow Inference Engine — loading Senku model")
        logger.info("    Device     : %s", self._device.upper())
        logger.info("    Checkpoint : %s", CHECKPOINT_PATH)
        logger.info("=" * 60)

        # ── Tokenizer ─────────────────────────────────────────────────
        _tok, encode, decode, stop_ids, real_vocab = _load_tokenizer(TOKENIZER_NAME)
        self._encode_fn  = encode
        self._decode_fn  = decode
        self._stop_ids   = stop_ids
        self._real_vocab = real_vocab

        # ── Model ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        model, cfg, _ckpt = _load_checkpoint(CHECKPOINT_PATH, self._device)
        self._model = model
        self._cfg   = cfg
        logger.info(
            "Model ready in %.2f s — %.1f M parameters",
            time.perf_counter() - t0,
            model.parameter_count() / 1e6,
        )

        # ── Padding-slot logit mask ────────────────────────────────────
        # Slots real_vocab … cfg.vocab_size-1 are untrained padding rows;
        # mask them to -inf so they're never sampled.
        pad_mask = torch.zeros(cfg.vocab_size, device=self._device)
        if real_vocab < cfg.vocab_size:
            pad_mask[real_vocab:] = float("-inf")
        self._pad_mask = pad_mask

        logger.info("✅  Engine ready — accepting requests")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: GenerationRequest) -> dict:
        """Blocking generation — returns dict with response, tokens_generated, elapsed_ms, device."""
        if self._model is None:
            raise RuntimeError("Model is not loaded.  Call load_model() first.")
        with self._lock:
            return self._run_generation(req, stream_queue=None, loop=None)

    def generate_stream(
        self,
        req: GenerationRequest,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Streaming generation — pushes tokens into *queue* as they are produced.
        Queue message format:  ("token", str) | ("done", int) | ("error", str)
        Called from a ThreadPoolExecutor (blocks).
        """
        if self._model is None:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", "Model not loaded"))
            return
        with self._lock:
            self._run_generation(req, stream_queue=queue, loop=loop)

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _run_generation(
        self,
        req: GenerationRequest,
        stream_queue: Optional[asyncio.Queue],
        loop: Optional[asyncio.AbstractEventLoop],
    ) -> dict:
        """
        Two-phase RangeFlow autoregressive generation.

        Phase A (capture)  — one forward pass over the full prompt to record
                             the K/V bounding-box anchors in every attention layer.
        Phase B (guard)    — token-by-token generation with K/V clamped to the
                             epsilon-expanded anchor box.
        """
        streaming = stream_queue is not None

        logger.info("-" * 50)
        logger.info("📥  Generation [%s]", "stream" if streaming else "full")
        logger.info("    Prompt     : %r  (%d chars)", req.prompt[:80], len(req.prompt))
        logger.info("    max_tokens=%d  temp=%.3f  top_k=%d  top_p=%.3f  "
                    "rep=%.3f  ε=%.3f",
                    req.max_tokens, req.temperature, req.top_k,
                    req.top_p, req.repetition_penalty, req.range_epsilon)
        logger.info("-" * 50)

        t_start = time.perf_counter()
        model   = self._model
        cfg     = self._cfg
        device  = self._device

        # Determine autocast dtype
        # bf16 on CUDA, fp16 on MPS, fp32 on CPU
        if device == "cuda":
            amp_dtype = torch.bfloat16
        elif device == "mps":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32

        # ── Encode prompt (wrapped in ChatML format) ───────────────────
        chat_prompt = _build_chat_prompt(req.prompt, system=req.system)
        prompt_ids  = self._encode_fn(chat_prompt)
        if not prompt_ids:
            logger.warning("Empty prompt — returning empty response")
            if streaming:
                loop.call_soon_threadsafe(stream_queue.put_nowait, ("done", 0))
            return {"response": "", "tokens_generated": 0, "elapsed_ms": 0.0, "device": device}

        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        logger.info("    Prompt tokens: %d", tokens.size(1))

        generated_ids: list[int] = []

        # ── Phase A: CAPTURE ──────────────────────────────────────────
        logger.debug("Phase A: capture — establishing RangeFlow anchor")
        model.clear_anchors()
        model.set_epsilon(req.range_epsilon)
        model.set_range_mode("capture")

        with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                enabled=(device in ("cuda", "mps"))):
            model(tokens)

        logger.debug("Phase A complete — anchor captured across %d layers", cfg.n_layers)

        # ── Phase B: GUARD — autoregressive generation ────────────────
        logger.debug("Phase B: guard — autoregressive generation begins")
        model.set_range_mode("guard")

        idx = tokens   # running context window

        for step in range(req.max_tokens):

            # Crop to max_seq_len
            idx_cond = idx[:, -cfg.max_seq_len:]

            # Forward
            with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                    enabled=(device in ("cuda", "mps"))):
                logits_all = model(idx_cond)

            logits = logits_all[0, -1, :].float()   # (vocab_size,) — fp32 for stability

            # Mask padding slots
            logits = logits + self._pad_mask

            # Sampling stack
            logits = _apply_repetition_penalty(
                logits,
                recent_ids = prompt_ids + generated_ids,
                penalty    = req.repetition_penalty,
                window     = 64,
            )
            if req.temperature != 1.0:
                logits = logits / max(req.temperature, 1e-8)
            if req.top_k > 0:
                logits = _top_k_filter(logits.unsqueeze(0), req.top_k).squeeze(0)
            if req.top_p < 1.0:
                logits = _top_p_filter(logits.unsqueeze(0), req.top_p).squeeze(0)

            probs    = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()

            # Stop check
            if token_id in self._stop_ids or token_id >= self._real_vocab:
                logger.debug("    [step %d] stop token %d — halting", step, token_id)
                break

            generated_ids.append(token_id)
            idx = torch.cat(
                [idx, torch.tensor([[token_id]], dtype=torch.long, device=device)],
                dim=1,
            )

            # Stream the new token immediately
            if streaming:
                # Incremental decode to handle multi-byte UTF-8 split across tokens
                current_text = self._decode_fn(generated_ids)
                prev_text    = self._decode_fn(generated_ids[:-1]) if len(generated_ids) > 1 else ""
                new_chars    = current_text[len(prev_text):]
                if new_chars:
                    loop.call_soon_threadsafe(stream_queue.put_nowait, ("token", new_chars))

            if (step + 1) % 50 == 0:
                logger.debug("    [step %d] %d tokens so far", step + 1, len(generated_ids))

        # ── Wrap up ───────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "✅  Generation complete — %d tokens in %.1f ms  (%.1f tok/s)",
            len(generated_ids), elapsed_ms,
            len(generated_ids) / max(elapsed_ms / 1000, 1e-6),
        )

        model.set_range_mode("standard")

        if streaming:
            loop.call_soon_threadsafe(stream_queue.put_nowait, ("done", len(generated_ids)))
            return {}

        response_text = self._decode_fn(generated_ids)
        logger.info("    Response preview: %r …", response_text[:80])
        return {
            "response":         response_text,
            "tokens_generated": len(generated_ids),
            "elapsed_ms":       round(elapsed_ms, 2),
            "device":           device,
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device


# Module-level singleton
engine = InferenceEngine()