"""
chat.py — Interactive inference for the SFT'd checkpoint
============================================================
Usage:
    python chat.py --ckpt checkpoints_sft_v2/best.pt
    python chat.py --ckpt checkpoints_sft_v2/best.pt --device cpu
    python chat.py --ckpt checkpoints_sft_v2/best.pt --temperature 0.8 --top-p 0.9 --repetition-penalty 1.15

Must live in the same directory as model.py, sft_tokenizer.py, and
checkpoint.py (or those must be importable) — this script builds on
top of them rather than duplicating their logic, exactly like
sft_train.py does.

── What this adds on top of sft_train.py's generate_chat() ──────────
generate_chat() in sft_train.py re-runs the WHOLE sequence through the
model on every single new token (no cache) — fine for periodic
training-time sampling, way too slow for an interactive chat. This
script instead:

  1. Prefills the prompt once (one forward pass over all prompt
     tokens), then decodes one new token at a time, feeding only that
     ONE new token through the model per step. Past K/V are read from
     a preallocated cache instead of being recomputed. This turns
     per-token generation cost from O(context_length) into O(1)
     (ignoring attention's own O(context_length) cost, which is
     unavoidable and exactly what flash attention below speeds up).

  2. The cache PERSISTS across turns in a conversation, not just
     within one response — a follow-up message only prefills the new
     user turn's tokens, not the entire chat history again.

  3. Explicitly requests the flash-attention SDPA backend (falls back
     automatically if unavailable, e.g. on CPU).

  4. Strips the checkpoint down to weights-only at load time — a
     checkpoint saved by checkpoint.py also contains a full AdamW
     optimizer state (2x the model size) plus train_cfg/step/etc,
     none of which inference needs. We keep the "config" (to rebuild
     ModelConfig) and "model" (the state_dict) and drop the rest
     immediately so it can be garbage-collected.

  5. Logs metrics (prefill time, decode throughput, context length,
     memory) after every response, both to stdout and to a .jsonl
     file.

── Why RoPE + KV cache need a `start_pos` argument ───────────────────
model.py's RotaryEmbedding always rotates assuming the sequence you
hand it starts at position 0 (`cos_cached[:, :, :seq_len, :]`). That's
correct for a full-context forward pass but WRONG for cached decoding:
if the cache already holds 47 tokens and we're feeding token #48, that
token's rotation angle must correspond to absolute position 47, not 0.
CachedAttention below slices `cos_cached`/`sin_cached` starting at
`start_pos` instead of 0 to fix this — the single silent bug that
would otherwise make cached generation diverge from uncached
generation after the first turn.

── Why is_causal=True still works during decode (T=1, S=cache_len+1) ─
PyTorch's scaled_dot_product_attention, when query length L and key
length S differ, applies the causal mask bottom-right-aligned: query
position i may attend to key position j iff j <= i + (S - L). With
L=1 that reduces to "attend to everything in the cache so far" — which
is exactly correct, since there IS nothing after this new query token
yet. So the same `is_causal=True` call used for prefill (L == S) is
also correct, unchanged, for single-token decode. No separate branch
needed.
"""

import os
import gc
import json
import math
import time
import argparse
import contextlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    ModelConfig, GPT, TransformerBlock, MultiHeadAttention,
    _apply_rotary,
)
from sft_tokenizer import (
    build_chat_tokenizer, render_prompt_for_generation, decode_response,
    ROLE_TOKEN_ID, EOT_ID, END_ID, PAD_ID, MIN_VOCAB_SIZE_REQUIRED,
)

# Prefer the flash-attention SDPA backend when available (torch>=2.1).
# Falls back to the older global-flag API on older torch, and to
# "just let SDPA pick" if neither context-manager API exists.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def flash_attention_context():
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                             SDPBackend.EFFICIENT_ATTENTION,
                             SDPBackend.MATH])
except ImportError:
    try:
        def flash_attention_context():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            )
    except AttributeError:
        def flash_attention_context():
            return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────
# KV cache
# ─────────────────────────────────────────────────────────────

class KVCache:
    """
    One preallocated (n_layers, B, n_heads, max_seq_len, d_k) buffer for
    keys and one for values. `length` is the number of real (non-empty)
    positions currently written — shared across all layers, since every
    layer processes the same set of token positions in a given forward
    call.
    """

    def __init__(self, n_layers: int, batch_size: int, n_heads: int, d_k: int,
                 max_seq_len: int, device: torch.device, dtype: torch.dtype):
        self.k = torch.zeros(n_layers, batch_size, n_heads, max_seq_len, d_k,
                              device=device, dtype=dtype)
        self.v = torch.zeros(n_layers, batch_size, n_heads, max_seq_len, d_k,
                              device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.length = 0

    def capacity_left(self) -> int:
        return self.max_seq_len - self.length

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write this layer's new K/V at [length : length+T], return the
        full K/V (including everything written so far) for attention."""
        T = k.shape[2]
        s, e = self.length, self.length + T
        if e > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: tried to write up to position {e}, "
                f"capacity is {self.max_seq_len}. Caller should have "
                f"trimmed context first — see fit_context()."
            )
        self.k[layer_idx, :, :, s:e, :] = k
        self.v[layer_idx, :, :, s:e, :] = v
        return self.k[layer_idx, :, :, :e, :], self.v[layer_idx, :, :, :e, :]

    def advance(self, T: int):
        """Call once per forward pass (not per layer) after all layers
        have written their K/V for this chunk."""
        self.length += T

    def reset(self):
        self.length = 0


# ─────────────────────────────────────────────────────────────
# Cache-aware model classes
# ─────────────────────────────────────────────────────────────
# These subclass model.py's classes and override ONLY forward(), so the
# parameter names/shapes in state_dict() are byte-for-byte identical to
# the originals — a checkpoint trained with the plain GPT class loads
# into these with a normal strict load_state_dict(), no key remapping.

class CachedAttention(MultiHeadAttention):
    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None,
                layer_idx: int = 0, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        # RoPE at the *absolute* position of these tokens, not position 0
        # — see module docstring for why this matters under caching.
        cos = self.rope.cos_cached[:, :, start_pos:start_pos + T, :]
        sin = self.rope.sin_cached[:, :, start_pos:start_pos + T, :]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.write(layer_idx, k, v)

        S = k.shape[2]
        # NOTE: empirically verified (see test in the accompanying
        # message) that F.scaled_dot_product_attention's is_causal=True
        # is NOT reliably bottom-right-aligned when query length T !=
        # key length S on every backend/torch build — relying on it
        # silently produced wrong attention for cached decoding. So we
        # only use is_causal=True in the one case it's unambiguous
        # (T == S), use the mask-free fast path when it's trivially
        # correct anyway (T == 1: a single new query has nothing in its
        # "future" to mask), and fall back to an explicit mask only for
        # the remaining case (T > 1 appended onto a non-empty cache).
        if T == S:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            offset = S - T   # bottom-right-aligned causal mask
            mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class CachedBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.attn = CachedAttention(cfg)   # replace with the cache-aware attention

    def forward(self, x, kv_cache=None, layer_idx=0, start_pos=0):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos)
        x = x + self.ff(self.ln2(x))
        return x


class CachedGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # Rebuild blocks using CachedBlock; module names (blocks.i.attn.*
        # etc.) are identical to the base GPT, so load_state_dict below
        # still works with strict=True.
        self.blocks = nn.ModuleList([CachedBlock(cfg) for _ in range(cfg.n_layers)])
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, idx: torch.Tensor, kv_cache: Optional[KVCache] = None,
                start_pos: int = 0):
        x = self.drop(self.token_embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i, start_pos=start_pos)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if kv_cache is not None:
            kv_cache.advance(idx.shape[1])
        return logits

    def new_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        d_k = self.cfg.d_model // self.cfg.n_heads
        return KVCache(self.cfg.n_layers, batch_size, self.cfg.n_heads, d_k,
                        self.cfg.max_seq_len, device, dtype)


# ─────────────────────────────────────────────────────────────
# Checkpoint loading — strip optimizer state / training bookkeeping
# ─────────────────────────────────────────────────────────────

def load_inference_model(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> CachedGPT:
    print(f"Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in raw or "config" not in raw:
        raise SystemExit(
            f"'{ckpt_path}' doesn't look like a checkpoint.py-produced file "
            f"(missing 'model' and/or 'config' keys). Found keys: {list(raw.keys())}"
        )

    kept = {"model": raw["model"], "config": raw["config"]}
    dropped = [k for k in raw.keys() if k not in kept]
    dropped_step = raw.get("step")
    dropped_val_loss = raw.get("val_loss")
    del raw
    gc.collect()

    print(f"  Kept   : model weights + config")
    print(f"  Dropped: {dropped}  (optimizer state, train_cfg, step, tokens_seen, ...)")
    if dropped_step is not None:
        print(f"  (checkpoint was step={dropped_step}, val_loss={dropped_val_loss})")

    model_cfg = ModelConfig(**kept["config"])
    if model_cfg.vocab_size < MIN_VOCAB_SIZE_REQUIRED:
        raise SystemExit(
            f"model_cfg.vocab_size={model_cfg.vocab_size} < {MIN_VOCAB_SIZE_REQUIRED} "
            f"required for the chat special tokens — this checkpoint predates SFT "
            f"or was never given room for them."
        )

    model = CachedGPT(model_cfg)
    model.load_state_dict(kept["model"], strict=True)
    del kept
    gc.collect()

    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = model.count_params(non_embedding=True)
    print(f"  Architecture: d_model={model_cfg.d_model} n_layers={model_cfg.n_layers} "
          f"n_heads={model_cfg.n_heads} max_seq_len={model_cfg.max_seq_len}")
    print(f"  Params (non-embedding): {n_params/1e6:.1f}M")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model


# ─────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,           # (1, vocab_size)
    generated_ids: List[int],
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
) -> int:
    logits = logits.clone()

    # Rows 50262.. (up to vocab_size) were never trained during
    # pretraining OR SFT — never sample them (see sft_train.py's
    # generate_chat, which applies the same mask).
    logits[..., MIN_VOCAB_SIZE_REQUIRED:] = float("-inf")

    if repetition_penalty != 1.0 and generated_ids:
        uniq = torch.tensor(sorted(set(generated_ids)), device=logits.device)
        vals = logits[..., uniq]
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        logits[..., uniq] = vals

    logits = logits / max(temperature, 1e-5)

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1, None]] = float("-inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        remove = cum_probs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    return next_tok.item()


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class ResponseMetrics:
    turn: int
    prompt_tokens: int          # tokens prefilled this turn (new turn only, not full history)
    context_tokens_after: int   # total tokens now sitting in the KV cache
    generated_tokens: int
    prefill_time_s: float
    decode_time_s: float
    total_time_s: float
    prefill_tok_per_s: float
    decode_tok_per_s: float
    stopped_reason: str         # "end_token" | "max_new_tokens"
    gpu_mem_allocated_mb: Optional[float] = None
    gpu_mem_reserved_mb: Optional[float] = None

    def print_summary(self):
        print(
            f"  [metrics] turn={self.turn} "
            f"prefill={self.prefill_tok_per_s:6.1f} tok/s ({self.prompt_tokens} tok, {self.prefill_time_s*1000:.0f} ms)  "
            f"decode={self.decode_tok_per_s:6.1f} tok/s ({self.generated_tokens} tok, {self.decode_time_s*1000:.0f} ms)  "
            f"ctx={self.context_tokens_after}  stop={self.stopped_reason}"
            + (f"  gpu_mem={self.gpu_mem_allocated_mb:.0f}MB" if self.gpu_mem_allocated_mb is not None else "")
        )


class MetricsLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

    def log(self, m: ResponseMetrics):
        m.print_summary()
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({**m.__dict__, "ts": time.time()}) + "\n")


# ─────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────

def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def generate_reply(
    model: CachedGPT,
    enc,
    kv_cache: KVCache,
    device: torch.device,
    new_token_ids: List[int],   # tokens NOT yet in the cache (new turn's prompt suffix, ending in the open assistant tag)
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
    stream: bool = True,
):
    """
    Prefills `new_token_ids` against the existing kv_cache (which already
    holds the prior conversation), then decodes one token at a time.
    Returns (reply_text, generated_ids_including_end, metrics_partial).
    """
    prompt_len = len(new_token_ids)
    if kv_cache.capacity_left() < prompt_len + 1:
        raise RuntimeError("Not enough KV cache capacity for this turn — caller must trim context first.")

    t0 = time.perf_counter()

    # ── Prefill ──────────────────────────────────────────────
    x = torch.tensor([new_token_ids], dtype=torch.long, device=device)
    start_pos = kv_cache.length
    with torch.no_grad(), amp_context(device), flash_attention_context():
        logits = model(x, kv_cache=kv_cache, start_pos=start_pos)
    next_logits = logits[:, -1, :].float()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_time = t1 - t0

    # ── Decode loop ──────────────────────────────────────────
    generated: List[int] = []
    stopped_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        next_id = sample_next_token(next_logits, generated, temperature, top_k, top_p, repetition_penalty)
        generated.append(next_id)

        if stream:
            piece = enc.decode([next_id]) if next_id != END_ID else ""
            print(piece, end="", flush=True)

        if next_id == END_ID:
            stopped_reason = "end_token"
            break

        if kv_cache.capacity_left() < 1:
            stopped_reason = "context_full"
            break

        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        start_pos = kv_cache.length
        with torch.no_grad(), amp_context(device), flash_attention_context():
            logits = model(x, kv_cache=kv_cache, start_pos=start_pos)
        next_logits = logits[:, -1, :].float()

    else:
        # Loop exhausted max_new_tokens without a natural END — the
        # cache still needs an END token flushed into it so the next
        # turn's template stays well-formed (see module docstring).
        pass

    if stopped_reason != "end_token" and kv_cache.capacity_left() >= 1:
        x = torch.tensor([[END_ID]], dtype=torch.long, device=device)
        start_pos = kv_cache.length
        with torch.no_grad(), amp_context(device), flash_attention_context():
            model(x, kv_cache=kv_cache, start_pos=start_pos)   # write END's own K/V, discard logits
        generated.append(END_ID)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    decode_time = t2 - t1

    reply_text = decode_response(generated, enc)

    metrics = dict(
        prompt_tokens=prompt_len,
        context_tokens_after=kv_cache.length,
        generated_tokens=len(generated),
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        total_time_s=t2 - t0,
        prefill_tok_per_s=prompt_len / prefill_time if prefill_time > 0 else float("inf"),
        decode_tok_per_s=(len(generated) - 1) / decode_time if decode_time > 0 and len(generated) > 1 else 0.0,
        stopped_reason=stopped_reason,
    )
    return reply_text, generated, metrics


# ─────────────────────────────────────────────────────────────
# Context-length management
# ─────────────────────────────────────────────────────────────

def render_history_prefix(messages: List[Dict], enc) -> List[int]:
    """
    Tokens covering the full history up to and including the last
    message's <|end|> — i.e. render_prompt_for_generation()'s output
    minus its trailing open "<|assistant|>" tag (that tag is added
    separately, right before generation, in main()).
    """
    full = render_prompt_for_generation(messages, enc)
    return full[:-1]


def fit_and_rebuild_if_needed(
    model: CachedGPT, enc, messages: List[Dict], kv_cache: KVCache,
    device: torch.device, max_new_tokens: int,
) -> bool:
    """
    If the full history (re-rendered) plus room for a max_new_tokens
    reply would overflow the model's max_seq_len, drop the oldest
    user/assistant pair(s) (keeping any leading system message) and
    rebuild the KV cache from scratch by reprefilling `messages` up to
    (not including) the trailing open assistant tag.

    Returns True if a rebuild happened (kv_cache is now populated up
    through the last message's <|end|> and the caller should NOT
    prefill history again — just feed the open assistant tag). Returns
    False if nothing needed trimming (kv_cache untouched).
    """
    max_seq_len = model.cfg.max_seq_len
    has_system = bool(messages) and messages[0]["role"] == "system"
    floor = 2 if has_system else 1   # minimum messages we refuse to drop below

    trimmed = False
    while len(messages) > floor:
        needed = len(render_prompt_for_generation(messages, enc)) + max_new_tokens
        if needed <= max_seq_len:
            break
        drop_at = 1 if has_system else 0   # drop the oldest user/assistant pair
        del messages[drop_at:drop_at + 2]
        trimmed = True

    if not trimmed:
        return False

    print(f"  [context] trimmed oldest turn(s) to fit max_seq_len={max_seq_len}")
    kv_cache.reset()
    history_ids = render_history_prefix(messages, enc)
    if len(history_ids) > kv_cache.capacity_left():
        raise RuntimeError(
            "Even after trimming to the minimum, history + max_new_tokens "
            "doesn't fit in max_seq_len. Lower --max-new-tokens or shorten "
            "your system prompt / message."
        )
    x = torch.tensor([history_ids], dtype=torch.long, device=device)
    with torch.no_grad(), amp_context(device), flash_attention_context():
        model(x, kv_cache=kv_cache, start_pos=0)

    return True


# ─────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Interactive chat with an SFT'd checkpoint.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to an SFT checkpoint (.pt)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 disables nucleus sampling")
    p.add_argument("--repetition-penalty", type=float, default=1.15, help="1.0 disables it")
    p.add_argument("--system", type=str, default=None, help="Optional system prompt")
    p.add_argument("--metrics-file", type=str, default="chat_metrics.jsonl")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        dtype = torch.bfloat16
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        dtype = torch.float32
        print("Running on CPU — flash attention is CUDA-only, falling back to "
              "the math/mem-efficient SDPA backend (still correct, just slower).")

    model = load_inference_model(args.ckpt, device, dtype)
    enc = build_chat_tokenizer()
    metrics_logger = MetricsLogger(args.metrics_file)

    messages: List[Dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    kv_cache = model.new_kv_cache(batch_size=1, device=device, dtype=dtype)
    # True once the kv_cache covers history up through a previous
    # assistant turn's <|end|> — i.e. we're in the normal steady state
    # where a new turn only needs an incremental (user + assistant-tag)
    # suffix prefilled, not the whole history again.
    steady_state = False

    print("\nType your message and press Enter. Commands: /reset  /system <text>  /exit\n")

    turn = 0
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            kv_cache.reset()
            steady_state = False
            print("  [conversation reset]")
            continue
        if user_input.startswith("/system "):
            new_system = user_input[len("/system "):].strip()
            messages = [{"role": "system", "content": new_system}]
            kv_cache.reset()
            steady_state = False
            print("  [system prompt set, conversation reset]")
            continue

        messages.append({"role": "user", "content": user_input})

        # Trim + rebuild the cache from scratch if this turn (+ a full
        # reply) wouldn't fit in max_seq_len. A rebuild prefills history
        # itself, so it also puts us back in the "need assistant tag
        # only" case below.
        did_rebuild = fit_and_rebuild_if_needed(
            model, enc, messages, kv_cache, device, args.max_new_tokens,
        )
        if did_rebuild:
            steady_state = False

        if not steady_state:
            if kv_cache.length == 0:
                # First-ever turn (or just /reset): prefill the whole
                # history ourselves (fit_and_rebuild_if_needed had
                # nothing to trim, so it left the cache untouched).
                history_ids = render_history_prefix(messages, enc)
                x = torch.tensor([history_ids], dtype=torch.long, device=device)
                with torch.no_grad(), amp_context(device), flash_attention_context():
                    model(x, kv_cache=kv_cache, start_pos=0)
            new_token_ids = [ROLE_TOKEN_ID["assistant"]]
        else:
            # Steady state: cache already covers everything through the
            # previous assistant turn's <|end|> — only prefill the new
            # user turn + the open assistant tag.
            content_ids = enc.encode_ordinary(user_input.strip())
            new_token_ids = [ROLE_TOKEN_ID["user"]] + content_ids + [END_ID] + [ROLE_TOKEN_ID["assistant"]]

        print("assistant> ", end="", flush=True)
        reply_text, generated_ids, m = generate_reply(
            model, enc, kv_cache, device, new_token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=(args.top_k or None),
            top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        )
        print()  # newline after streamed tokens

        messages.append({"role": "assistant", "content": reply_text})
        steady_state = True

        gpu_alloc = gpu_reserved = None
        if device.type == "cuda":
            gpu_alloc = torch.cuda.memory_allocated() / 1e6
            gpu_reserved = torch.cuda.memory_reserved() / 1e6

        rm = ResponseMetrics(turn=turn, gpu_mem_allocated_mb=gpu_alloc, gpu_mem_reserved_mb=gpu_reserved, **m)
        metrics_logger.log(rm)
        turn += 1


if __name__ == "__main__":
    main()