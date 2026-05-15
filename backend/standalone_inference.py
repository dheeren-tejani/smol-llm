"""
inference.py — Interactive inference for Senku (pretrained + SFT)
==================================================================

Features:
  • Auto-loads checkpoint (model weights + config)
  • tiktoken gpt2 tokenizer (matches training)
  • ChatML prompt format matching SFT training:
      <|system|>\n{system}\n
      <|user|>\n{user}\n
      <|assistant|>\n
  • Token-by-token streaming (typewriter effect)
  • Repetition penalty (applied at logit level, pre-softmax)
  • TTFT  — Time To First Token
  • Total generation time
  • Tokens/sec throughput
  • Stops cleanly at <|end|> token

Usage:
    # Interactive chat (SFT model)
    python inference.py --ckpt checkpoints/sft/best.pt

    # Single prompt (pretrain model, raw completion)
    python inference.py --ckpt checkpoints/best.pt --prompt "The meaning of life is"

    # All options:
    python inference.py \
        --ckpt   checkpoints/sft/best.pt \
        --preset gpt2-small \
        --max-new-tokens 512 \
        --temperature 0.7 \
        --top-k 50 \
        --rep-penalty 1.3 \
        --rep-window 64 \
        --dtype bf16 \
        --device cuda \
        --no-chat
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── Local imports ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ════════════════════════════════════════════════════════════
# 1.  ANSI colour helpers  (graceful fallback on Windows)
# ════════════════════════════════════════════════════════════

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def dim(t):    return _c("2",    t)
def bold(t):   return _c("1",    t)
def green(t):  return _c("32",   t)
def cyan(t):   return _c("36",   t)
def yellow(t): return _c("33",   t)
def magenta(t):return _c("35",   t)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:  int   = 50304
    d_model:     int   = 768
    n_layers:    int   = 12
    n_heads:     int   = 12
    d_ff:        int   = 0          # auto-set to int(d_model * 2.67) if 0
    max_seq_len: int   = 1024
    dropout:     float = 0.0        # 0.0 during pretraining

    def __post_init__(self):
        if self.d_ff == 0:
            # SwiGLU canonical ratio; round to multiple of 64 for efficiency
            raw = int(self.d_model * 8 / 3)
            self.d_ff = (raw + 63) // 64 * 64

        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        
# ─────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for stability, cast back to original dtype (bf16)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight).type_as(x)


# ─────────────────────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t    = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# ─────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward
# ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU(x) = (W1·x ⊙ silu(W3·x)) · W2"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff,    d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ─────────────────────────────────────────────────────────────
# Multi-Head Attention (Flash SDPA + RoPE)
# ─────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k     = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.rope      = RotaryEmbedding(self.d_k, max_seq_len=cfg.max_seq_len)

        # ── RangeFlow state ──────────────────────────────────────
        self.mode    = "standard"   # "capture" | "guard" | "standard"
        self.epsilon = 0.2          # tightness of the constraint

        self.register_buffer("anchor_k_min", None)
        self.register_buffer("anchor_k_max", None)
        self.register_buffer("anchor_v_min", None)
        self.register_buffer("anchor_v_max", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        # Apply RoPE to q and k
        q, k = self.rope(q, k, T)

        # ── RangeFlow logic (applied AFTER RoPE) ─────────────────
        if self.mode == "capture":
            self.anchor_k_min = k.min(dim=2, keepdim=True)[0].detach()
            self.anchor_k_max = k.max(dim=2, keepdim=True)[0].detach()
            self.anchor_v_min = v.min(dim=2, keepdim=True)[0].detach()
            self.anchor_v_max = v.max(dim=2, keepdim=True)[0].detach()

        elif self.mode == "guard" and self.anchor_k_min is not None:
            # Build epsilon-expanded intervals for current token
            k_min_curr = k - self.epsilon
            k_max_curr = k + self.epsilon
            v_min_curr = v - self.epsilon
            v_max_curr = v + self.epsilon

            # Intersect with anchor bounding box
            valid_k_min = torch.max(k_min_curr, self.anchor_k_min)
            valid_k_max = torch.min(k_max_curr, self.anchor_k_max)
            valid_v_min = torch.max(v_min_curr, self.anchor_v_min)
            valid_v_max = torch.min(v_max_curr, self.anchor_v_max)

            # Fix empty intervals (when intersection is empty, collapse to lower bound)
            valid_k_min = torch.min(valid_k_min, valid_k_max)
            valid_v_min = torch.min(valid_v_min, valid_v_max)

            # Clamp k and v into valid range
            k = torch.clamp(k, min=valid_k_min, max=valid_k_max)
            v = torch.clamp(v, min=valid_v_min, max=valid_v_max)

        # Flash Attention — O(N) memory, fused kernel via SDPA
        attn_drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=attn_drop,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff   = SwiGLU(cfg.d_model, cfg.d_ff)
        self.ln1  = RMSNorm(cfg.d_model)
        self.ln2  = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────
# Full GPT Model
# ─────────────────────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop        = nn.Dropout(cfg.dropout)
        self.blocks      = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final    = RMSNorm(cfg.d_model)
        self.lm_head     = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # GPT-2 style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.token_embed.weight.numel()
        return n
    
    def set_range_mode(self, mode: str, epsilon: float = None):
        """Set RangeFlow mode on all attention layers.

        mode: "capture" | "guard" | "standard"
        epsilon: optional, overrides the constraint tightness (lower = stricter)
        """
        for block in self.blocks:
            block.attn.mode = mode
            if epsilon is not None:
                block.attn.epsilon = epsilon

    def forward(
        self,
        idx: torch.Tensor,                    # (B, T)
        targets: Optional[torch.Tensor] = None,  # (B, T)
    ):
        x = self.drop(self.token_embed(idx))  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────
# Preset model sizes (easy to swap in train.py)
# ─────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    # name        : (d_model, n_layers, n_heads)
    "gpt2-small"  : ModelConfig(d_model=768,  n_layers=12, n_heads=12),
    "gpt2-medium" : ModelConfig(d_model=1024, n_layers=24, n_heads=16),
    "gpt2-large"  : ModelConfig(d_model=1280, n_layers=36, n_heads=20),
    "gpt2-xl"     : ModelConfig(d_model=1600, n_layers=48, n_heads=25),
    "llama-1b"    : ModelConfig(d_model=2048, n_layers=16, n_heads=16, max_seq_len=2048),
    "llama-7b"    : ModelConfig(d_model=4096, n_layers=32, n_heads=32, max_seq_len=2048),
}


# ════════════════════════════════════════════════════════════
# 2.  Tokenizer  (tiktoken gpt2, matching training setup)
# ════════════════════════════════════════════════════════════

# The SFT trainer uses tiktoken gpt2 (vocab=50257) but the model has
# vocab_size=50304 (padded to a multiple of 64 for efficiency).
# Slots 50257–50303 are padding — the model can accidentally sample them
# and tiktoken will crash with KeyError.
#
# Strategy:
#   1. Mask those slots to -inf in logits before sampling (prevent sampling).
#   2. Wrap decode_fn to gracefully handle any that slip through (safety net).
#   3. Treat any id >= tiktoken_vocab as a stop signal (they mean nothing).

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>", "<|eot|>", "</|user|"]

# The SFT trainer encodes these as plain text with tiktoken gpt2.
# Pre-compute their multi-token encodings so we know what to stop on.
# "<|end|>" → tiktoken encodes as e.g. [27, 91, 437, 91, 29] — we stop
# on the full suffix sequence by watching for the final token of each.
_STOP_STRINGS = ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]


def load_tokenizer(name: str = "gpt2"):
    """
    Returns (encode_fn, safe_decode_fn, eos_id, stop_ids_set, tiktoken_vocab_size).

    safe_decode_fn  — never raises; maps unknown ids to "" silently.
    stop_ids_set    — set of int ids that should halt generation.
    tiktoken_vocab_size — the actual vocab size of the tokenizer (50257 for gpt2).
                          Used to mask padding logits in generate_stream.
    """

    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    eos_id  = tok.eos_token_id or 0
    tikv    = len(tok)
    end_id  = tok.convert_tokens_to_ids("<|end|>") or eos_id
    user_id = tok.convert_tokens_to_ids("<|user|>") or eos_id
    stop_ids = {eos_id, end_id, user_id}

    print(f"{dim('[tokenizer]')} HuggingFace '{name}'  "
          f"vocab={tikv}  eos={eos_id}  stop_ids={sorted(stop_ids)}")
    
    def safe_decode(ids: List[int]) -> str:
        try:
            return tok.decode([i for i in ids if i < tikv])
        except Exception:
            return ""
        
    return (
        lambda t: tok.encode(t, add_special_tokens=False),
        safe_decode,
        eos_id,
        stop_ids,
        tikv,
    )

# ════════════════════════════════════════════════════════════
# 3.  Checkpoint loader
# ════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    ckpt_path: str,
    preset_override: Optional[str],
    device: torch.device,
) -> Tuple[GPT, ModelConfig, dict]:
    """
    Load GPT model from a checkpoint file.

    Config priority:
      1. --preset CLI flag (explicit override)
      2. 'config' key stored in checkpoint  (most common)
      3. 'train_cfg' key as fallback
    """
    print(f"{dim('[checkpoint]')} Loading  {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Resolve ModelConfig ────────────────────────────────────
    if preset_override:
        if preset_override not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_override}'. "
                f"Choose from: {list(MODEL_PRESETS)}"
            )
        cfg = MODEL_PRESETS[preset_override]
        print(f"{dim('[checkpoint]')} Using preset override: {preset_override}")

    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        # The SFT trainer stores vars(args) — the full argparse namespace —
        # under "config", so we must filter to only the fields ModelConfig knows.
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
        cfg = ModelConfig(**filtered)
        print(f"{dim('[checkpoint]')} Restored ModelConfig from checkpoint "
              f"(filtered {len(ckpt['config']) - len(filtered)} non-model keys).")

    else:
        # Best-effort: scan train_cfg for model fields
        tc = ckpt.get("train_cfg", {})
        cfg = ModelConfig(
            vocab_size  = tc.get("vocab_size",  50304),
            d_model     = tc.get("d_model",     768),
            n_layers    = tc.get("n_layers",    12),
            n_heads     = tc.get("n_heads",     12),
            max_seq_len = tc.get("max_seq_len", 1024),
        )
        print(f"{dim('[checkpoint]')} Inferred ModelConfig from train_cfg.")

    # ── Build & load model ────────────────────────────────────
    model = GPT(cfg).to(device)
    state = ckpt["model"]

    # Strip DDP / compile prefixes if present
    fixed = {}
    for k, v in state.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        fixed[k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"{yellow('[warn]')} Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"{yellow('[warn]')} Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()

    # Report
    n_params = model.count_params(non_embedding=False)
    n_params_ne = model.count_params(non_embedding=True)
    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", None)
    tokens_seen = ckpt.get("tokens_seen", None)

    print(f"{dim('[checkpoint]')} step={step}  "
          f"params={n_params/1e6:.1f}M  (non-embed={n_params_ne/1e6:.1f}M)  "
          f"val_loss={val_loss}  "
          f"tokens_seen={f'{tokens_seen/1e9:.2f}B' if tokens_seen else '?'}")

    return model, cfg, ckpt


# ════════════════════════════════════════════════════════════
# 4.  Repetition penalty
# ════════════════════════════════════════════════════════════

def apply_repetition_penalty(
    logits: torch.Tensor,          # (vocab_size,)
    generated_ids: List[int],
    penalty: float,
    window: int,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits (in-place safe via clone).

    For each token id that already appears in the recent `window` tokens:
      - If logit > 0  →  divide by penalty   (reduce probability)
      - If logit < 0  →  multiply by penalty  (push even more negative)

    This is the standard formulation from:
      "CTRL: A Conditional Transformer Language Model for Controllable Generation"
      (Keskar et al., 2019) — same as used by HF Transformers.

    penalty = 1.0  → no effect
    penalty = 1.3  → moderate suppression
    penalty = 1.5+ → strong suppression (can hurt coherence)
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    # Look at only the most recent `window` tokens
    recent = set(generated_ids[-window:])
    logits = logits.clone()

    for token_id in recent:
        if token_id < logits.size(0):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty

    return logits


# ════════════════════════════════════════════════════════════
# 5.  Core streaming generate
# ════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_stream(
    model: GPT,
    prompt_ids: List[int],
    encode_fn,
    decode_fn,                          # must be the safe_decode wrapper
    max_new_tokens: int      = 512,
    temperature: float       = 0.8,
    top_k: Optional[int]     = 50,
    rep_penalty: float       = 1.3,
    rep_window: int          = 64,
    stop_ids: Optional[set]  = None,    # set of int ids that halt generation
    tiktoken_vocab: int      = 50257,   # real tokenizer vocab size (not padded)
    device: torch.device     = torch.device("cpu"),
    amp_dtype: torch.dtype   = torch.bfloat16,
    range_epsilon: float     = None,
) -> dict:
    """
    Streaming token-by-token generator.

    Prints tokens to stdout as they are produced (typewriter effect).
    Returns a stats dict:
      {
        "ttft_ms"       : float,   # Time To First Token (ms)
        "total_ms"      : float,   # Total wall-clock generation time (ms)
        "tokens_gen"    : int,     # Number of tokens generated
        "tok_per_sec"   : float,   # Average throughput (tok/s)
        "stop_reason"   : str,     # "max_tokens" | "stop_token"
        "output_text"   : str,     # Full decoded output string
      }
    """
    stop_ids = stop_ids or set()

    # Build a logit-mask tensor to permanently zero-out:
    #   • Padding slots  (tiktoken_vocab … model_vocab_size-1)
    #     These are untrained embedding rows — the model produces garbage
    #     for them and tiktoken crashes trying to decode them.
    #   • The EOS / stop ids themselves (prevent them from ever being
    #     sampled as content — we want them only as stop signals).
    model_vocab = model.cfg.vocab_size
    pad_mask = torch.zeros(model_vocab, device=device)
    if tiktoken_vocab < model_vocab:
        pad_mask[tiktoken_vocab:] = float("-inf")   # kill padding range
    # for sid in stop_ids:
    #     if sid < model_vocab:
    #         pad_mask[sid] = float("-inf")           # stop tokens → never sample
# 
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids: List[int] = []

    # ── RangeFlow: Phase A — CAPTURE ─────────────────────────
    if range_epsilon is not None:
        model.set_range_mode("capture", epsilon=range_epsilon)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
            model(idx)   # single forward pass to set anchor buffers
        model.set_range_mode("guard")
    # ─────────────────────────────────────────────────────────

    t_start     = time.perf_counter()
    t_first     = None
    stop_reason = "max_tokens"

    for i in range(max_new_tokens):

        # ── Context window crop ────────────────────────────────
        idx_cond = idx[:, -model.cfg.max_seq_len:]

        # ── Forward pass ──────────────────────────────────────
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
            logits, _ = model(idx_cond)

        logits = logits[0, -1, :].float()   # (vocab_size,)  → fp32 for stability

        # ── Mask padding / stop slots ─────────────────────────
        logits = logits + pad_mask           # adds -inf to forbidden ids

        # ── Repetition penalty ────────────────────────────────
        logits = apply_repetition_penalty(
            logits,
            generated_ids = prompt_ids + generated_ids,
            penalty       = rep_penalty,
            window        = rep_window,
        )

        # ── Temperature ───────────────────────────────────────
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        # ── Top-k filtering ───────────────────────────────────
        if top_k is not None and top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_vals[-1]] = float("-inf")

        # ── Sample ────────────────────────────────────────────
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        # ── TTFT (measured right after first forward + sample) ─
        if i == 0:
            t_first = time.perf_counter()

        # ── Stop check ────────────────────────────────────────
        # Double-safety: even with the pad_mask, check explicitly.
        if next_id in stop_ids or next_id >= tiktoken_vocab:
            stop_reason = "stop_token"
            break

        # ── Decode & stream ───────────────────────────────────
        generated_ids.append(next_id)

        # Incremental decode: diff full decoded string vs previous to get
        # only the newly added characters.  This avoids partial-UTF8 issues
        # when a multi-byte character is split across token boundaries.
        current_text = decode_fn(generated_ids)
        prev_text    = decode_fn(generated_ids[:-1]) if len(generated_ids) > 1 else ""
        new_chars    = current_text[len(prev_text):]

        if new_chars:
            print(new_chars, end="", flush=True)

        # ── Extend context ────────────────────────────────────
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

    # ── Final newline ─────────────────────────────────────────
    print()

    model.set_range_mode("standard")

    t_end    = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    ttft_ms  = (t_first - t_start) * 1000 if t_first else 0.0
    n_gen    = len(generated_ids)
    tps      = n_gen / ((t_end - t_start) + 1e-9)

    return {
        "ttft_ms"    : ttft_ms,
        "total_ms"   : total_ms,
        "tokens_gen" : n_gen,
        "tok_per_sec": tps,
        "stop_reason": stop_reason,
        "output_text": decode_fn(generated_ids),
    }


# ════════════════════════════════════════════════════════════
# 6.  Prompt builders
# ════════════════════════════════════════════════════════════

DEFAULT_SYSTEM = (
    "You are Senku, a helpful and knowledgeable AI assistant. "
    "Answer clearly and concisely."
)

def build_chat_prompt(
    user_message: str,
    encode_fn,
    system: str = DEFAULT_SYSTEM,
    history: Optional[List[dict]] = None,
) -> List[int]:
    """
    Build a ChatML-formatted prompt matching the SFT training format:

        <|system|>
        {system}
        <|user|>
        {user}
        <|assistant|>

    The model generates from the open <|assistant|> tag onward.
    Optionally prepend conversation history for multi-turn.
    """
    parts = []

    # System turn
    parts.append(f"<|system|>\n{system}\n")

    # History turns (if any)
    if history:
        for turn in history:
            role    = turn["role"]
            content = turn["content"]
            if role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|end|>\n")
            else:
                parts.append(f"<|{role}|>\n{content}\n")

    # Current user turn + open assistant tag
    parts.append(f"<|user|>\n{user_message}\n<|assistant|>\n")

    full_prompt = "".join(parts)
    return encode_fn(full_prompt)


def build_raw_prompt(text: str, encode_fn) -> List[int]:
    """Plain text prompt for pretrained (non-SFT) model."""
    return encode_fn(text)


# ════════════════════════════════════════════════════════════
# 7.  Stats printer
# ════════════════════════════════════════════════════════════

def print_stats(stats: dict):
    ttft    = stats["ttft_ms"]
    total   = stats["total_ms"]
    n       = stats["tokens_gen"]
    tps     = stats["tok_per_sec"]
    reason  = stats["stop_reason"]

    sep = dim("─" * 60)
    print(f"\n{sep}")
    print(
        f"  {bold('TTFT')}          {cyan(f'{ttft:.1f} ms')}   "
        f"({dim('time to first token')})"
    )
    print(
        f"  {bold('Total time')}    {cyan(f'{total:.1f} ms')}   "
        f"({dim(f'{total/1000:.2f} s')})"
    )
    print(
        f"  {bold('Tokens gen')}    {cyan(str(n))}"
    )
    print(
        f"  {bold('Throughput')}    {cyan(f'{tps:.1f} tok/s')}"
    )
    print(
        f"  {bold('Stop reason')}   {yellow(reason)}"
    )
    print(f"{sep}\n")


# ════════════════════════════════════════════════════════════
# 8.  Interactive chat loop
# ════════════════════════════════════════════════════════════

def run_chat(args, model, cfg, encode_fn, decode_fn, stop_ids, tiktoken_vocab, device, amp_dtype):
    """
    Multi-turn interactive chat session.
    Type 'exit', 'quit', or Ctrl-C to quit.
    Type '/reset' to clear conversation history.
    Type '/system <text>' to change the system prompt.
    """
    history: List[dict] = []
    system_prompt = args.system

    print(f"\n{bold('Senku')} {dim('— Interactive Chat')}")
    print(dim(f"  Model    : {args.ckpt}"))
    print(dim(f"  Device   : {device}  |  dtype: {args.dtype}"))
    print(dim(f"  Sampling : temp={args.temperature}  top_k={args.top_k}  "
              f"rep_penalty={args.rep_penalty}  rep_window={args.rep_window}"))
    print(dim(f"  Commands : /reset  /system <text>  exit"))
    print(dim("─" * 60))

    while True:
        # ── Input ─────────────────────────────────────────────
        try:
            user_input = input(f"\n{green('You')} › ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{dim('Goodbye.')}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print(dim("Goodbye."))
            break

        if user_input.lower() == "/reset":
            history.clear()
            print(dim("[history cleared]"))
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            print(dim(f"[system prompt updated: {system_prompt[:80]}]"))
            continue

        # ── Build prompt ──────────────────────────────────────
        prompt_ids = build_chat_prompt(
            user_message = user_input,
            encode_fn    = encode_fn,
            system       = system_prompt,
            history      = history,
        )

        print(f"\n{magenta('Senku')} › ", end="", flush=True)

        # ── Generate ──────────────────────────────────────────
        stats = generate_stream(
            model           = model,
            prompt_ids      = prompt_ids,
            encode_fn       = encode_fn,
            decode_fn       = decode_fn,
            max_new_tokens  = args.max_new_tokens,
            temperature     = args.temperature,
            top_k           = args.top_k,
            rep_penalty     = args.rep_penalty,
            rep_window      = args.rep_window,
            stop_ids        = stop_ids,
            tiktoken_vocab  = tiktoken_vocab,
            device          = device,
            amp_dtype       = amp_dtype,
            range_epsilon = args.range_epsilon,
        )

        # ── Stats ─────────────────────────────────────────────
        print_stats(stats)

        # ── Update history ────────────────────────────────────
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": stats["output_text"]})


# ════════════════════════════════════════════════════════════
# 9.  Single-shot mode (--prompt flag)
# ════════════════════════════════════════════════════════════

def run_single(args, model, cfg, encode_fn, decode_fn, stop_ids, tiktoken_vocab, device, amp_dtype):
    """
    Single prompt → generate → print stats.  Non-interactive.
    """
    if args.no_chat:
        prompt_ids = build_raw_prompt(args.prompt, encode_fn)
        mode_label = "raw completion"
    else:
        prompt_ids = build_chat_prompt(args.prompt, encode_fn, system=args.system)
        mode_label = "chat"

    print(f"\n{bold('Prompt')} ({mode_label}, {len(prompt_ids)} tokens):")
    print(dim(f"  {args.prompt}"))
    print(f"\n{bold('Output')}:")

    stats = generate_stream(
        model           = model,
        prompt_ids      = prompt_ids,
        encode_fn       = encode_fn,
        decode_fn       = decode_fn,
        max_new_tokens  = args.max_new_tokens,
        temperature     = args.temperature,
        top_k           = args.top_k,
        rep_penalty     = args.rep_penalty,
        rep_window      = args.rep_window,
        stop_ids        = stop_ids,
        tiktoken_vocab  = tiktoken_vocab,
        device          = device,
        amp_dtype       = amp_dtype,
        range_epsilon = args.range_epsilon,
    )

    print_stats(stats)


# ════════════════════════════════════════════════════════════
# 10.  CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference script for Senku — pretrained + SFT GPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoint ────────────────────────────────────────────
    p.add_argument("--ckpt",   required=True,
                   help="Path to checkpoint file (.pt)")
    p.add_argument("--preset", default=None,
                   help=f"Override ModelConfig preset: {list(MODEL_PRESETS)}")

    # ── Tokenizer ─────────────────────────────────────────────
    p.add_argument("--tokenizer", default="gpt2",
                   help="tiktoken encoding or HF model id")

    # ── Mode ──────────────────────────────────────────────────
    p.add_argument("--prompt",  default=None,
                   help="Run single prompt and exit (omit for interactive mode)")
    p.add_argument("--no-chat", action="store_true",
                   help="Raw completion mode (no ChatML wrapping, for pretrain ckpts)")
    p.add_argument("--system",  default=DEFAULT_SYSTEM,
                   help="System prompt for chat mode")

    # ── Sampling ──────────────────────────────────────────────
    p.add_argument("--max-new-tokens", type=int,   default=512,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature",    type=float, default=0.7,
                   help="Sampling temperature (0 = greedy, 1 = raw)")
    p.add_argument("--top-k",          type=int,   default=50,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--rep-penalty",    type=float, default=1.3,
                   help="Repetition penalty ≥ 1.0 (1.0 = off, 1.3 = moderate)")
    p.add_argument("--rep-window",     type=int,   default=64,
                   help="Look-back window (tokens) for repetition penalty")
    p.add_argument("--range-epsilon", type=float, default=None,
               help="RangeFlow epsilon (e.g. 0.05=strict, 0.2=loose). Omit to disable.")

    # ── Hardware ──────────────────────────────────────────────
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu  (auto-detect if omitted)")
    p.add_argument("--dtype",  default="bf16",
                   choices=["bf16", "fp16", "fp32"],
                   help="Autocast dtype (bf16 matches training)")

    return p.parse_args()


# ════════════════════════════════════════════════════════════
# 11.  Entry point
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    amp_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = amp_dtype_map[args.dtype]

    # bf16 autocast is not supported on CPU; silently fall back to fp32
    if device.type == "cpu" and amp_dtype != torch.float32:
        print(f"{yellow('[warn]')} CPU device: forcing fp32 (bf16/fp16 not supported).")
        amp_dtype = torch.float32

    print(f"{dim('[device]')} {device}  |  amp_dtype={args.dtype}")

    # ── Tokenizer ─────────────────────────────────────────────
    encode_fn, decode_fn, eos_id, stop_ids, tiktoken_vocab = load_tokenizer(args.tokenizer)

    # ── Model ─────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model, cfg, ckpt_meta = load_model_from_checkpoint(
        ckpt_path       = args.ckpt,
        preset_override = args.preset,
        device          = device,
    )

    # top_k=0 means disabled
    if args.top_k == 0:
        args.top_k = None

    # ── Run ───────────────────────────────────────────────────
    if args.prompt is not None:
        run_single(args, model, cfg, encode_fn, decode_fn, stop_ids, tiktoken_vocab, device, amp_dtype)
    else:
        run_chat(args, model, cfg, encode_fn, decode_fn, stop_ids, tiktoken_vocab, device, amp_dtype)


if __name__ == "__main__":
    main()