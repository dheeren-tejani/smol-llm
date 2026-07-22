"""
model.py — RangeFlow-aware GPT architecture (Smol-lmbackend).
Shared by sft_train.py, standalone_inference.py, and the backend server.

Adds KV-cache support directly into the RangeFlow attention path (rather
than a separate subclass, per standalone_inference.py's approach) so the
backend can decode one token at a time instead of recomputing the whole
sequence every step — this is the O(T^2)->O(T) fix for serving.

RangeFlow semantics are UNCHANGED:
  • "standard" — plain causal self-attention.
  • "capture"  — records min/max K/V bounding box (per head) from
                 whatever chunk is passed through — used once, over the
                 prompt, during prefill.
  • "guard"    — clamps each new chunk's K/V into the anchor box
                 (± epsilon) before it's used for attention AND before
                 it's written into the KV cache.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

logger = logging.getLogger("dheeren's_chat.model")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight).type_as(x)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int, seq_len: int):
        cos = self.cos_cached[:, :, start_pos:start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos:start_pos + seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff,    d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# KV Cache — one preallocated (n_layers, B, n_heads, max_seq_len, d_k) buffer
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self, n_layers: int, batch_size: int, n_heads: int, d_k: int,
                 max_seq_len: int, device, dtype):
        self.k = torch.zeros(n_layers, batch_size, n_heads, max_seq_len, d_k,
                              device=device, dtype=dtype)
        self.v = torch.zeros(n_layers, batch_size, n_heads, max_seq_len, d_k,
                              device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.length = 0

    def capacity_left(self) -> int:
        return self.max_seq_len - self.length

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T = k.shape[2]
        s, e = self.length, self.length + T
        if e > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: tried to write up to position {e}, "
                f"capacity is {self.max_seq_len}."
            )
        self.k[layer_idx, :, :, s:e, :] = k
        self.v[layer_idx, :, :, s:e, :] = v
        return self.k[layer_idx, :, :, :e, :], self.v[layer_idx, :, :, :e, :]

    def advance(self, T: int):
        self.length += T

    def reset(self):
        self.length = 0


# ---------------------------------------------------------------------------
# RangeFlow-aware Multi-Head Attention — RoPE + optional KV cache + RangeFlow
# ---------------------------------------------------------------------------

class RangeAwareAttention(nn.Module):
    """
    Multi-head self-attention with RoPE, an optional external KV cache for
    incremental decoding, and optional RangeFlow K/V clamping.

    Anchor buffers are plain Python attributes (not persistent nn.Buffers —
    they're per-request scratch state, never meant to be checkpointed, so we
    don't want them showing up as "missing keys" on every load).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k     = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope     = RotaryEmbedding(self.d_k, max_seq_len=cfg.max_seq_len)

        # RangeFlow runtime state
        self.mode:    str   = "standard"
        self.epsilon: float = 0.1
        self.anchor_k_min = None
        self.anchor_k_max = None
        self.anchor_v_min = None
        self.anchor_v_max = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: "KVCache | None" = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        # RoPE at the ABSOLUTE position of these tokens — required once a
        # cache is in play (see standalone_inference.py's docstring: with
        # cos_cached sliced from 0 every call, a cached decode step would
        # rotate as if it were position 0, silently diverging from an
        # uncached forward pass after the first cached token).
        q, k = self.rope(q, k, start_pos, T)

        # ── RangeFlow ──────────────────────────────────────────────────
        if self.mode == "capture":
            # Anchor is captured from THIS chunk's raw K/V (the prompt,
            # during prefill) — unchanged from the original semantics.
            self.anchor_k_min = k.min(dim=2, keepdim=True)[0].detach()
            self.anchor_k_max = k.max(dim=2, keepdim=True)[0].detach()
            self.anchor_v_min = v.min(dim=2, keepdim=True)[0].detach()
            self.anchor_v_max = v.max(dim=2, keepdim=True)[0].detach()

        elif self.mode == "guard" and self.anchor_k_min is not None:
            k_lo = torch.max(k - self.epsilon, self.anchor_k_min)
            k_hi = torch.min(k + self.epsilon, self.anchor_k_max)
            v_lo = torch.max(v - self.epsilon, self.anchor_v_min)
            v_hi = torch.min(v + self.epsilon, self.anchor_v_max)

            k_lo = torch.min(k_lo, k_hi)   # guard degenerate intervals
            v_lo = torch.min(v_lo, v_hi)

            k = torch.max(k_lo, torch.min(k, k_hi))
            v = torch.max(v_lo, torch.min(v, v_hi))
        # ──────────────────────────────────────────────────────────────

        # Write (possibly clamped) K/V into the cache, if any, AFTER
        # RangeFlow so the cache always holds what attention actually used.
        if kv_cache is not None:
            k, v = kv_cache.write(layer_idx, k, v)

        S = k.shape[2]
        attn_drop = self.dropout if self.training else 0.0

        # Same bottom-right-aligned-causal-mask reasoning as
        # standalone_inference.py's CachedAttention: is_causal=True is only
        # unambiguous when T == S (full/no-cache forward). T == 1 (normal
        # single-token decode) trivially has nothing to mask. T > 1 appended
        # onto a non-empty cache needs an explicit offset mask.
        if T == S:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_drop, is_causal=True,
            )
        elif T == 1:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_drop, is_causal=False,
            )
        else:
            offset = S - T
            mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=attn_drop, is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)

    def clear_anchor(self) -> None:
        self.anchor_k_min = None
        self.anchor_k_max = None
        self.anchor_v_min = None
        self.anchor_v_max = None


# Alias for anything importing the plain-attention name (e.g. a script that
# expects "MultiHeadAttention") — same class, same state_dict keys either way.
MultiHeadAttention = RangeAwareAttention


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = RangeAwareAttention(cfg)
        self.ff   = SwiGLU(cfg.d_model, cfg.d_ff)
        self.ln1  = RMSNorm(cfg.d_model)
        self.ln2  = RMSNorm(cfg.d_model)

    def forward(self, x, kv_cache=None, layer_idx=0, start_pos=0):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT Model
# ---------------------------------------------------------------------------

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

    def forward(
        self,
        idx: torch.Tensor,
        targets: "torch.Tensor | None" = None,
        kv_cache: "KVCache | None" = None,
        start_pos: int = 0,
    ):
        """Returns (logits, loss). loss is None when targets is None —
        matches sft_train.py / standalone_inference.py's `logits, _ = model(x)`
        calling convention."""
        x = self.drop(self.token_embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i, start_pos=start_pos)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if kv_cache is not None:
            kv_cache.advance(idx.shape[1])

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
            )
        return logits, loss

    # ── RangeFlow helpers ──────────────────────────────────────────────

    def set_range_mode(self, mode: str) -> None:
        for block in self.blocks:
            block.attn.mode = mode

    def set_epsilon(self, epsilon: float) -> None:
        for block in self.blocks:
            block.attn.epsilon = epsilon

    def clear_anchors(self) -> None:
        for block in self.blocks:
            block.attn.clear_anchor()

    def new_kv_cache(self, batch_size: int, device, dtype) -> KVCache:
        d_k = self.cfg.d_model // self.cfg.n_heads
        return KVCache(self.cfg.n_layers, batch_size, self.cfg.n_heads, d_k,
                        self.cfg.max_seq_len, device, dtype)

    # ── Utility ───────────────────────────────────────────────────────

    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.token_embed.weight.numel()
        return n

    # kept for backward compat with any code still calling the old name
    parameter_count = count_params


GPTModel = GPT