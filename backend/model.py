"""
model.py — RangeFlow-aware GPT architecture (Senku backend).

Architecture (matches the trained Senku checkpoint):
  • RMSNorm           — pre-norm, fp32 upcast for numerical stability
  • RoPE              — rotary positional embeddings applied to Q and K
  • SwiGLU            — feed-forward with gated activation (d_ff = 8/3 · d_model)
  • Flash SDPA        — fused causal attention via F.scaled_dot_product_attention
  • RangeFlow         — K/V bounding-box constraint for on-topic generation

RangeFlow modes (set per attention layer via GPT.set_range_mode):
  • "standard" — plain causal self-attention, no constraints.
  • "capture"  — one forward pass over the prompt; records the min/max
                  bounding box of K and V (per head, keepdim on seq axis)
                  as anchors.
  • "guard"    — each new token's K/V is clamped into the anchor box
                  expanded by ±epsilon, steering generation back toward
                  the prompt's semantic neighbourhood.

range_epsilon controls tightness:
  ≈ 0.05 → strict (stays very close to the prompt).
  ≈ 0.20 → loose (more creative, still grounded).
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
        # Upcast to fp32 for stability, cast back to original dtype
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU(x) = (W1·x ⊙ silu(W3·x)) · W2  — no bias, matches training."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff,    d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# RangeFlow-aware Multi-Head Attention (Flash SDPA + RoPE)
# ---------------------------------------------------------------------------

class RangeAwareAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and optional RangeFlow constraints.

    Anchor buffers (anchor_k_min / _max / anchor_v_min / _max) are registered
    so they move with .to(device), but start as None and are populated only
    during a 'capture' pass.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k     = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.rope      = RotaryEmbedding(self.d_k, max_seq_len=cfg.max_seq_len)

        # RangeFlow runtime state
        self.mode:    str   = "standard"
        self.epsilon: float = 0.1

        # Anchor buffers — None until a 'capture' pass runs
        self.register_buffer("anchor_k_min", None)
        self.register_buffer("anchor_k_max", None)
        self.register_buffer("anchor_v_min", None)
        self.register_buffer("anchor_v_max", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, T)

        # ── RangeFlow ──────────────────────────────────────────────────
        if self.mode == "capture":
            self.anchor_k_min = k.min(dim=2, keepdim=True)[0].detach()
            self.anchor_k_max = k.max(dim=2, keepdim=True)[0].detach()
            self.anchor_v_min = v.min(dim=2, keepdim=True)[0].detach()
            self.anchor_v_max = v.max(dim=2, keepdim=True)[0].detach()

        elif self.mode == "guard" and self.anchor_k_min is not None:
            k_lo = torch.max(k - self.epsilon, self.anchor_k_min)
            k_hi = torch.min(k + self.epsilon, self.anchor_k_max)
            v_lo = torch.max(v - self.epsilon, self.anchor_v_min)
            v_hi = torch.min(v + self.epsilon, self.anchor_v_max)

            # Guard against degenerate (empty) intervals
            k_lo = torch.min(k_lo, k_hi)
            v_lo = torch.min(v_lo, v_hi)

            k = torch.max(k_lo, torch.min(k, k_hi))
            v = torch.max(v_lo, torch.min(v, v_hi))
        # ──────────────────────────────────────────────────────────────

        # Flash Attention — fused causal kernel via SDPA
        attn_drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=attn_drop,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)

    def clear_anchor(self) -> None:
        """Reset anchor buffers so a fresh capture can be run."""
        self.anchor_k_min = None
        self.anchor_k_max = None
        self.anchor_v_min = None
        self.anchor_v_max = None


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT-style decoder-only language model with RangeFlow-aware attention.
    Architecture matches the Senku training checkpoint.
    """

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

        logger.debug(
            "GPT initialised — vocab=%d, d_model=%d, layers=%d, heads=%d, d_ff=%d",
            cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.token_embed(idx))  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_final(x))  # (B, T, vocab_size)

    # ── RangeFlow helpers ──────────────────────────────────────────────

    def set_range_mode(self, mode: str) -> None:
        """Propagate RangeFlow mode to every attention layer."""
        for block in self.blocks:
            block.attn.mode = mode

    def set_epsilon(self, epsilon: float) -> None:
        """Set the RangeFlow epsilon on every attention layer."""
        for block in self.blocks:
            block.attn.epsilon = epsilon

    def clear_anchors(self) -> None:
        """Clear all captured anchors — call before each new prompt."""
        for block in self.blocks:
            block.attn.clear_anchor()

    # ── Utility ───────────────────────────────────────────────────────

    def parameter_count(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.token_embed.weight.numel()
        return n


# Alias for any code that still imports GPTModel by the old name
GPTModel = GPT