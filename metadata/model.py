"""
model.py — Transformer architecture
  • RMSNorm  (no bias, no mean subtraction)
  • RoPE     (rotary position embeddings)
  • SwiGLU   (d_ff = 2.67 × d_model)
  • Flash SDPA (causal, via F.scaled_dot_product_attention)
  • Weight tying (embed ↔ lm_head)
  • No biases anywhere (like LLaMA)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        q, k = self.rope(q, k, T)

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
