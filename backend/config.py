"""
config.py — Central configuration for the backend.
Updated to match the Senku model architecture (RoPE + RMSNorm + SwiGLU).
"""

import os
import math
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size:  int   = 50304          # padded to multiple of 64 for efficiency
    d_model:     int   = 768
    n_layers:    int   = 12
    n_heads:     int   = 12
    d_ff:        int   = 0              # auto-computed from d_model if 0 (SwiGLU ratio)
    max_seq_len: int   = 1024
    dropout:     float = 0.0

    def __post_init__(self):
        if self.d_ff == 0:
            # SwiGLU canonical ratio 8/3, rounded up to nearest multiple of 64
            raw = int(self.d_model * 8 / 3)
            self.d_ff = (raw + 63) // 64 * 64

        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


# Available preset sizes — pass --preset to override at CLI / startup
MODEL_PRESETS: dict[str, ModelConfig] = {
    "gpt2-small"  : ModelConfig(d_model=768,  n_layers=12, n_heads=12),
    "gpt2-medium" : ModelConfig(d_model=1024, n_layers=24, n_heads=16),
    "gpt2-large"  : ModelConfig(d_model=1280, n_layers=36, n_heads=20),
    "gpt2-xl"     : ModelConfig(d_model=1600, n_layers=48, n_heads=25),
    "llama-1b"    : ModelConfig(d_model=2048, n_layers=16, n_heads=16, max_seq_len=2048),
    "llama-7b"    : ModelConfig(d_model=4096, n_layers=32, n_heads=32, max_seq_len=2048),
}

# Active config — override via PRESET env var or leave as default (gpt2-small)
_preset_name = os.environ.get("MODEL_PRESET", "gpt2-small")
if _preset_name not in MODEL_PRESETS:
    raise ValueError(
        f"Unknown MODEL_PRESET '{_preset_name}'. "
        f"Choose from: {list(MODEL_PRESETS)}"
    )
ACTIVE_CONFIG: ModelConfig = MODEL_PRESETS[_preset_name]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Real tiktoken/HF vocab size before padding — slots above this are masked
TOKENIZER_NAME: str = os.environ.get("TOKENIZER_NAME", "gpt2")
TOKENIZER_REAL_VOCAB: int = 50257   # gpt2 base; slots 50257-50303 are padding

# Special tokens added on top of gpt2 base vocab to match SFT training
SPECIAL_TOKENS: list[str] = [
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|end|>", "<|pad|>", "<|eot|>", "</|user|>",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = os.environ.get(
    "CHECKPOINT_PATH",
    "model/llm_model.pt"
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

SERVER_HOST: str = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT: int = int(os.environ.get("PORT", "8000"))

# Allowed CORS origins — frontend dev server + production placeholder
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite default
    "http://localhost:3000",   # CRA / alternative
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "https://smol-llm.netlify.app"
]


# ---------------------------------------------------------------------------
# Generation defaults (used as fallback when client omits a field)
# ---------------------------------------------------------------------------

@dataclass
class GenerationDefaults:
    MAX_TOKENS:         int   = 512
    TEMPERATURE:        float = 0.7
    TOP_P:              float = 0.9
    TOP_K:              int   = 50
    REPETITION_PENALTY: float = 1.1
    RANGE_EPSILON:      float = 0.1

GenerationDefaults = GenerationDefaults()   # singleton instance