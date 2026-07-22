"""
config.py — Central configuration for the backend.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size:  int   = 50304
    d_model:     int   = 768
    n_layers:    int   = 12
    n_heads:     int   = 12
    d_ff:        int   = 0
    max_seq_len: int   = 1024
    dropout:     float = 0.0

    def __post_init__(self):
        if self.d_ff == 0:
            raw = int(self.d_model * 8 / 3)
            self.d_ff = (raw + 63) // 64 * 64
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


MODEL_PRESETS: dict[str, ModelConfig] = {
    "smol-124m": ModelConfig(d_model=768, n_layers=12, n_heads=12),
}
ACTIVE_CONFIG: ModelConfig = MODEL_PRESETS["smol-124m"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = "/app/models/best_sft.pt"


# ---------------------------------------------------------------------------
# Precision — fp32 only, everywhere.
# ---------------------------------------------------------------------------

FORCE_FP32: bool = True


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

SERVER_HOST: str = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT: int = int(os.environ.get("PORT", "8000"))

CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "https://smol-llm.netlify.app",
]


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

@dataclass
class GenerationDefaults:
    MAX_TOKENS:         int   = 256
    TEMPERATURE:        float = 0.7
    TOP_P:              float = 0.9
    TOP_K:              int   = 40
    REPETITION_PENALTY: float = 1.15
    RANGE_EPSILON:      float = 0.1

GenerationDefaults = GenerationDefaults()


# ---------------------------------------------------------------------------
# Rate limiting / abuse protection — two tiers: burst (per-minute) and
# budget (per-day), both per client IP, in-process sliding windows.
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))
RATE_LIMIT_PER_DAY:    int = int(os.environ.get("RATE_LIMIT_PER_DAY", "20"))
MAX_CONCURRENT_GENERATIONS: int = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "3"))

# Only True if you run a reverse proxy YOU control that overwrites
# X-Forwarded-For itself — otherwise clients can spoof it and dodge limits.
TRUST_FORWARDED_FOR: bool = os.environ.get("TRUST_FORWARDED_FOR", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

LOG_DIR: str = os.environ.get("LOG_DIR", "logs")

LOGGING_NOTICE: str = (
    "Conversations may be logged to help prevent abuse or exploitation."
)


# ---------------------------------------------------------------------------
# Request authentication — AES-256-GCM shared secret. Empty AUTH_SECRET_KEY
# or AUTH_KEY_VALUE disables auth entirely (useful for local dev before the
# frontend crypto is wired up).
# ---------------------------------------------------------------------------

AUTH_SECRET_KEY: str = os.environ.get("AUTH_SECRET_KEY", "")   # base64, must decode to 32 bytes
AUTH_KEY_VALUE:  str = os.environ.get("AUTH_KEY_VALUE", "")     # shared "identity" string
AUTH_TOKEN_MAX_AGE_SECONDS: float = float(os.environ.get("AUTH_TOKEN_MAX_AGE_SECONDS", "30"))
AUTH_ENABLED: bool = bool(AUTH_SECRET_KEY and AUTH_KEY_VALUE)