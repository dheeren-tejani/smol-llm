"""
config.py — Central configuration for the GGUF backend.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# GGUF Model Runtime Configuration
# ---------------------------------------------------------------------------

GGUF_MODEL_PATH:   str  = os.environ.get("GGUF_MODEL_PATH", "/app/models/model_q8_0.gguf")
GGUF_N_CTX:        int  = int(os.environ.get("GGUF_N_CTX", "1024"))
GGUF_N_THREADS:    int  = int(os.environ.get("GGUF_N_THREADS", "4"))
GGUF_N_GPU_LAYERS: int  = int(os.environ.get("GGUF_N_GPU_LAYERS", "0"))  # 0 for CPU, -1 for all on GPU
GGUF_FLASH_ATTN:   bool = os.environ.get("GGUF_FLASH_ATTN", "true").lower() == "true"
GGUF_TYPE_K:       int  = int(os.environ.get("GGUF_TYPE_K", "2"))        # 2 = Q8_0 KV cache
GGUF_TYPE_V:       int  = int(os.environ.get("GGUF_TYPE_V", "2"))        # 2 = Q8_0 KV cache


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
    "http://localhost:8888/w"
]


# ---------------------------------------------------------------------------
# Generation Defaults (preserves original 0.7 temperature)
# ---------------------------------------------------------------------------

@dataclass
class GenerationDefaults:
    MAX_TOKENS:         int   = 256
    TEMPERATURE:        float = 0.7
    TOP_P:              float = 0.9
    TOP_K:              int   = 40
    REPETITION_PENALTY: float = 1.15
    RANGE_EPSILON:      float = 0.1  # Retained so client payload validation remains compatible

GenerationDefaults = GenerationDefaults()


# ---------------------------------------------------------------------------
# Rate Limiting & Concurrency
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))
RATE_LIMIT_PER_DAY:    int = int(os.environ.get("RATE_LIMIT_PER_DAY", "20"))
MAX_CONCURRENT_GENERATIONS: int = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "3"))
TRUST_FORWARDED_FOR: bool = os.environ.get("TRUST_FORWARDED_FOR", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------

LOG_DIR: str = os.environ.get("LOG_DIR", "logs")
MODAL_LOG_VOLUME_NAME: str = os.environ.get("MODAL_LOG_VOLUME_NAME", "smol-lm-logs")
LOGGING_NOTICE: str = "Conversations may be logged to help prevent abuse or exploitation."


# ---------------------------------------------------------------------------
# Request Authentication
# ---------------------------------------------------------------------------

AUTH_SECRET_KEY: str = os.environ.get("AUTH_SECRET_KEY", "")
AUTH_KEY_VALUE:  str = os.environ.get("AUTH_KEY_VALUE", "")
AUTH_TOKEN_MAX_AGE_SECONDS: float = float(os.environ.get("AUTH_TOKEN_MAX_AGE_SECONDS", "30"))
AUTH_ENABLED: bool = bool(AUTH_SECRET_KEY and AUTH_KEY_VALUE)