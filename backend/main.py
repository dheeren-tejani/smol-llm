"""
main.py — FastAPI application for the Noir Whisper backend.

Endpoints
---------
GET  /health             Liveness + readiness probe.
POST /generate           Full response at once (kept for compatibility).
POST /generate/stream    Server-Sent Events — streams one token at a time.
GET  /config             Exposes current model & server configuration.
"""

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from config import (
    ACTIVE_CONFIG,
    CHECKPOINT_PATH,
    CORS_ORIGINS,
    GenerationDefaults,
    SERVER_HOST,
    SERVER_PORT,
    TOKENIZER_NAME,
)
from inference import DEFAULT_SYSTEM, GenerationRequest, engine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("noir_whisper.api")


# ---------------------------------------------------------------------------
# Lifespan — model loaded once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Noir Whisper Backend — starting up")
    logger.info("=" * 60)

    try:
        engine.load_model()
        logger.info("🟢  Model ready — server is accepting requests")
    except RuntimeError as exc:
        logger.critical("🔴  Failed to load model: %s", exc)
        logger.critical("    The server will start but /generate will return 503.")
    except Exception as exc:
        logger.critical("🔴  Unexpected error during startup: %s", exc, exc_info=True)

    yield

    logger.info("Noir Whisper Backend — shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Noir Whisper — RangeFlow LLM Backend",
    description=(
        "Local inference server for the Noir Whisper chat UI. "
        "Powered by a RangeFlow-constrained Senku GPT model."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    system: str = Field(default=DEFAULT_SYSTEM, max_length=2048)

    max_tokens:         int   = Field(default=GenerationDefaults.MAX_TOKENS,          ge=1,    le=ACTIVE_CONFIG.max_seq_len)
    temperature:        float = Field(default=GenerationDefaults.TEMPERATURE,         ge=0.01, le=5.0)
    top_p:              float = Field(default=GenerationDefaults.TOP_P,               ge=0.0,  le=1.0)
    top_k:              int   = Field(default=GenerationDefaults.TOP_K,               ge=1,    le=200)
    repetition_penalty: float = Field(default=GenerationDefaults.REPETITION_PENALTY,  ge=1.0,  le=3.0)
    range_epsilon:      float = Field(default=GenerationDefaults.RANGE_EPSILON,       ge=0.0,  le=2.0)

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("system")
    @classmethod
    def strip_system(cls, v: str) -> str:
        return v.strip() or DEFAULT_SYSTEM


class GenerateResponse(BaseModel):
    response:         str
    tokens_generated: int
    elapsed_ms:       float
    device:           str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    checkpoint:   str
    uptime_s:     float


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

_server_start = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0      = time.perf_counter()
    resp    = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("%-6s %-30s → %d  (%.1f ms)",
                request.method, request.url.path, resp.status_code, elapsed)
    return resp


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s",
                 request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error — check backend logs."},
    )


# ---------------------------------------------------------------------------
# Routes — meta
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    ready = engine.is_ready
    body  = HealthResponse(
        status       = "ok" if ready else "degraded",
        model_loaded = ready,
        device       = engine.device,
        checkpoint   = CHECKPOINT_PATH,
        uptime_s     = round(time.time() - _server_start, 1),
    )
    return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())


@app.get("/config", tags=["meta"])
async def get_config():
    cfg = ACTIVE_CONFIG
    return {
        "model": {
            "vocab_size":  cfg.vocab_size,
            "d_model":     cfg.d_model,
            "n_layers":    cfg.n_layers,
            "n_heads":     cfg.n_heads,
            "d_ff":        cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
        },
        "tokenizer": TOKENIZER_NAME,
        "defaults": {
            "max_tokens":         GenerationDefaults.MAX_TOKENS,
            "temperature":        GenerationDefaults.TEMPERATURE,
            "top_p":              GenerationDefaults.TOP_P,
            "top_k":              GenerationDefaults.TOP_K,
            "repetition_penalty": GenerationDefaults.REPETITION_PENALTY,
            "range_epsilon":      GenerationDefaults.RANGE_EPSILON,
        },
        "checkpoint": CHECKPOINT_PATH,
        "device":     engine.device,
    }


# ---------------------------------------------------------------------------
# Routes — inference
# ---------------------------------------------------------------------------

def _make_req(body: GenerateRequest) -> GenerationRequest:
    return GenerationRequest(
        prompt             = body.prompt,
        system             = body.system,
        max_tokens         = body.max_tokens,
        temperature        = body.temperature,
        top_p              = body.top_p,
        top_k              = body.top_k,
        repetition_penalty = body.repetition_penalty,
        range_epsilon      = body.range_epsilon,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(body: GenerateRequest):
    """Non-streaming — waits for full generation then returns the whole response."""
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = engine.generate(_make_req(body))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected generation error.")

    return GenerateResponse(**result)


@app.post("/generate/stream", tags=["inference"])
async def generate_stream(body: GenerateRequest):
    """
    Streaming generation via Server-Sent Events (SSE).

    Tokens are emitted one by one as JSON:
        data: {"token": " hello"}\\n\\n
        data: {"token": " world"}\\n\\n
        data: {"done": true, "tokens_generated": 42}\\n\\n

    On error:
        data: {"error": "something went wrong"}\\n\\n
    """
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    req  = _make_req(body)
    loop = asyncio.get_event_loop()

    queue: asyncio.Queue = asyncio.Queue()

    def run_in_thread():
        try:
            engine.generate_stream(req, queue, loop)
        except Exception as exc:
            logger.error("Streaming thread error: %s", exc, exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    async def event_stream():
        loop.run_in_executor(None, run_in_thread)
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                yield f"data: {json.dumps({'token': payload})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'done': True, 'tokens_generated': payload})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': payload})}\n\n"
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Uvicorn on %s:%d", SERVER_HOST, SERVER_PORT)
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="warning",
        workers=1,
    )