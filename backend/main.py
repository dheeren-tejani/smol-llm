"""
main.py — FastAPI application for the backend.
"""

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from config import (
    CHECKPOINT_PATH, CORS_ORIGINS, GenerationDefaults,
    SERVER_HOST, SERVER_PORT, LOG_DIR, LOGGING_NOTICE,
    RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_DAY,
    MAX_CONCURRENT_GENERATIONS, TRUST_FORWARDED_FOR,
    AUTH_SECRET_KEY, AUTH_KEY_VALUE, AUTH_TOKEN_MAX_AGE_SECONDS, AUTH_ENABLED,
)
from inference import DEFAULT_SYSTEM, GenerationRequest, engine
from rate_limit import SlidingWindowRateLimiter, ConcurrencyGuard, get_client_key
from modal_volume_logger import ModalVolumeRequestLogger, RequestLogEntry
from auth import make_verify_auth_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("dheeren's_chat.api")

# ---------------------------------------------------------------------------
# Try to get the Modal Volume object so the logger can .commit() after each
# flush. Falls back to None for local dev (writes still happen locally, just
# without the explicit durability guarantee Modal's commit() provides).
# ---------------------------------------------------------------------------

try:
    from modal_app import log_volume
except ImportError:
    log_volume = None
    logger.warning("[volume_logger] modal_app.log_volume not importable — "
                    "running without Modal Volume persistence (OK for local dev).")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

per_minute_limiter = SlidingWindowRateLimiter(limit=RATE_LIMIT_PER_MINUTE, window_seconds=60.0)
per_day_limiter    = SlidingWindowRateLimiter(limit=RATE_LIMIT_PER_DAY,    window_seconds=86400.0)
concurrency_guard  = ConcurrencyGuard(max_concurrent=MAX_CONCURRENT_GENERATIONS)
req_logger         = ModalVolumeRequestLogger(log_dir=LOG_DIR, volume=log_volume)

verify_auth_token = make_verify_auth_token(
    AUTH_SECRET_KEY, AUTH_KEY_VALUE, AUTH_TOKEN_MAX_AGE_SECONDS, enabled=AUTH_ENABLED,
)

if not AUTH_ENABLED:
    logger.warning(
        "[auth] AUTH_SECRET_KEY / AUTH_KEY_VALUE not set — request auth is DISABLED. "
        "Endpoints are open to anyone with the URL (rate limits still apply)."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Backend — starting up (server binds immediately; "
                "model loads in the background)")

    async def _load():
        try:
            await asyncio.to_thread(engine.load_model)
        except Exception as exc:
            logger.critical("🔴 Failed to load model: %s", exc, exc_info=True)

    async def _prune_loop():
        while True:
            await asyncio.sleep(600)
            dropped_min = per_minute_limiter.prune(max_idle_seconds=3600)
            dropped_day = per_day_limiter.prune(max_idle_seconds=90000)
            if dropped_min or dropped_day:
                logger.info("[rate_limit] pruned idle entries — per_minute=%d per_day=%d",
                            dropped_min, dropped_day)

    load_task  = asyncio.create_task(_load())
    prune_task = asyncio.create_task(_prune_loop())
    await req_logger.start_background_flush()   # <-- was missing: logs would only
                                                  #     flush at 200 buffered entries

    yield

    load_task.cancel()
    prune_task.cancel()
    await req_logger.stop_and_final_flush()      # <-- was missing: flush + commit
                                                  #     whatever's left on shutdown
    logger.info("Backend — shutting down")


app = FastAPI(
    title="LLM Backend",
    version="3.2.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    system: str = Field(default=DEFAULT_SYSTEM, max_length=2048)

    max_tokens:         int   = Field(default=GenerationDefaults.MAX_TOKENS, ge=1, le=1024)
    temperature:        float = Field(default=GenerationDefaults.TEMPERATURE, ge=0.01, le=5.0)
    top_p:              float = Field(default=GenerationDefaults.TOP_P, ge=0.0, le=1.0)
    top_k:              int   = Field(default=GenerationDefaults.TOP_K, ge=1, le=200)
    repetition_penalty: float = Field(default=GenerationDefaults.REPETITION_PENALTY, ge=1.0, le=3.0)
    range_epsilon:      float = Field(default=GenerationDefaults.RANGE_EPSILON, ge=0.0, le=2.0)

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


_server_start = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("%-6s %-30s → %d  (%.1f ms)", request.method, request.url.path, resp.status_code, elapsed)
    return resp


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error — check backend logs."})


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    ready = engine.is_ready
    body = HealthResponse(
        status="ok" if ready else "loading",
        model_loaded=ready,
        device=engine.device,
        checkpoint=CHECKPOINT_PATH,
        uptime_s=round(time.time() - _server_start, 1),
    )
    return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())


@app.get("/config", tags=["meta"])
async def get_config():
    cfg = engine._cfg
    return {
        "model": None if cfg is None else {
            "vocab_size":  cfg.vocab_size,
            "d_model":     cfg.d_model,
            "n_layers":    cfg.n_layers,
            "n_heads":     cfg.n_heads,
            "d_ff":        cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
        },
        "tokenizer": "gpt2_chat (tiktoken, sft_tokenizer.py)",
        "precision": "fp32",
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
        "model_loaded": engine.is_ready,
        "logging_notice": LOGGING_NOTICE,
        "rate_limit": {
            "per_minute": RATE_LIMIT_PER_MINUTE,
            "per_day": RATE_LIMIT_PER_DAY,
            "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
        },
        "auth_enabled": AUTH_ENABLED,
    }


def _make_req(body: GenerateRequest) -> GenerationRequest:
    return GenerationRequest(
        prompt=body.prompt, system=body.system,
        max_tokens=body.max_tokens, temperature=body.temperature,
        top_p=body.top_p, top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        range_epsilon=body.range_epsilon,
    )


def _check_rate_limit_and_capacity(request: Request) -> str:
    client_key = get_client_key(request, trust_forwarded_for=TRUST_FORWARDED_FOR)

    allowed_day, retry_day = per_day_limiter.check(client_key)
    if not allowed_day:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({RATE_LIMIT_PER_DAY}/day). "
                   f"Try again in {retry_day/3600:.1f}h.",
            headers={"Retry-After": str(int(retry_day) + 1)},
        )

    allowed_min, retry_min = per_minute_limiter.check(client_key)
    if not allowed_min:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). "
                   f"Try again in {retry_min:.1f}s.",
            headers={"Retry-After": str(int(retry_min) + 1)},
        )

    if not concurrency_guard.try_acquire():
        raise HTTPException(status_code=503, detail="Server is at capacity — please retry shortly.")

    return client_key


@app.post("/generate", response_model=GenerateResponse, tags=["inference"],
          dependencies=[Depends(verify_auth_token)])
async def generate(body: GenerateRequest, request: Request):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    client_key = _check_rate_limit_and_capacity(request)

    request_id = ModalVolumeRequestLogger.new_id()
    t0 = time.perf_counter()
    entry = RequestLogEntry(
        request_id=request_id,
        timestamp=ModalVolumeRequestLogger.now_iso(),
        endpoint="/generate",
        client_key=client_key,
        status="ok",
        prompt=body.prompt,
        system=body.system,
        params={
            "max_tokens": body.max_tokens, "temperature": body.temperature,
            "top_p": body.top_p, "top_k": body.top_k,
            "repetition_penalty": body.repetition_penalty,
            "range_epsilon": body.range_epsilon,
        },
        user_agent=request.headers.get("user-agent"),
    )

    try:
        result = await asyncio.to_thread(engine.generate, _make_req(body))
        entry.response         = result["response"]
        entry.tokens_generated = result["tokens_generated"]
        entry.elapsed_ms       = result["elapsed_ms"]
        return GenerateResponse(**result)

    except RuntimeError as exc:
        entry.status, entry.error = "error", str(exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        entry.status, entry.error = "error", str(exc)
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected generation error.")
    finally:
        entry.elapsed_ms = entry.elapsed_ms or round((time.perf_counter() - t0) * 1000, 2)
        req_logger.log(entry)
        concurrency_guard.release()


@app.post("/generate/stream", tags=["inference"],
          dependencies=[Depends(verify_auth_token)])
async def generate_stream(body: GenerateRequest, request: Request):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    client_key = _check_rate_limit_and_capacity(request)

    req = _make_req(body)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    request_id = ModalVolumeRequestLogger.new_id()
    t0 = time.perf_counter()
    entry = RequestLogEntry(
        request_id=request_id,
        timestamp=ModalVolumeRequestLogger.now_iso(),
        endpoint="/generate/stream",
        client_key=client_key,
        status="ok",
        prompt=body.prompt,
        system=body.system,
        params={
            "max_tokens": body.max_tokens, "temperature": body.temperature,
            "top_p": body.top_p, "top_k": body.top_k,
            "repetition_penalty": body.repetition_penalty,
            "range_epsilon": body.range_epsilon,
        },
        user_agent=request.headers.get("user-agent"),
    )

    def run_in_thread():
        try:
            engine.generate_stream(req, queue, loop)
        except Exception as exc:
            logger.error("Streaming thread error: %s", exc, exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    async def event_stream():
        loop.run_in_executor(None, run_in_thread)
        chunks = []
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    chunks.append(payload)
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    entry.tokens_generated = payload
                    yield f"data: {json.dumps({'done': True, 'tokens_generated': payload})}\n\n"
                    break
                elif kind == "error":
                    entry.status, entry.error = "error", payload
                    yield f"data: {json.dumps({'error': payload})}\n\n"
                    break
        finally:
            entry.response = "".join(chunks)
            entry.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            req_logger.log(entry)
            concurrency_guard.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    logger.info("Starting Uvicorn on %s:%d", SERVER_HOST, SERVER_PORT)
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT,
                reload=False, log_level="warning", workers=1)