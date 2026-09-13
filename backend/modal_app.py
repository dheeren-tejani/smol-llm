import os
import sys
import modal

app = modal.App("smol-lm-backend")

# 1. Clean image dependencies (removed tiktoken, using Debian slim)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi", "uvicorn", "pydantic", "python-dotenv",
        "cryptography", "numpy", "llama-cpp-python",
    )
    .add_local_file("models/model_q8_0.gguf", remote_path="/app/models/model_q8_0.gguf")
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["*.pt", "__pycache__", ".git", "logs", ".env", "llm/**", "playground/**", "models/**"],
    )
)

log_volume = modal.Volume.from_name("smol-lm-logs", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,                              # 4 CPU cores for llama.cpp token generation
    enable_memory_snapshot=True,        # ⚡ Freezes RAM after warmup for ~300ms boots
    secrets=[modal.Secret.from_name("smol-lm-secrets")],
    volumes={"/logs": log_volume},
    min_containers=0,                   # Set to 1 if you need strictly 0ms warm availability
    scaledown_window=1200,               # Keep alive for 5 minutes after last request
    timeout=300,                        # 5-minute maximum request ceiling
)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/app")
    os.environ["LOG_DIR"] = "/logs"
    os.environ["GGUF_MODEL_PATH"] = "/app/models/model_q8_0.gguf"
    os.environ["GGUF_N_GPU_LAYERS"] = "0"
    os.environ["GGUF_N_THREADS"] = "4"
    
    from main import app as fastapi_instance
    import main
    main.req_logger.volume = log_volume
    return fastapi_instance