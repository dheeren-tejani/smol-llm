import os
import sys
import modal

app = modal.App("smol-lm-backend")

# Build image with CUDA support and explicit GCC pointers
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "ninja-build")
    .env({"CC": "gcc", "CXX": "g++"})  # Fix: overrides the phantom clang environment variable
    .pip_install(
        "fastapi", "uvicorn", "pydantic", "python-dotenv",
        "cryptography", "numpy",
    )
    .run_commands(
        "CC=gcc CXX=g++ CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75' "
        "pip install llama-cpp-python --no-binary llama-cpp-python"
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
    gpu="T4",
    cpu=2,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("smol-lm-secrets")],
    volumes={"/logs": log_volume},
    min_containers=0,
    scaledown_window=600,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/app")
    os.environ["LOG_DIR"] = "/logs"
    os.environ["GGUF_MODEL_PATH"] = "/app/models/model_q8_0.gguf"
    os.environ["GGUF_N_GPU_LAYERS"] = "-1"
    os.environ["GGUF_N_THREADS"] = "2"
    
    from main import app as fastapi_instance
    import main
    main.req_logger.volume = log_volume
    return fastapi_instance