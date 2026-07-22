"""
modal_app.py — Modal deployment for the Smol-lm backend.
"""
import modal

app = modal.App("smol-lm-backend")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "fastapi", "uvicorn", "pydantic", "python-dotenv",
        "tiktoken", "cryptography", "numpy",
    )
    .add_local_dir(".", remote_path="/app",
                   ignore=["*.pt", "__pycache__", ".git", "logs", ".env", "llm/**", "playground/**"])
    .add_local_file("models/best_sft.pt", remote_path="/app/models/best_sft.pt")
)
    
secrets = [modal.Secret.from_name("smol-lm-secrets")]

# Persistent volume for request/response logs — survives container restarts
# and scale-to-zero. Cost: $0.09/GiB-month, 1 TiB/month free (effectively
# $0 for text logs at this scale).
log_volume = modal.Volume.from_name("smol-lm-logs", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    secrets=secrets,
    volumes={"/logs": log_volume},
    min_containers=0,
    scaledown_window=120,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    sys.path.insert(0, "/app")
    os.environ["LOG_DIR"] = "/logs"
    from main import app as fastapi_instance
    return fastapi_instance