# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install dependencies in a separate layer so the final image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install CPU-only PyTorch first (saves ~1.8 GB vs the default CUDA wheel)
#RUN pip install --no-cache-dir \
#    torch==2.3.1+cpu \
#    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY config.py inference.py model.py main.py ./

# Bake in the model checkpoint and tokenizer cache
# Expects: model/llm_model.pt  (matches CHECKPOINT_PATH default)
COPY model/ ./model/

# HuggingFace tokenizer will be downloaded at build time and cached here.
# Set HF_HOME so the cache lands inside /app and gets baked into the image.
ENV HF_HOME=/app/.hf_cache
ARG TOKENIZER_NAME=gpt2
RUN python - <<'EOF'
from transformers import AutoTokenizer
import os
tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_NAME"])
print(f"Tokenizer '{os.environ['TOKENIZER_NAME']}' cached successfully.")
EOF

# Cloud Run injects PORT automatically; default to 8000 for local docker run
ENV PORT=7860
ENV HOST=0.0.0.0

# Disable Python output buffering so logs appear immediately in Cloud Run
ENV PYTHONUNBUFFERED=1

# Tell HuggingFace to use the baked-in cache and never reach out to the hub
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

CMD ["python", "main.py"]