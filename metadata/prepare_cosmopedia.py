
"""
prepare_cosmopedia.py
=====================
Phase 1 — Download: pulls all Cosmopedia Parquet shards from HuggingFace
          at maximum speed using hf_transfer (Rust-based multi-part HTTP).

Phase 2 — Tokenize: processes all shards in parallel across all CPU cores,
          tokenizes with GPT-2, and writes flat uint16 binary files:
              data/train.bin   (~99% of data)
              data/val.bin     (~1%  of data)

Requirements:
    pip install huggingface_hub[hf_transfer] datasets tiktoken numpy tqdm

Speed vs. original streaming script:
    - Download  : hf_transfer saturates bandwidth (multi-part, Rust HTTP)
    - Tokenize  : multiprocessing.Pool across all CPU cores vs single-threaded
    - No overlap between phases — cleaner, more predictable, resumable
"""

import os
import glob
import math
import time
import struct
import numpy as np
import tiktoken
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HF_REPO          = "HuggingFaceTB/cosmopedia"
HF_TOKEN         = os.environ.get("HF_TOKEN", None)   # or hardcode: "hf_..."

RAW_DIR          = "raw_cosmopedia"    # downloaded Parquet shards go here
OUT_DIR          = "data"             # train.bin / val.bin go here

VAL_FRACTION     = 0.01               # 1% validation
SHARD_FLUSH_TOKS = 100_000_000        # flush to disk every 100M tokens
TEXT_COLUMN      = "text"
DTYPE            = np.uint16          # GPT-2 vocab=50257 fits in uint16

NUM_WORKERS      = 24   # leave 1 core for I/O

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DOWNLOAD WITH hf_transfer
# ─────────────────────────────────────────────────────────────────────────────

def download():
    # Enable hf_transfer — must be set before importing huggingface_hub internals
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        import hf_transfer  # noqa: F401 — just verify it's installed
    except ImportError:
        raise SystemExit(
            "\n[ERROR] hf_transfer not found.\n"
            "Install with:  pip install huggingface_hub[hf_transfer]\n"
        )

    from huggingface_hub import snapshot_download, login

    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    else:
        print("[WARN] No HF_TOKEN set. Downloads will be rate-limited.")
        print("       Set HF_TOKEN env var or hardcode it in CONFIG above.\n")

    print("=" * 60)
    print("  PHASE 1 — Downloading Cosmopedia shards")
    print("=" * 60)
    print(f"  Repo   : {HF_REPO}")
    print(f"  Dest   : {RAW_DIR}/")
    print(f"  Engine : hf_transfer (multi-part Rust HTTP)")
    print()

    t0 = time.time()

    local_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=RAW_DIR,
        allow_patterns=[f"data/**/*.parquet"],  # only the subset we want
        ignore_patterns=["*.json", "*.md", "*.txt"],
        token=HF_TOKEN,
    )

    elapsed = time.time() - t0
    shards = glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True)
    total_bytes = sum(os.path.getsize(p) for p in shards)

    print(f"\n  Downloaded {len(shards)} shards  ({total_bytes/1e9:.1f} GB)")
    print(f"  Time     : {elapsed/60:.1f} min")
    print(f"  Speed    : {total_bytes/elapsed/1e6:.0f} MB/s")
    print()

    return sorted(shards)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TOKENIZE IN PARALLEL
# ─────────────────────────────────────────────────────────────────────────────

# Worker initializer — each process gets its own tokenizer instance
_enc = None

def _worker_init():
    global _enc
    _enc = tiktoken.get_encoding("gpt2")


def _tokenize_shard(parquet_path: str) -> np.ndarray:
    """
    Read one Parquet shard, tokenize all documents, return uint16 array.
    Each document is prepended with the EOT token (50256).
    """
    import pyarrow.parquet as pq

    eot = _enc.encode_single_token("<|endoftext|>")  # 50256
    table = pq.read_table(parquet_path, columns=[TEXT_COLUMN])
    texts = table[TEXT_COLUMN].to_pylist()

    tokens = []
    for text in texts:
        ids = _enc.encode_ordinary(text)
        tokens.append(eot)
        tokens.extend(ids)

    return np.array(tokens, dtype=np.uint16)


def tokenize(shards: list[str]):
    os.makedirs(OUT_DIR, exist_ok=True)

    train_path = os.path.join(OUT_DIR, "train.bin")
    val_path   = os.path.join(OUT_DIR, "val.bin")

    # Remove stale files
    for p in [train_path, val_path]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed old {p}")

    print("=" * 60)
    print("  PHASE 2 — Tokenizing")
    print("=" * 60)
    print(f"  Shards  : {len(shards)}")
    print(f"  Workers : {NUM_WORKERS} CPU cores")
    print(f"  Val     : first {VAL_FRACTION*100:.0f}% of tokens → val.bin")
    print()

    # We'll collect val tokens until we have ~1% of the expected total,
    # estimated from the first shard. Everything else → train.
    # Since we don't know total tokens upfront, we fill val to a fixed
    # estimate based on ~25B total tokens.
    VAL_TOKEN_TARGET = int(25_000_000_000 * VAL_FRACTION)  # ~250M tokens for val

    train_buf:     list[np.ndarray] = []
    val_buf:       list[np.ndarray] = []
    train_buf_len: int = 0
    val_buf_len:   int = 0
    total_tokens:  int = 0
    val_tokens:    int = 0
    train_tokens:  int = 0

    t0 = time.time()

    ctx = mp.get_context("spawn")   # safer than fork with tiktoken

    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
    ) as pool:
        pbar = tqdm(
            total=len(shards),
            unit="shard",
            desc="Tokenizing shards",
            dynamic_ncols=True,
        )

        for arr in pool.imap(_tokenize_shard, shards, chunksize=1):
            n = len(arr)

            if val_tokens < VAL_TOKEN_TARGET:
                # Split this shard between val and train if we're on the boundary
                val_need = VAL_TOKEN_TARGET - val_tokens
                if val_need >= n:
                    val_buf.append(arr)
                    val_buf_len += n
                    val_tokens  += n
                else:
                    val_buf.append(arr[:val_need])
                    val_buf_len += val_need
                    val_tokens  += val_need
                    train_buf.append(arr[val_need:])
                    train_buf_len += n - val_need
                    train_tokens  += n - val_need
            else:
                train_buf.append(arr)
                train_buf_len += n
                train_tokens  += n

            total_tokens += n

            # Flush train buffer periodically to keep RAM in check
            if train_buf_len >= SHARD_FLUSH_TOKS:
                _flush(train_buf, train_path)
                train_buf     = []
                train_buf_len = 0

            # Val is small — flush once when full
            if val_buf_len >= SHARD_FLUSH_TOKS:
                _flush(val_buf, val_path)
                val_buf     = []
                val_buf_len = 0

            elapsed = time.time() - t0
            toks_per_sec = total_tokens / elapsed
            pbar.set_postfix({
                "total": f"{total_tokens/1e9:.2f}B",
                "speed": f"{toks_per_sec/1e6:.1f}M tok/s",
            })
            pbar.update(1)

        pbar.close()

    # Flush remaining buffers
    if train_buf:
        _flush(train_buf, train_path)
    if val_buf:
        _flush(val_buf, val_path)

    elapsed = time.time() - t0

    # ── Report ──────────────────────────────────────────────────────────────
    def fcnt(p): return os.path.getsize(p) // 2

    tr = fcnt(train_path)
    vl = fcnt(val_path)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  train.bin : {tr:>15,} tokens  ({os.path.getsize(train_path)/1e9:.2f} GB)")
    print(f"  val.bin   : {vl:>15,} tokens  ({os.path.getsize(val_path)/1e9:.2f} GB)")
    print(f"  Total     : {tr+vl:>15,} tokens  ({(os.path.getsize(train_path)+os.path.getsize(val_path))/1e9:.2f} GB)")
    print(f"  Time      : {elapsed/60:.1f} min")
    print(f"  Speed     : {(tr+vl)/elapsed/1e6:.1f}M tok/s  (across {NUM_WORKERS} cores)")
    print("=" * 60)

    # ── Sanity check ─────────────────────────────────────────────────────────
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.encode_single_token("<|endoftext|>")
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    first10 = data[:10].tolist()
    print(f"\nSanity check — first 10 tokens: {first10}")
    print(f"Decoded: {enc.decode([t for t in first10 if t != eot])}")
    print("\nLoad in training loop with:")
    print("  data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')")
    print("  x    = torch.from_numpy(data[i:i+block_size].astype(np.int32))")


def _flush(buf: list[np.ndarray], path: str):
    arr = np.concatenate(buf)
    with open(path, "ab") as f:
        arr.tofile(f)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    shards = download()
    tokenize(shards)
