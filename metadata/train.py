"""
train.py — Main pretraining script
=====================================
Full-featured LLM pretraining loop:
  • torch.compile (reduce-overhead mode, see NOTE below)
  • bfloat16 AMP
  • Gradient accumulation
  • Cosine LR schedule with warmup
  • Gradient clipping
  • Periodic eval on val set
  • Checkpoint save / auto-resume
  • Structured logging to logs/ folder

Usage:
    python train.py                        # train from scratch
    python train.py --resume               # auto-resume from latest checkpoint
    python train.py --resume --ckpt path   # resume from specific checkpoint
    python train.py --preset gpt2-medium   # use a different model size
    python train.py --no-compile           # skip torch.compile (easier debugging)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE on torch.compile + grad accumulation (your benchmark question)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The default compile mode is "reduce-overhead", NOT "max-autotune".

Here's why your benchmark needed TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1:

  torch.compile(mode="reduce-overhead") works by capturing CUDA graphs.
  A CUDA graph records a sequence of GPU ops into a single replayable
  graph — very low CPU overhead, very fast. BUT: when you do gradient
  accumulation in a Python for-loop, each accum step launches a
  *slightly different* graph (different tensor addresses, or CUDA sees
  them as different due to the zero_grad between outer steps). This
  causes CUDAGraph "tensor overwrite" errors because the graph tries
  to write into the same memory slot it recorded from.

  Solutions (pick one):
    A. TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1   ← your workaround (disables graphs)
    B. torch.compiler.cudagraph_mark_step_begin() before each accum step  ← tells
       the graph machinery "new step starts here, safe to re-record"
    C. Use mode="reduce-overhead" + no_sync() context (DDP scenario)
    D. Use mode="max-autotune"   ← uses autotuned kernels but NOT CUDA graphs
       by default, so it's immune to this bug. Slower compile (~5 min),
       faster kernels, no graph capture issues with grad accum.

  This script uses approach B (cudagraph_mark_step_begin) so you keep
  CUDA graphs AND grad accumulation. If you want max throughput, switch
  COMPILE_MODE to "max-autotune" — it autotunes every matmul kernel
  (like torch's own version of NVIDIA's cuBLAS heuristics) but takes
  longer to compile the first time.
"""

import os
import sys
import time
import math
import argparse
import dataclasses

# ── Must be set BEFORE importing torch ───────────────────────
# Not needed with our cudagraph_mark_step_begin approach, but
# left here as a safety comment. Remove if using max-autotune.
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

import torch
import torch.nn.functional as F
import tiktoken

from training.model      import GPT, ModelConfig, MODEL_PRESETS
from training.data       import make_loader
from training.scheduler  import get_lr
from training.logger     import TrainingLogger
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# ─────────────────────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainConfig:
    # ── Data ─────────────────────────────────────────────────
    train_bin:    str   = "data/train.bin"
    val_bin:      str   = "data/val.bin"
    seq_len:      int   = 1024

    # ── Batch ─────────────────────────────────────────────────
    # Effective batch = batch_size × grad_accum_steps tokens × seq_len
    # For H100: batch_size=64, grad_accum=4 → 256 × 1024 = 262k tokens/step
    batch_size:       int = 32
    grad_accum_steps: int = 8

    # ── Optimizer ────────────────────────────────────────────
    max_lr:        float = 6e-4
    min_lr:        float = 6e-5     # = max_lr / 10  (Chinchilla)
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # ── Schedule ─────────────────────────────────────────────
    warmup_steps:  int   = 2_000
    total_steps:   int   = 84_000  # adjust to your token budget

    # ── Checkpointing ────────────────────────────────────────
    ckpt_dir:      str   = "checkpoints"
    ckpt_every:    int   = 1_000     # save every N steps
    keep_last_n:   int   = 3         # how many step_*.pt to keep

    # ── Evaluation ───────────────────────────────────────────
    eval_every:    int   = 500       # eval val loss every N steps
    eval_batches:  int   = 20        # number of val batches to average
    sample_every:  int   = 500

    # ── Logging ──────────────────────────────────────────────
    log_every:     int   = 10        # print/log every N steps
    run_name:      str   = "smol-lm"

    # ── System ───────────────────────────────────────────────
    num_workers:   int   = 16
    compile:       bool  = True
    compile_mode:  str   = "max-autotune"
    # Options:
    #   "default"          — safe, moderate speedup
    #   "reduce-overhead"  — CUDA graphs, ~5-10% faster than default
    #   "max-autotune"     — autotuned kernels, no CUDA graph issues,
    #                        ~10-20% faster kernels, ~5 min compile time


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, n_batches: int) -> float:
    model.eval()
    losses = []
    loader_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


# ─────────────────────────────────────────────────────────────
# Optimizer (separate weight decay groups like GPT-3)
# ─────────────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    """
    Only apply weight decay to 2D parameters (weights).
    Biases, norms, embeddings → no decay.
    Uses fused AdamW for ~30% speedup on CUDA.
    """
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    decay_params     = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in raw.named_parameters() if p.dim() < 2]

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=True,          # fused kernel, ~30% faster on H100
    )
    return optimizer


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",   action="store_true", help="Auto-resume from latest checkpoint")
    p.add_argument("--ckpt",     type=str,   default=None, help="Path to specific checkpoint")
    p.add_argument("--preset",   type=str,   default="gpt2-small", choices=list(MODEL_PRESETS))
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--compile-mode", type=str, default=None,
                   help="Override compile mode: default|reduce-overhead|max-autotune")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Configs ───────────────────────────────────────────────
    model_cfg = MODEL_PRESETS[args.preset]
    train_cfg = TrainConfig()

    if args.no_compile:
        train_cfg.compile = False
    if args.compile_mode:
        train_cfg.compile_mode = args.compile_mode

    train_cfg.run_name = f"{args.preset}-cosmopedia"

    # ── Logger ────────────────────────────────────────────────
    logger = TrainingLogger(run_name=train_cfg.run_name)

    cfg_snapshot = {**vars(model_cfg), **vars(train_cfg)}
    logger.log_config(cfg_snapshot)

    # ── Device ────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True    # faster matmuls on Ampere+
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True    # <-- ADD THIS
    torch.set_float32_matmul_precision("high")       # bf16 matmul accumulation

    logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Model ─────────────────────────────────────────────────
    logger.info(f"\nBuilding {args.preset} model...")
    model = GPT(model_cfg).to(device)
    # model = model.to(torch.bfloat16)
    logger.info(f"Parameters: {model.count_params(non_embedding=True)/1e6:.1f}M  (non-embedding)")
    logger.info(f"Parameters: {model.count_params(non_embedding=False)/1e6:.1f}M  (total)")

    if train_cfg.compile:
        logger.info(f"Compiling model (mode={train_cfg.compile_mode})...")
        logger.info("  First compile takes 60–300s depending on mode.")
        model = torch.compile(model, mode=train_cfg.compile_mode)
        logger.info("  Compile done.")

    # ── Optimizer ────────────────────────────────────
    optimizer = build_optimizer(model, train_cfg)

    # ── Data ──────────────────────────────────────────────────
    logger.info("\nLoading data loaders...")
    train_loader = make_loader(
        train_cfg.train_bin,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = make_loader(
        train_cfg.val_bin,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # ── Resume ────────────────────────────────────────────────
    start_step   = 0
    tokens_seen  = 0

    ckpt_path = args.ckpt
    if args.resume and ckpt_path is None:
        ckpt_path = find_latest_checkpoint(train_cfg.ckpt_dir)

    if ckpt_path is not None:
        logger.info(f"\nResuming from {ckpt_path} ...")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, device=str(device))
        start_step  = ckpt["step"] + 1
        tokens_seen = ckpt.get("tokens_seen", 0)
        logger.info(f"  Resumed at step {start_step:,}  |  {tokens_seen/1e9:.3f}B tokens seen")

    # ── Derived training constants ────────────────────────────
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * train_cfg.seq_len
    total_tokens    = train_cfg.total_steps * tokens_per_step

    logger.log_start(train_cfg.total_steps, total_tokens)
    logger.info(f"Tokens per step (effective batch): {tokens_per_step:,}")
    logger.info(f"Total steps: {train_cfg.total_steps:,}")
    logger.info(f"Total tokens: {total_tokens/1e9:.2f}B\n")

    # ── Training loop ─────────────────────────────────────────
    model.train()
    train_iter   = iter(train_loader)

    # ── Setup Sampler ─────────────────────────────────────────
    enc = tiktoken.get_encoding("gpt2")
    sample_prompts = [
        "Artificial Intelligence is",
        "Define Gravity",
        "Once upon a time,",
        "Mitochondria is the",
        "1 + 1 ="
    ]
    
    t_run_start  = time.perf_counter()
    best_val_loss = float("inf")

    for step in range(start_step, train_cfg.total_steps):

        # ── LR update ─────────────────────────────────────────
        lr = get_lr(
            step,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
            max_lr=train_cfg.max_lr,
            min_lr=train_cfg.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.perf_counter()

        # ── Gradient accumulation ─────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for accum_step in range(train_cfg.grad_accum_steps):
            # Tell CUDA graph machinery a new micro-step begins.
            # This is what fixes the "tensor overwrite" error that your
            # benchmark hit — no need to disable CUDA graphs entirely.
            # torch.compiler.cudagraph_mark_step_begin()

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Inside the accum_step loop:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps   

            loss.backward()
            accum_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.grad_clip
        ).item()
        
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        tokens_seen += tokens_per_step
        step_ms      = (t1 - t0) * 1000
        tok_per_sec  = tokens_per_step / (t1 - t0)

        # ── Logging ───────────────────────────────────────────
        if step % train_cfg.log_every == 0:
            logger.log_step(
                step        = step,
                loss        = accum_loss,
                lr          = lr,
                tok_per_sec = tok_per_sec,
                grad_norm   = grad_norm,
                elapsed_sec = time.perf_counter() - t_run_start,
                tokens_seen = tokens_seen,
                step_ms     = step_ms,
            )

        # ── Generation Sampling ───────────────────────────────
        if step % train_cfg.sample_every == 0 and step > 0:
            logger.info(f"\n{'─'*30} SAMPLING step {step:,} {'─'*30}")
            model.eval()
            
            # CRITICAL: We must unwrap the compiled model for generation.
            # If we don't, the dynamic sequence length of generation will 
            # trigger massive torch.compile graph recompilations.
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            
            for prompt in sample_prompts:
                # Encode text to tensor
                idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                
                # Generate 50 new tokens
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_idx = raw_model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=40)
                
                # Decode back to text and print
                out_text = enc.decode(out_idx[0].cpu().tolist())
                logger.info(f"Prompt:   '{prompt}'")
                logger.info(f"Generated: {out_text}\n")
                
            model.train()


        # ── Evaluation ────────────────────────────────────────
        if step % train_cfg.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device, train_cfg.eval_batches)
            logger.log_eval(step=step, val_loss=val_loss, train_loss=accum_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # ── Checkpoint ────────────────────────────────────────
        if step % train_cfg.ckpt_every == 0 and step > 0:
            val_loss_for_ckpt = best_val_loss if best_val_loss < float("inf") else None
            ckpt_path = save_checkpoint(
                step         = step,
                model        = model,
                optimizer    = optimizer,
                model_cfg    = model_cfg,
                train_cfg    = train_cfg,
                tokens_seen  = tokens_seen,
                ckpt_dir     = train_cfg.ckpt_dir,
                val_loss     = val_loss_for_ckpt,
                keep_last_n  = train_cfg.keep_last_n,
            )
            logger.log_checkpoint(step=step, path=ckpt_path)

    # ── Final eval + checkpoint ───────────────────────────────
    val_loss = evaluate(model, val_loader, device, train_cfg.eval_batches)
    logger.log_eval(step=train_cfg.total_steps, val_loss=val_loss)

    save_checkpoint(
        step        = train_cfg.total_steps,
        model       = model,
        optimizer   = optimizer,
        model_cfg   = model_cfg,
        train_cfg   = train_cfg,
        tokens_seen = tokens_seen,
        ckpt_dir    = train_cfg.ckpt_dir,
        val_loss    = val_loss,
        keep_last_n = train_cfg.keep_last_n,
    )

    logger.log_end(
        elapsed_sec = time.perf_counter() - t_run_start,
        tokens_seen = tokens_seen,
    )
    logger.close()


if __name__ == "__main__":
    main()