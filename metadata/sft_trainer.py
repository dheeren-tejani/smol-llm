
"""
sft_train.py — Supervised Fine-Tuning for Senku

Picks up exactly where pretraining left off:
  • Same model.py  (GPT + ModelConfig + MODEL_PRESETS)
  • Same scheduler.py  (cosine LR + warmup)
  • Same logger.py  (CSV + text log)

New SFT-specific pieces:
  • SFTDataset  — reads the prepared JSON, tokenises on-the-fly,
                  builds loss masks so only assistant tokens are trained
  • ChatFormatter — converts conversations to a flat prompt string
  • Full fine-tune (all parameters updated)
  • Gradient accumulation, gradient clipping, bf16 autocast
  • Eval loop on a held-out val split
  • Checkpoint saves full model state + optimizer + step

Usage:
    python sft_train.py \
        --data   data/senku_sft_50k.json \
        --ckpt   checkpoints/pretrain_final.pt \
        --preset gpt2-small \
        --out-dir checkpoints/sft

    # Resume from a previous SFT checkpoint:
    python sft_train.py ... --resume checkpoints/sft/step_500.pt
"""

import os
import sys
import math
import time
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ── Local modules (same directory) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from training.model     import GPT, ModelConfig, MODEL_PRESETS
from training.scheduler import get_lr
from training.logger    import TrainingLogger


# ════════════════════════════════════════════════════════════
# 1.  Tokenizer loader  (tiktoken → HF → crash with message)
# ════════════════════════════════════════════════════════════

def load_tokenizer(name: str = "gpt2"):
    """
    Returns (encode_fn, decode_fn, vocab_size, eos_id, pad_id).
    encode_fn(text) -> List[int]
    """
    tok = AutoTokenizer.from_pretrained(name)
    
    # 1. ADD YOUR SFT SPECIAL TOKENS
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    tok.add_special_tokens({'additional_special_tokens': special_tokens})
    
    eos = tok.eos_token_id or 0
    pad = tok.pad_token_id or eos
    
    # 2. HARDCODE TO MATCH YOUR PRETRAINED CHECKPOINT
    padded_vocab_size = 50304 
    
    print(f"[tokenizer] HuggingFace '{name}'  active_vocab={len(tok)}  padded_vocab={padded_vocab_size}  eos={eos}")
    return (
        lambda t: tok.encode(t, add_special_tokens=False),
        lambda ids: tok.decode(ids),
        padded_vocab_size,  # <--- This feeds 50304 directly into your ModelConfig
        eos,
        pad,
    )


# ════════════════════════════════════════════════════════════
# 2.  Chat formatter
# ════════════════════════════════════════════════════════════

# Simple ChatML-style format:
#   <|system|>\n{content}\n
#   <|user|>\n{content}\n
#   <|assistant|>\n{content}\n<|end|>\n
#
# Loss is computed ONLY on tokens inside assistant turns (after <|assistant|>\n).

ROLE_TOKENS = {
    "system"   : "<|system|>",
    "user"     : "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


class ChatFormatter:
    """
    Converts a conversation (list of role/content dicts) into
    a flat token sequence + a boolean mask where True = train on this token.
    """

    def __init__(self, encode_fn, eos_id: int, max_seq_len: int):
        self.encode      = encode_fn
        self.eos_id      = eos_id
        self.max_seq_len = max_seq_len

        # Pre-encode role headers and end token
        self._role_ids: Dict[str, List[int]] = {
            role: encode_fn(tok) for role, tok in ROLE_TOKENS.items()
        }
        self._end_ids = encode_fn(END_TOKEN)

    def format(
        self,
        conversations: List[Dict],
    ) -> Tuple[List[int], List[bool]]:
        """
        Returns:
            token_ids : flat list of ints, length ≤ max_seq_len
            loss_mask : parallel bool list; True where loss is computed
        """
        token_ids: List[int] = []
        loss_mask: List[bool] = []

        for turn in conversations:
            role    = turn["role"]
            content = turn["content"]

            header_ids = self._role_ids.get(role, self._encode_role(role))
            body_ids   = self.encode(content)

            is_assistant = (role == "assistant")

            # Header tokens — never trained on
            token_ids.extend(header_ids)
            loss_mask.extend([False] * len(header_ids))

            # Body tokens — trained on only for assistant turns
            token_ids.extend(body_ids)
            loss_mask.extend([is_assistant] * len(body_ids))

            # End token after assistant turns
            if is_assistant:
                token_ids.extend(self._end_ids)
                loss_mask.extend([True] * len(self._end_ids))

            if len(token_ids) >= self.max_seq_len:
                break

        # Truncate
        token_ids = token_ids[: self.max_seq_len]
        loss_mask = loss_mask[: self.max_seq_len]

        return token_ids, loss_mask

    def _encode_role(self, role: str) -> List[int]:
        return self.encode(f"<|{role}|>")


# ════════════════════════════════════════════════════════════
# 3.  SFT Dataset
# ════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    Reads the prepared JSON (list of {"conversations": [...]}).
    Tokenises each conversation on-the-fly and returns:
        input_ids  : (seq_len,)  int64
        labels     : (seq_len,)  int64  (-100 where masked)
    """

    IGNORE_IDX = -100  # standard PyTorch cross-entropy ignore index

    def __init__(
        self,
        json_path: str,
        formatter: ChatFormatter,
        max_seq_len: int,
    ):
        path = Path(json_path)
        assert path.exists(), f"Data file not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.formatter   = formatter
        self.max_seq_len = max_seq_len
        print(f"[data] Loaded {len(self.records):,} SFT records from {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        convs = self.records[idx]["conversations"]
        token_ids, loss_mask = self.formatter.format(convs)

        # Need at least 2 tokens for a (input, label) pair
        if len(token_ids) < 2:
            # Return a dummy; collate_fn will handle it
            token_ids = [0, 0]
            loss_mask = [False, False]

        # Input  = all tokens except last
        # Labels = all tokens except first, masked where loss_mask=False
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        raw_labels = token_ids[1:]
        raw_mask   = loss_mask[1:]
        labels = torch.tensor(
            [t if m else self.IGNORE_IDX for t, m in zip(raw_labels, raw_mask)],
            dtype=torch.long,
        )
        return input_ids, labels


def sft_collate(batch):
    """Pad a batch of variable-length sequences to the longest in the batch."""
    inputs, labels = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), SFTDataset.IGNORE_IDX, dtype=torch.long)

    for i, (inp, lbl) in enumerate(zip(inputs, labels)):
        L = inp.size(0)
        padded_inputs[i, :L] = inp
        padded_labels[i, :L] = lbl

    return padded_inputs, padded_labels


def make_sft_loaders(
    json_path: str,
    formatter: ChatFormatter,
    max_seq_len: int,
    batch_size: int,
    val_ratio: float = 0.02,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split into train/val and return both DataLoaders."""
    full_ds = SFTDataset(json_path, formatter, max_seq_len)

    n_val   = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=g)

    print(f"[data] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=sft_collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=sft_collate,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ════════════════════════════════════════════════════════════
# 4.  Checkpoint helpers
# ════════════════════════════════════════════════════════════

def load_pretrain_checkpoint(path: str, model: GPT, device: torch.device):
    """Load weights from a pretraining checkpoint (model state_dict only)."""
    ckpt = torch.load(path, map_location=device)
    # Support both raw state_dict and wrapped {"model": state_dict, ...}
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[ckpt] Loaded pretrain weights from {path}")


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    cfg: dict,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step"     : step,
            "val_loss" : val_loss,
            "config"   : cfg,
        },
        path,
    )


def resume_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step     = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"[ckpt] Resumed from {path}  (step={step}, val_loss={val_loss:.4f})")
    return step, val_loss


# ════════════════════════════════════════════════════════════
# 5.  Evaluation
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, model.cfg.vocab_size),
                y.view(-1),
                ignore_index=SFTDataset.IGNORE_IDX,
                reduction="sum",
            )
        n_toks = (y != SFTDataset.IGNORE_IDX).sum().item()
        total_loss += loss.item()
        total_toks += n_toks

    model.train()
    return total_loss / max(total_toks, 1)


# ════════════════════════════════════════════════════════════
# 6.  Training loop
# ════════════════════════════════════════════════════════════

def train(args):
    # ── Device ──────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] Device: {device}")

    # ── Model ───────────────────────────────────────────────
    if args.preset in MODEL_PRESETS:
        cfg = MODEL_PRESETS[args.preset]
    else:
        raise ValueError(f"Unknown preset '{args.preset}'. Options: {list(MODEL_PRESETS)}")

    # Allow overriding seq_len
    cfg.max_seq_len = args.seq_len
    cfg.dropout     = args.dropout

    model = GPT(cfg).to(device)
    n_params = model.count_params()
    print(f"[model] {args.preset}  |  {n_params/1e6:.1f}M non-embedding params")

    # ── Load pretrained weights ──────────────────────────────
    if args.ckpt:
        load_pretrain_checkpoint(args.ckpt, model, device)

    # ── Tokenizer + formatter ────────────────────────────────
    encode_fn, _, vocab_size, eos_id, _ = load_tokenizer(args.tokenizer)
    formatter = ChatFormatter(encode_fn, eos_id, args.seq_len)

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader = make_sft_loaders(
        json_path   = args.data,
        formatter   = formatter,
        max_seq_len = args.seq_len,
        batch_size  = args.batch_size,
        val_ratio   = args.val_ratio,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps    = args.warmup_steps if args.warmup_steps >= 0 else total_steps // 20
    min_lr          = args.lr / 10.0

    print(f"[train] steps/epoch={steps_per_epoch:,}  total_steps={total_steps:,}  warmup={warmup_steps}")

    # ── Optimizer ────────────────────────────────────────────
    # Separate weight-decay params (matrices) from no-decay (norms, biases, embeds)
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() <  2]
    param_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )

    # ── Resume ───────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = resume_checkpoint(args.resume, model, optimizer, device)

    # ── Logger ───────────────────────────────────────────────
    run_cfg = vars(args)
    run_cfg.update({"total_steps": total_steps, "n_params_M": round(n_params / 1e6, 1)})
    logger = TrainingLogger(run_name=f"senku_sft_{args.preset}")
    logger.log_config(run_cfg)
    logger.log_start(total_steps, total_steps * args.batch_size * args.seq_len)

    # ── Compile (optional) ───────────────────────────────────
    if args.compile:
        print("[train] Compiling model with torch.compile ...")
        model = torch.compile(model)

    # ── AMP scaler (not needed for bf16, but kept for fp16 compat) ──
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "fp16"))

    # ════════════════════════════════════════════════════════
    # Training loop
    # ════════════════════════════════════════════════════════
    model.train()
    optimizer.zero_grad()

    global_step  = start_step
    tokens_seen  = start_step * args.batch_size * args.seq_len * args.grad_accum
    t_start      = time.time()
    t_step_start = time.time()
    accum_loss   = 0.0

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for epoch in range(args.epochs):
        for micro_step, (x, y) in enumerate(train_loader):

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward + loss
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    y.view(-1),
                    ignore_index=SFTDataset.IGNORE_IDX,
                )
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # Gradient accumulation
            is_last_batch = (micro_step + 1) == len(train_loader)

            if (micro_step + 1) % args.grad_accum != 0 and not is_last_batch:
                continue

            # ── Optimizer step ───────────────────────────────
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Update LR
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # ── Bookkeeping ──────────────────────────────────
            global_step += 1
            batch_tokens = args.batch_size * x.size(1) * args.grad_accum
            tokens_seen += batch_tokens

            step_ms     = (time.time() - t_step_start) * 1000
            elapsed     = time.time() - t_start
            tok_per_sec = batch_tokens / (step_ms / 1000 + 1e-9)
            t_step_start = time.time()

            # ── Log ──────────────────────────────────────────
            if global_step % args.log_every == 0:
                logger.log_step(
                    step        = global_step,
                    loss        = accum_loss / args.log_every,
                    lr          = lr,
                    tok_per_sec = tok_per_sec,
                    grad_norm   = grad_norm.item(),
                    elapsed_sec = elapsed,
                    tokens_seen = tokens_seen,
                    step_ms     = step_ms,
                )
                accum_loss = 0.0

            # ── Eval ─────────────────────────────────────────
            if global_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
                logger.log_eval(global_step, val_loss)

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.out_dir, "best.pt")
                    save_checkpoint(best_path, model, optimizer, global_step, val_loss, run_cfg)
                    logger.log_checkpoint(global_step, best_path + "  [BEST]")

            # ── Periodic checkpoint ───────────────────────────
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"step_{global_step:07d}.pt")
                save_checkpoint(ckpt_path, model, optimizer, global_step, best_val_loss, run_cfg)
                logger.log_checkpoint(global_step, ckpt_path)

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    # ── Final checkpoint ─────────────────────────────────────
    final_path = os.path.join(args.out_dir, "final.pt")
    val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
    save_checkpoint(final_path, model, optimizer, global_step, val_loss, run_cfg)
    logger.log_eval(global_step, val_loss)
    logger.log_checkpoint(global_step, final_path + "  [FINAL]")
    logger.log_end(time.time() - t_start, tokens_seen)
    logger.close()


# ════════════════════════════════════════════════════════════
# 7.  CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SFT trainer for Senku")

    # Data
    p.add_argument("--data",       required=True,         help="Path to prepared SFT JSON")
    p.add_argument("--tokenizer",  default="gpt2",        help="Tokenizer (tiktoken or HF model id)")
    p.add_argument("--val-ratio",  type=float, default=0.02, help="Fraction held out for validation")

    # Model
    p.add_argument("--preset",     default="gpt2-small",  help=f"Model preset: {list(MODEL_PRESETS)}")
    p.add_argument("--ckpt",       default=None,          help="Pretrain checkpoint to load (.pt)")
    p.add_argument("--resume",     default=None,          help="Resume from SFT checkpoint (.pt)")
    p.add_argument("--seq-len",    type=int, default=1024, help="Max sequence length")
    p.add_argument("--dropout",    type=float, default=0.1, help="Dropout rate (0.0 = off)")

    # Training
    p.add_argument("--epochs",      type=int,   default=3,      help="Number of epochs")
    p.add_argument("--batch-size",  type=int,   default=32,      help="Batch size per GPU step")
    p.add_argument("--grad-accum",  type=int,   default=8,      help="Gradient accumulation steps")
    p.add_argument("--lr",          type=float, default=2e-5,   help="Peak learning rate")
    p.add_argument("--weight-decay",type=float, default=0.1,    help="AdamW weight decay")
    p.add_argument("--grad-clip",   type=float, default=1.0,    help="Gradient clip norm")
    p.add_argument("--warmup-steps",type=int,   default=-1,     help="LR warmup steps (-1=auto 5%)")
    p.add_argument("--dtype",       default="bf16",             help="bf16 | fp16 | fp32")
    p.add_argument("--compile",     action="store_true",        help="torch.compile the model")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default=None,               help="cuda | cpu (auto-detect if omitted)")
    p.add_argument("--num-workers", type=int,   default=4,      help="DataLoader workers")

    # Logging & checkpoints
    p.add_argument("--out-dir",    default="checkpoints/sft",   help="Checkpoint output directory")
    p.add_argument("--log-every",  type=int, default=10,        help="Log every N optimizer steps")
    p.add_argument("--eval-every", type=int, default=200,       help="Eval every N optimizer steps")
    p.add_argument("--save-every", type=int, default=500,       help="Save ckpt every N optimizer steps")
    p.add_argument("--eval-batches",type=int, default=50,       help="Max val batches per eval")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args)
