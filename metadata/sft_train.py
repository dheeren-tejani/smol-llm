"""
sft_train.py — Supervised fine-tuning on top of the pretrained checkpoint
============================================================================
Usage:
    python sft_train.py --base-ckpt checkpoints/best.pt
    python sft_train.py --resume                          # continue an SFT run
    python sft_train.py --base-ckpt checkpoints/best.pt --epochs 2 --compile

Run prepare_sft_data.py first to produce data/train_sft.pt + data/val_sft.pt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SFT MISTAKES THIS SCRIPT SPECIFICALLY AVOIDS — read this before changing
defaults, most of these are "worked fine in a toy run, silently degraded
the real model" bugs:

  1. Loss on the prompt.  If you train on the whole sequence (prompt +
     response) with no masking, the model spends most of its gradient
     budget learning to predict tokens it will always be GIVEN at
     inference time, and effective response-side signal gets diluted.
     -> training.sft_tokenizer.encode_example masks everything except
        assistant content + the end-of-turn token (label = -100).

  2. Forgetting the stop token.  If the end-of-turn marker is never a
     training target, the model never learns when to stop — you get
     endless rambling at inference. -> <|end|> is explicitly UNMASKED
     as part of every assistant turn's loss.

  3. Reusing the pretraining LR / schedule.  6e-4 with a long warmup is
     tuned for training from random init over billions of tokens; reused
     verbatim for SFT it will wreck a converged model in a few hundred
     steps. -> max_lr defaults ~12x lower, short warmup, cosine decay.

  4. Reusing the pretraining optimizer state.  Adam's momentum/variance
     estimates from the end of pretraining were tuned for a totally
     different LR regime. -> SFT always starts a fresh AdamW optimizer.

  5. Too many epochs.  SFT datasets are tiny relative to pretraining
     corpora (thousands vs billions of examples) — memorization can set
     in within 1-2 epochs. -> default 3 epochs, val loss tracked every
     eval_every steps, best.pt always kept, optional early stopping.

  6. Naive sequence packing.  Concatenating unrelated conversations into
     one packed sequence (great for pretraining throughput) lets later
     conversations' attention leak into earlier ones unless you build a
     block-diagonal mask, which the fused SDPA call in model.py doesn't
     support. -> one conversation per sequence, right-padded (see
     training/sft_data.py's docstring for why this is safe under a
     purely causal mask with no attention-mask changes needed).

  7. Template injection via naive string formatting.  Concatenating a
     full chat-templated string and re-tokenizing it with special tokens
     "allowed" lets literal tag-like text INSIDE a message get parsed as
     a real control token. -> encode_example tokenizes each field with
     encode_ordinary and splices in real special-token ids itself; see
     the self-test at the bottom of training/sft_tokenizer.py.

  8. Sequential (non-shuffled) batches.  Fine for pretraining (one huge
     stream), bad for SFT (dataset order/clustering can bias each epoch).
     -> train loader shuffles every epoch (length-grouped, see below).

  9. Padding chosen for convenience, not efficiency.  Padding every batch
     to a fixed max_seq_len wastes huge amounts of compute on an
     instruction dataset with wildly varying response lengths. -> a
     length-grouped batch sampler clusters similar-length examples so
     per-batch padding stays small (see training/sft_data.py).

 10. Silent architecture mismatch.  Picking a --preset by hand for SFT
     risks loading pretrained weights into a differently-shaped model.
     -> this script always reconstructs ModelConfig from the checkpoint
     being loaded (base or resumed) — there is no --preset flag here.

 11. torch.compile + variable-length batches.  Padding dynamically means
     seq_len changes almost every step, and torch.compile's default
     static-shape mode recompiles on every new shape, which is slower
     than not compiling at all. -> compile defaults OFF; pass --compile
     to opt in (uses dynamic=True, still expect more recompiles than
     the fixed-shape pretraining loop).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import time
import math
import argparse
import dataclasses

import torch
import torch.nn.functional as F

from model      import GPT, ModelConfig
from sft_data   import make_sft_loader
from sft_tokenizer import (
    build_chat_tokenizer,
    render_prompt_for_generation,
    decode_response,
    MIN_VOCAB_SIZE_REQUIRED,
    END_ID,
)
from scheduler  import get_lr
from logger     import TrainingLogger
from checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# ─────────────────────────────────────────────────────────────
# SFT Config
# ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SFTConfig:
    # ── Data ─────────────────────────────────────────────────
    train_data:   str   = "data/train_sft.pt"
    val_data:     str   = "data/val_sft.pt"
    max_seq_len:  int   = 1024        # must be <= the base model's max_seq_len

    # ── Batch ────────────────────────────────────────────────
    batch_size:       int = 32
    grad_accum_steps: int = 8         # effective batch = batch_size * grad_accum_steps sequences

    # ── Optimizer ────────────────────────────────────────────
    # ~12x lower than typical pretraining max_lr — SFT nudges a converged
    # model, it doesn't train one from scratch.
    max_lr:        float = 5e-5
    min_lr:        float = 5e-6
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0
    label_smoothing: float = 0.0      # try 0.05-0.1 if the model gets overconfident

    # ── NEFTune (optional) ───────────────────────────────────
    # Noisy Embeddings Fine-Tuning (Jain et al. 2023): adds uniform noise
    # to token embeddings during training only. Often a free quality bump
    # on small instruction sets. 0 disables it. Typical range: 5-15.
    neftune_alpha: float = 0.0

    # ── Schedule ─────────────────────────────────────────────
    epochs:        int   = 3          # SFT is epoch-based, not token-budget-based
    warmup_ratio:  float = 0.03       # short warmup relative to the (much shorter) run

    # ── Checkpointing ────────────────────────────────────────
    ckpt_dir:      str   = "checkpoints_sft_v2"
    ckpt_every:    int   = 200
    keep_last_n:   int   = 3

    # ── Evaluation / early stopping ──────────────────────────
    eval_every:    int   = 200
    eval_batches:  int   = 40         # 0 = evaluate the full val set
    sample_every:  int   = 200
    early_stop_patience: int = 5      # eval events with no val_loss improvement; 0 disables

    # ── Logging ──────────────────────────────────────────────
    log_every:     int   = 10
    run_name:      str   = "smol-lm-sft"

    # ── System ───────────────────────────────────────────────
    num_workers:   int   = 4
    length_grouped_batching: bool = True
    compile:       bool  = False
    # See footgun #11 above for why this defaults off.


SAMPLE_CONVERSATIONS = [
    [{"role": "user", "content": "Explain what a black hole is in two sentences."}],
    [{"role": "user", "content": "Write a short poem about the ocean."}],
    [{"role": "system", "content": "You are a helpful assistant that answers concisely."},
     {"role": "user", "content": "What's 17 * 24?"}],
    [{"role": "user", "content": "Give me three tips for staying focused while studying."}],
]


# ─────────────────────────────────────────────────────────────
# NEFTune
# ─────────────────────────────────────────────────────────────

def apply_neftune(raw_model: torch.nn.Module, alpha: float):
    """
    Registers a forward hook on the token embedding layer that adds
    uniform noise, scaled per Jain et al. 2023, ONLY while the module is
    in training mode (so eval/generation are unaffected automatically).

    NOTE: this hook operates on the *eager* submodule. If you also pass
    --compile, torch.compile may fuse the embedding lookup in a way that
    this hook doesn't observe — the two features are not verified to
    compose. If you want NEFTune, leave --compile off.
    """
    if alpha <= 0:
        return

    def _hook(module, inputs, output):
        if not module.training:
            return output
        seq_len, d_model = output.shape[1], output.shape[2]
        mag = alpha / math.sqrt(seq_len * d_model)
        noise = torch.empty_like(output).uniform_(-mag, mag)
        return output + noise

    raw_model.token_embed.register_forward_hook(_hook)


# ─────────────────────────────────────────────────────────────
# Optimizer — fresh AdamW, see footgun #4
# ─────────────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, cfg: SFTConfig):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    decay_params    = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in raw.named_parameters() if p.dim() < 2]

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=True,
    )


# ─────────────────────────────────────────────────────────────
# Evaluation — val loss AND response-token accuracy
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size: int, max_batches: int = 0, label_smoothing: float = 0.0):
    model.eval()
    total_loss, total_correct, total_tokens, n_batches = 0.0, 0, 0, 0

    for x, y, _pad_frac in val_loader:
        if max_batches and n_batches >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # 1. Forward pass in low-precision (bf16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            
        # 2. Compute loss in stable full-precision (fp32) outside the autocast block
        micro_loss = F.cross_entropy(
            logits.float().view(-1, vocab_size), 
            y.view(-1),
            ignore_index=-100, 
            label_smoothing=label_smoothing,
        )

        mask = y.view(-1) != -100
        n_tok = mask.sum().item()
        if n_tok > 0:
            preds = logits.view(-1, vocab_size).argmax(-1)
            total_correct += (preds[mask] == y.view(-1)[mask]).sum().item()
            total_tokens  += n_tok
            total_loss    += micro_loss.item() # <── FIXED: Changed loss.item() to micro_loss.item()
            n_batches     += 1

    model.train()
    avg_loss = total_loss / n_batches if n_batches else float("nan")
    accuracy = total_correct / total_tokens if total_tokens else float("nan")
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────
# Chat generation — stops at <|end|>, unlike the fixed-length
# max_new_tokens sampler used during pretraining.
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_chat(raw_model, enc, device, messages, max_new_tokens=150,
                   temperature=0.7, top_k=40):
    prompt_ids = render_prompt_for_generation(messages, enc)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    start_len = idx.size(1)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -raw_model.cfg.max_seq_len:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = raw_model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits[..., MIN_VOCAB_SIZE_REQUIRED:] = float("-inf")
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
        if next_tok.item() == END_ID:
            break

    gen_ids = idx[0, start_len:].tolist()
    return decode_response(gen_ids, enc)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", type=str, default=None,
                    help="Pretrained checkpoint to initialize from (required unless --resume finds an SFT checkpoint)")
    p.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint in ckpt_dir")
    p.add_argument("--ckpt", type=str, default=None, help="Resume from a specific SFT checkpoint path")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--neftune-alpha", type=float, default=None)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True); see footgun #11")
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--val-data", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SFTConfig()

    if args.epochs is not None:        cfg.epochs = args.epochs
    if args.max_lr is not None:        cfg.max_lr = args.max_lr
    if args.batch_size is not None:    cfg.batch_size = args.batch_size
    if args.neftune_alpha is not None: cfg.neftune_alpha = args.neftune_alpha
    if args.train_data is not None:    cfg.train_data = args.train_data
    if args.val_data is not None:      cfg.val_data = args.val_data
    cfg.compile = args.compile

    # ── Resolve which checkpoint to load architecture + weights from ──
    resume_path = args.ckpt
    if args.resume and resume_path is None and os.path.isdir(cfg.ckpt_dir):
        # (checkpoint.py's find_latest_checkpoint assumes ckpt_dir already
        # exists; guard here so a first-ever --resume doesn't hard-crash)
        resume_path = find_latest_checkpoint(cfg.ckpt_dir)

    if resume_path is None and args.base_ckpt is None:
        raise SystemExit(
            "Provide --base-ckpt <path to pretrained checkpoint> to start a new "
            "SFT run, or --resume to continue an existing one in "
            f"'{cfg.ckpt_dir}'."
        )

    arch_source = resume_path if resume_path is not None else args.base_ckpt

    # ── Logger ────────────────────────────────────────────────
    logger = TrainingLogger(run_name=cfg.run_name)

    # ── Device ────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Reconstruct architecture EXACTLY from the checkpoint (footgun #10) ──
    logger.info(f"\nReading architecture from {arch_source} ...")
    peek = torch.load(arch_source, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**peek["config"])
    del peek

    if model_cfg.vocab_size < MIN_VOCAB_SIZE_REQUIRED:
        raise SystemExit(
            f"model_cfg.vocab_size={model_cfg.vocab_size} is smaller than the "
            f"{MIN_VOCAB_SIZE_REQUIRED} needed for the chat special tokens. "
            f"This checkpoint wasn't pretrained with room for them."
        )
    if cfg.max_seq_len > model_cfg.max_seq_len:
        raise SystemExit(
            f"SFTConfig.max_seq_len ({cfg.max_seq_len}) exceeds the base model's "
            f"max_seq_len ({model_cfg.max_seq_len}) — RoPE's cache won't cover it. "
            f"Lower max_seq_len or re-run prepare_sft_data.py with a shorter one."
        )

    cfg_snapshot = {**dataclasses.asdict(model_cfg), **dataclasses.asdict(cfg)}
    logger.log_config(cfg_snapshot)

    # ── Model ─────────────────────────────────────────────────
    model = GPT(model_cfg).to(device)
    logger.info(f"Parameters: {model.count_params(non_embedding=True)/1e6:.1f}M (non-embedding)")

    raw_model_for_hooks = model   # NEFTune must be attached before torch.compile
    apply_neftune(raw_model_for_hooks, cfg.neftune_alpha)
    if cfg.neftune_alpha > 0:
        logger.info(f"NEFTune enabled: alpha={cfg.neftune_alpha}")

    if cfg.compile:
        logger.info("Compiling model (dynamic=True, see footgun #11 for why)...")
        model = torch.compile(model, dynamic=True)

    optimizer = build_optimizer(model, cfg)

    # ── Load weights (fresh SFT run) or full state (resume) ────
    start_step  = 0
    tokens_seen = 0
    if resume_path is not None:
        logger.info(f"\nResuming SFT run from {resume_path} ...")
        ckpt = load_checkpoint(resume_path, model, optimizer, device=str(device))
        start_step  = ckpt["step"] + 1
        tokens_seen = ckpt.get("tokens_seen", 0)
        logger.info(f"  Resumed at step {start_step:,}")
    else:
        logger.info(f"\nInitializing from pretrained checkpoint {args.base_ckpt} "
                     f"(model weights only — fresh AdamW state, see footgun #4)")
        load_checkpoint(args.base_ckpt, model, optimizer=None, device=str(device))
        logger.warning(
            "5 chat special-token embeddings (ids 50257-50261) were reserved but "
            "never trained during pretraining. SFT will train them from scratch "
            "over the next few hundred steps — expected, low-risk (5 of "
            f"{model_cfg.vocab_size} rows)."
        )

    # ── Data ──────────────────────────────────────────────────
    logger.info("\nLoading SFT data...")
    train_loader = make_sft_loader(
        cfg.train_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        shuffle_group_by_length=cfg.length_grouped_batching, seed=1234,
    )
    val_loader = make_sft_loader(
        cfg.val_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        shuffle_group_by_length=False,
    )

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = max(1, int(cfg.warmup_ratio * total_steps))

    logger.log_start(total_steps, total_steps * cfg.batch_size * cfg.grad_accum_steps * cfg.max_seq_len)
    logger.info(f"Steps/epoch: {steps_per_epoch:,}  |  Epochs: {cfg.epochs}  |  "
                f"Total steps: {total_steps:,}  |  Warmup steps: {warmup_steps:,}")

    if start_step >= total_steps:
        logger.warning(
            f"start_step ({start_step:,}) >= total_steps ({total_steps:,}) — "
            "nothing to do. Increase --epochs if you want to keep training."
        )
        logger.close()
        return

    # ── Chat tokenizer for sampling ─────────────────────────────
    enc = build_chat_tokenizer()

    # ── Training loop ────────────────────────────────────────────
    model.train()
    train_iter = iter(train_loader)

    t_run_start   = time.perf_counter()
    best_val_loss = float("inf")
    evals_since_improve = 0

    for step in range(start_step, total_steps):
        lr = get_lr(step, warmup_steps=warmup_steps, total_steps=total_steps,
                    max_lr=cfg.max_lr, min_lr=cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        accum_correct, accum_total = 0, 0
        accum_pad_frac = 0.0
        micro_steps = 0

        for _ in range(cfg.grad_accum_steps):
            try:
                x, y, pad_frac = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)   # reshuffles (see LengthGroupedSampler)
                x, y, pad_frac = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                micro_loss = F.cross_entropy(
                    logits.view(-1, model_cfg.vocab_size), y.view(-1),
                    ignore_index=-100, label_smoothing=cfg.label_smoothing,
                )

            (micro_loss / cfg.grad_accum_steps).backward()
            accum_loss += micro_loss.item()

            with torch.no_grad():
                mask = y.view(-1) != -100
                preds = logits.view(-1, model_cfg.vocab_size).argmax(-1)
                accum_correct += (preds[mask] == y.view(-1)[mask]).sum().item()
                accum_total   += mask.sum().item()
            accum_pad_frac += pad_frac
            micro_steps += 1
            tokens_seen += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000

        # ── Logging ───────────────────────────────────────────
        if step % cfg.log_every == 0:
            train_acc = accum_correct / accum_total if accum_total else float("nan")
            logger.log_step(
                step=step, loss=accum_loss / micro_steps, lr=lr,
                tok_per_sec=(cfg.batch_size * cfg.grad_accum_steps * cfg.max_seq_len) / (t1 - t0),
                grad_norm=grad_norm, elapsed_sec=time.perf_counter() - t_run_start,
                tokens_seen=tokens_seen, step_ms=step_ms,
                extra={"train_acc": round(train_acc, 4)} if not math.isnan(train_acc) else None,
            )
            logger.info(
                f"    (epoch {step // steps_per_epoch + 1}/{cfg.epochs}  "
                f"resp_tok_acc={train_acc*100:5.1f}%  "
                f"avg_pad_frac={accum_pad_frac/micro_steps*100:4.1f}%)"
            )

        # ── Sampling ─────────────────────────────────────────
        if step % cfg.sample_every == 0 and step > start_step:
            logger.info(f"\n{'─'*30} SAMPLING step {step:,} {'─'*30}")
            model.eval()
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            for convo in SAMPLE_CONVERSATIONS:
                reply = generate_chat(raw_model, enc, device, convo)
                logger.info(f"Prompt: {convo[-1]['content']!r}")
                logger.info(f"Reply : {reply!r}\n")
            model.train()

        # ── Evaluation + early stopping ────────────────────────
        if step % cfg.eval_every == 0 and step > start_step:
            val_loss, val_acc = evaluate(model, val_loader, device,
                                          model_cfg.vocab_size, cfg.eval_batches, label_smoothing=cfg.label_smoothing)
            logger.log_eval(step=step, val_loss=val_loss, train_loss=accum_loss / micro_steps)
            logger.info(f"    val_response_token_accuracy: {val_acc*100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_since_improve = 0
            else:
                evals_since_improve += 1

            ckpt_path = save_checkpoint(
                step=step, model=model, optimizer=optimizer, model_cfg=model_cfg,
                train_cfg=cfg, tokens_seen=tokens_seen, ckpt_dir=cfg.ckpt_dir,
                val_loss=val_loss, keep_last_n=cfg.keep_last_n,
            )
            logger.log_checkpoint(step=step, path=ckpt_path)

            if cfg.early_stop_patience > 0 and evals_since_improve >= cfg.early_stop_patience:
                logger.info(
                    f"\nEarly stopping: val_loss hasn't improved in "
                    f"{evals_since_improve} eval events (best={best_val_loss:.4f}). "
                    f"best.pt in {cfg.ckpt_dir}/ already holds the best checkpoint."
                )
                break

        elif step % cfg.ckpt_every == 0 and step > start_step:
            ckpt_path = save_checkpoint(
                step=step, model=model, optimizer=optimizer, model_cfg=model_cfg,
                train_cfg=cfg, tokens_seen=tokens_seen, ckpt_dir=cfg.ckpt_dir,
                val_loss=None, keep_last_n=cfg.keep_last_n,
            )
            logger.log_checkpoint(step=step, path=ckpt_path)

    # ── Final eval + checkpoint ───────────────────────────────
    final_step = min(step, total_steps - 1)
    val_loss, val_acc = evaluate(model, val_loader, device, model_cfg.vocab_size, 0)
    logger.log_eval(step=final_step, val_loss=val_loss)
    logger.info(f"    final val_response_token_accuracy: {val_acc*100:.2f}%")

    save_checkpoint(
        step=final_step, model=model, optimizer=optimizer, model_cfg=model_cfg,
        train_cfg=cfg, tokens_seen=tokens_seen, ckpt_dir=cfg.ckpt_dir,
        val_loss=val_loss, keep_last_n=cfg.keep_last_n,
    )

    logger.log_end(elapsed_sec=time.perf_counter() - t_run_start, tokens_seen=tokens_seen)
    logger.info(f"\nBest val_loss: {best_val_loss:.4f}  →  {cfg.ckpt_dir}/best.pt")
    logger.close()


if __name__ == "__main__":
    main()