"""
prepare_sft_data.py
====================
Turns a raw instruction/chat dataset into tokenized, loss-masked
(input_ids, labels) pairs ready for sft_train.py.

Supported sources (--dataset):
  smoltalk    HuggingFaceTB/smol-smoltalk  — The gold standard mix for <1B models.
                                              Automatically filters out complex math, 
                                              multi-turn loops, and long text.
  no_robots   HuggingFaceH4/no_robots      — small, hand-curated, already
                                              in {"messages": [...]} format.
  alpaca      tatsu-lab/alpaca             — classic single-turn
                                              instruction/output pairs.
  custom      --custom-jsonl PATH          — your own data.
"""

import os
import json
import time
import random
import hashlib
import argparse
import multiprocessing as mp

from tqdm import tqdm

from sft_tokenizer import (
    build_chat_tokenizer,
    encode_example,
    validate_messages,
    MIN_VOCAB_SIZE_REQUIRED,
)

OUT_DIR = "data"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_VAL_FRACTION = 0.02   # only used when the source has no built-in split

# ─────────────────────────────────────────────────────────────────────────
# Dataset adapters 
# ─────────────────────────────────────────────────────────────────────────

def load_smoltalk():
    from datasets import load_dataset
    print("  Downloading HuggingFaceTB/smol-smoltalk (curated for <1B models)...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk")
    return {"train": ds["train"], "val": None}

def load_no_robots():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots")
    return {"train": ds["train_sft"], "val": ds["test_sft"]}

def load_alpaca():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca")
    return {"train": ds["train"], "val": None}

def load_custom(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {"train": rows, "val": None}

DATASET_LOADERS = {
    "smoltalk": load_smoltalk,
    "no_robots": load_no_robots,
    "alpaca": load_alpaca,
}

# ─────────────────────────────────────────────────────────────────────────
# Row -> messages adapters
# ─────────────────────────────────────────────────────────────────────────

def row_to_messages_smoltalk(row) -> list:
    messages = row.get("messages", [])
    
    # 1. Enforce single-turn (User -> Assistant or System -> User -> Assistant)
    if len(messages) not in (2, 3):
        return [] # Returning empty list triggers the "invalid" drop downstream
        
    # 2. Token/Word length limits to prevent capacity crowding
    for m in messages:
        if len(m.get("content", "").split()) > 200:
            return []
            
    # 3. Drop heavy code execution or AI alignment refusals
    assistant_content = messages[-1].get("content", "")
    forbidden = ["def ", "import ", "class ", "```python", "As an AI", "I cannot fulfill"]
    if any(x in assistant_content for x in forbidden):
        return []
        
    return [{"role": m["role"], "content": m["content"]} for m in messages]

def row_to_messages_no_robots(row) -> list:
    return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]

def row_to_messages_alpaca(row) -> list:
    instruction = row["instruction"].strip()
    extra_input = row.get("input", "").strip()
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"].strip()},
    ]

def row_to_messages_custom(row) -> list:
    return row["messages"]

ROW_ADAPTERS = {
    "smoltalk": row_to_messages_smoltalk,
    "no_robots": row_to_messages_no_robots,
    "alpaca": row_to_messages_alpaca,
    "custom": row_to_messages_custom,
}

# ─────────────────────────────────────────────────────────────────────────
# Parallel tokenization
# ─────────────────────────────────────────────────────────────────────────

_enc = None
_max_len = None

def _worker_init(max_len: int):
    global _enc, _max_len
    _enc = build_chat_tokenizer()
    _max_len = max_len

def _process_one(messages):
    """Validate -> tokenize -> return (input_ids, labels, n_loss_tokens) or None."""
    if not validate_messages(messages):
        return "invalid"
    out = encode_example(messages, _enc, _max_len)
    if out is None:
        return "too_long"
    input_ids, labels = out
    n_loss = sum(1 for t in labels if t != -100)
    return {"input_ids": input_ids, "labels": labels, "n_loss": n_loss}

def _dedup_key(messages) -> str:
    canon = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()

def process_split(all_messages: list, max_len: int, num_workers: int, desc: str):
    """Dedup, then tokenize in parallel. Returns (examples, stats dict)."""
    seen = set()
    deduped = []
    n_dupe = 0
    for m in all_messages:
        key = _dedup_key(m)
        if key in seen:
            n_dupe += 1
            continue
        seen.add(key)
        deduped.append(m)

    examples = []
    n_invalid = 0
    n_too_long = 0
    n_loss_tokens = 0
    n_total_tokens = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(max_len,)) as pool:
        for result in tqdm(pool.imap(_process_one, deduped, chunksize=64),
                            total=len(deduped), desc=desc, dynamic_ncols=True):
            if result == "invalid":
                n_invalid += 1
            elif result == "too_long":
                n_too_long += 1
            else:
                examples.append({"input_ids": result["input_ids"], "labels": result["labels"]})
                n_loss_tokens += result["n_loss"]
                n_total_tokens += len(result["input_ids"])

    stats = {
        "n_input": len(all_messages),
        "n_dupe_dropped": n_dupe,
        "n_invalid_dropped": n_invalid,
        "n_too_long_dropped": n_too_long,
        "n_kept": len(examples),
        "n_total_tokens": n_total_tokens,
        "n_loss_tokens": n_loss_tokens,
        "loss_token_frac": (n_loss_tokens / n_total_tokens) if n_total_tokens else 0.0,
    }
    return examples, stats

def print_stats(name: str, stats: dict):
    print(f"\n[{name}]")
    print(f"  input examples      : {stats['n_input']:,}")
    print(f"  dropped (duplicate) : {stats['n_dupe_dropped']:,}")
    print(f"  dropped (malformed) : {stats['n_invalid_dropped']:,}")
    print(f"  dropped (too long)  : {stats['n_too_long_dropped']:,}")
    print(f"  kept                : {stats['n_kept']:,}")
    if stats["n_kept"]:
        print(f"  total tokens        : {stats['n_total_tokens']:,}")
        print(f"  tokens w/ loss      : {stats['n_loss_tokens']:,} "
              f"({stats['loss_token_frac']*100:.1f}% of all tokens)")

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="smoltalk",
                    choices=["smoltalk", "no_robots", "alpaca", "custom"])
    p.add_argument("--custom-jsonl", type=str, default=None,
                    help="Path to a .jsonl file (required if --dataset custom)")
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                    help="Used only when the source dataset has no built-in val split")
    p.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  SFT DATA PREP")
    print("=" * 60)
    print(f"  Dataset       : {args.dataset}")
    print(f"  Max seq len   : {args.max_seq_len}")
    print(f"  Workers       : {args.num_workers}")
    print(f"  Required vocab: >= {MIN_VOCAB_SIZE_REQUIRED} "
          f"(check this matches your ModelConfig.vocab_size)")
    print()

    # ── Phase 1: load raw rows ──────────────────────────────────────────
    t0 = time.time()
    if args.dataset == "custom":
        if not args.custom_jsonl:
            raise SystemExit("--custom-jsonl is required when --dataset custom")
        raw = load_custom(args.custom_jsonl)
    else:
        raw = DATASET_LOADERS[args.dataset]()
    print(f"  Loaded raw data in {time.time()-t0:.1f}s")

    adapter = ROW_ADAPTERS[args.dataset]
    train_messages = [adapter(r) for r in raw["train"]]

    if raw["val"] is not None:
        val_messages = [adapter(r) for r in raw["val"]]
    else:
        # Dynamic validation split: ensuring at least 500 samples exist for stable evaluation
        random.shuffle(train_messages)
        n_val = min(len(train_messages) - 1, max(500, int(len(train_messages) * args.val_fraction)))
        val_messages = train_messages[:n_val]
        train_messages = train_messages[n_val:]
        print(f"  No built-in val split — carved out {n_val:,} random "
              f"examples for validation.")

    # ── Phase 2: validate + dedup + tokenize (parallel) ─────────────────
    train_examples, train_stats = process_split(
        train_messages, args.max_seq_len, args.num_workers, desc="Tokenizing train"
    )
    val_examples, val_stats = process_split(
        val_messages, args.max_seq_len, args.num_workers, desc="Tokenizing val"
    )

    print_stats("train", train_stats)
    print_stats("val", val_stats)

    if train_stats["n_kept"] == 0:
        raise SystemExit("\n[ERROR] Zero usable training examples after filtering. "
                          "Check your --dataset / --custom-jsonl formatting.")

    if val_stats["n_kept"] == 0:
        print("\n[WARN] Zero usable val examples — eval/best-checkpoint tracking "
              "in sft_train.py will not work properly.")

    # ── Phase 3: save ────────────────────────────────────────────────────
    import torch 

    meta_common = {
        "dataset": args.dataset,
        "max_seq_len": args.max_seq_len,
        "min_vocab_size_required": MIN_VOCAB_SIZE_REQUIRED,
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    train_path = os.path.join(args.out_dir, "train_sft.pt")
    val_path = os.path.join(args.out_dir, "val_sft.pt")

    torch.save({"examples": train_examples, "meta": {**meta_common, "split": "train", **train_stats}}, train_path)
    torch.save({"examples": val_examples, "meta": {**meta_common, "split": "val", **val_stats}}, val_path)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  {train_path}  ({train_stats['n_kept']:,} examples)")
    print(f"  {val_path}  ({val_stats['n_kept']:,} examples)")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()