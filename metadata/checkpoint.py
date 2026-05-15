"""
checkpoint.py — Save and resume training checkpoints

Saves:
  checkpoints/
    step_001000.pt   ← full checkpoint (model + optimizer + scaler + step)
    latest.pt        ← symlink → most recent checkpoint (fast resume)
    best.pt          ← lowest val_loss so far

Checkpoint dict schema:
  {
    "step"       : int,
    "model"      : state_dict,
    "optimizer"  : state_dict,
    "config"     : ModelConfig.__dict__,
    "train_cfg"  : TrainConfig.__dict__,
    "val_loss"   : float | None,
    "tokens_seen": int,
  }
"""

import os
import torch
from typing import Optional


def save_checkpoint(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_cfg,
    train_cfg,
    tokens_seen: int,
    ckpt_dir: str = "checkpoints",
    val_loss: Optional[float] = None,
    keep_last_n: int = 3,
) -> str:
    """
    Save a full training checkpoint.

    Returns the path of the saved file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Unwrap torch.compile if needed
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod

    payload = {
        "step"        : step,
        "model"       : raw_model.state_dict(),
        "optimizer"   : optimizer.state_dict(),
        "config"      : vars(model_cfg),
        "train_cfg"   : vars(train_cfg),
        "val_loss"    : val_loss,
        "tokens_seen" : tokens_seen,
    }

    path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    torch.save(payload, path)

    # ── latest symlink ──────────────────────────────────────
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)

    # ── best checkpoint ─────────────────────────────────────
    if val_loss is not None:
        best_marker = os.path.join(ckpt_dir, "_best_val_loss.txt")
        best_loss = float("inf")
        if os.path.exists(best_marker):
            with open(best_marker) as f:
                best_loss = float(f.read().strip())

        if val_loss < best_loss:
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(payload, best_path)
            with open(best_marker, "w") as f:
                f.write(str(val_loss))

    # ── prune old checkpoints (keep_last_n) ─────────────────
    _prune_old(ckpt_dir, keep_last_n)

    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer + scaler).

    Returns the full checkpoint dict so the caller can restore
    step, tokens_seen, val_loss, etc.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Unwrap compiled model if needed
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod

    raw_model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to the most recent checkpoint, or None if none exist."""
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    # Fallback: scan for step_*.pt files
    files = sorted(
        f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")
    )
    if files:
        return os.path.join(ckpt_dir, files[-1])
    return None


# ── Internal ──────────────────────────────────────────────────

def _prune_old(ckpt_dir: str, keep_last_n: int):
    """Delete old step_*.pt files, keeping the N most recent."""
    files = sorted(
        f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")
    )
    to_delete = files[: max(0, len(files) - keep_last_n)]
    for f in to_delete:
        try:
            os.remove(os.path.join(ckpt_dir, f))
        except OSError:
            pass
