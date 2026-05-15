"""
scheduler.py — Cosine LR schedule with linear warmup

Standard for LLM pretraining (GPT-3, LLaMA, Chinchilla all use this).

  lr(step) =
    step / warmup_steps * max_lr           if step < warmup_steps
    min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π * decay_ratio))  otherwise
"""

import math


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Returns the learning rate for a given step.

    Args:
        step:          Current training step (0-indexed).
        warmup_steps:  Number of linear warmup steps.
        total_steps:   Total training steps (for cosine decay end point).
        max_lr:        Peak learning rate after warmup.
        min_lr:        Minimum LR at the end of cosine decay.
                       Typically max_lr / 10.
    """
    # 1. Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. After decay — clamp to min_lr
    if step >= total_steps:
        return min_lr

    # 3. Cosine decay
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
