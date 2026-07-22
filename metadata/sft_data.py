"""
training/sft_data.py — Dataset + batching for SFT
====================================================
Loads the pre-tokenized examples written by prepare_sft_data.py and
serves them as padded batches.

── Design choice: pad, don't pack ──────────────────────────────────
Pretraining (data.py) packs tokens into dense fixed-length chunks —
correct there, because every chunk is one continuous stream and a
document boundary mid-chunk is a non-issue for a next-token objective.

For SFT that same trick is a classic bug magnet: if you concatenate
several unrelated conversations into one packed sequence, the causal
attention in model.py has no notion of document boundaries, so tokens
from conversation B can attend to (and be predicted from) conversation
A's tokens sitting right before them in the same packed sequence.
Fixing that properly requires a block-diagonal attention mask, which
model.py's fused F.scaled_dot_product_attention(..., is_causal=True)
call doesn't support without surgery.

So instead: ONE conversation per training sequence, right-padded to
the batch's longest example. Because attention is causal, a real token
at position t can never attend to padding at position > t — padding is
always in the "future" relative to every real token, right up until we
run off the end of the real content, at which point the row's outputs
are pure padding-on-padding noise that we simply never compute loss on
(labels are -100 there). No custom attention mask needed, no leakage
between examples. The only cost is some wasted compute on pad tokens,
which the length-grouped sampler below minimizes.
"""

import math
import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from sft_tokenizer import PAD_ID


class SFTDataset(Dataset):
    """
    Wraps the list of {"input_ids": [...], "labels": [...]} examples
    produced by prepare_sft_data.py. Small/medium instruction datasets
    fit comfortably in RAM (unlike the multi-GB pretraining corpus), so
    unlike data.py's memmap approach we just load everything up front.
    """

    def __init__(self, path: str):
        blob = torch.load(path, weights_only=False)
        self.examples = blob["examples"]
        self.meta = blob.get("meta", {})
        self.lengths = [len(ex["input_ids"]) for ex in self.examples]

        if not self.examples:
            raise ValueError(f"{path} contains zero examples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels = torch.tensor(ex["labels"], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch, pad_id: int = PAD_ID):
    """
    Right-pad every sequence in the batch to the batch's longest length.
    input_ids padded with pad_id, labels padded with -100 (ignored by
    F.cross_entropy's default ignore_index).
    """
    max_len = max(x.size(0) for x, _ in batch)
    B = len(batch)

    x_out = torch.full((B, max_len), pad_id, dtype=torch.long)
    y_out = torch.full((B, max_len), -100, dtype=torch.long)

    n_real_tokens = 0
    for i, (x, y) in enumerate(batch):
        L = x.size(0)
        x_out[i, :L] = x
        y_out[i, :L] = y
        n_real_tokens += L

    # How much of the batch is real content vs padding — useful for
    # sanity-checking the length-grouped sampler is actually helping.
    pad_frac = 1.0 - (n_real_tokens / (B * max_len))
    return x_out, y_out, pad_frac


class LengthGroupedSampler(Sampler):
    """
    Shuffles the dataset, then groups nearby-length examples into the
    same batch to cut padding waste (a well-known trick, e.g. HF
    Trainer's `group_by_length`). Concretely: shuffle all indices, chunk
    into "mega-batches" of size batch_size * mega_batch_mult, sort each
    mega-batch by length, slice into real batches, then shuffle the
    *order* the batches are yielded in (so training doesn't see a
    short-to-long curriculum, just less padding per batch).

    Re-shuffles every epoch (DataLoader calls __iter__ fresh each time).
    """

    def __init__(self, lengths, batch_size: int, mega_batch_mult: int = 50, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_batch_mult = max(1, mega_batch_mult)
        self.seed = seed
        self.epoch = 0
        self._n_batches = math.ceil(len(lengths) / batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)

        mega = self.batch_size * self.mega_batch_mult
        batches = []
        for i in range(0, len(idx), mega):
            chunk = idx[i:i + mega]
            chunk.sort(key=lambda j: self.lengths[j], reverse=True)
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])

        rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)


def make_sft_loader(
    path: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle_group_by_length: bool = True,
    seed: int = 0,
) -> DataLoader:
    """
    Returns (DataLoader, dataset). Training loader uses the length-grouped
    batch sampler; pass shuffle_group_by_length=False for the val loader
    (plain sequential batches — eval order doesn't matter and this keeps
    the eval loop simple/deterministic).
    """
    ds = SFTDataset(path)

    if shuffle_group_by_length:
        batch_sampler = LengthGroupedSampler(ds.lengths, batch_size, seed=seed)
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )