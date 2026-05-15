"""
data.py — Ultra-fast memmap DataLoader for pre-tokenized .bin files

Reads the flat uint16 binary files produced by download_cosmopedia.py.
Uses np.memmap so the OS pages data in from disk on demand — zero RAM
overhead, zero copy into PyTorch (via from_numpy).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MemmapDataset(Dataset):
    """
    Wraps a memory-mapped uint16 .bin file as a PyTorch Dataset.

    Each item is a (input, target) pair of length `seq_len`, where
    target = input shifted by 1 (standard language-model objective).

    The dataset is contiguous — item[i] starts at token offset i*seq_len.
    No shuffling at the item level (sequential reads = max disk throughput).
    Shuffling happens at the DataLoader level via a sampler if desired, but
    for pretraining sequential reads are generally preferred.
    """

    def __init__(self, bin_path: str, seq_len: int):
        assert os.path.exists(bin_path), f"File not found: {bin_path}"
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # -1 because targets are shifted by 1
        self.n_tokens = len(self.data)
        self.n_chunks = (self.n_tokens - 1) // seq_len
        print(
            f"[data] Loaded {bin_path}  |  "
            f"{self.n_tokens:,} tokens  |  "
            f"{self.n_chunks:,} chunks of {seq_len}"
        )

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        # Slice a numpy view — zero copy, no RAM allocation
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def make_loader(
    bin_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 16,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """
    Returns a DataLoader wrapping a MemmapDataset.

    num_workers=4 is usually optimal for NVMe SSDs.
    pin_memory=True allows async CPU→GPU DMA transfers.
    shuffle=False for pretraining (sequential reads are fastest).
    """
    ds = MemmapDataset(bin_path, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,               # keeps batch size constant
        persistent_workers=num_workers > 0,
    )
