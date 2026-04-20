"""Token-stream dataset built from tokenizer_output/{train,val}.npy.

Owner:
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_token_arrays(tokenizer_dir: str):
    """Return (train_ids, val_ids) as numpy arrays."""
    root = Path(tokenizer_dir)
    train_ids = np.load(root / "train.npy")
    val_ids = np.load(root / "val.npy")
    return train_ids.astype(np.int64), val_ids.astype(
        np.int64
    )  # pytorch simplification is reccomended for ids


def make_batches(ids, batch_size: int, seq_len: int, seed: int = 0):
    """Yield (input_ids, target_ids) batches for next-token prediction."""
    ids = ids.copy()  # avoid modifying original

    if ids.ndim != 1:
        raise ValueError(f"Expected array to be 1D, got shape {ids.shape} instead")
    if len(ids) < batch_size * seq_len + 1:
        raise ValueError(
            f"Not enough tokens to create a single batch of size {batch_size} and seq_len {seq_len}. Got {len(ids)} tokens."
        )

    # shuffling tokens around to prevent overfitting
    randomizer = np.random.default_rng(seed)
    max_start = len(ids) - seq_len - 1

    while True:
        # randomly select starting indices for every sequence
        start_ind = randomizer.integers(0, max_start + 1, size=batch_size)
        x = np.stack([ids[i : i + seq_len] for i in start_ind])
        y = np.stack([ids[s + 1 : s + seq_len + 1] for s in start_ind])

        yield torch.from_numpy(x), torch.from_numpy(y)
