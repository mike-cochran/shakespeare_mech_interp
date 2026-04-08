"""Token-stream dataset built from tokenizer_output/{train,val}.npy.

Owner:
"""

from __future__ import annotations


def load_token_arrays(tokenizer_dir: str):
    """Return (train_ids, val_ids) as numpy arrays."""
    raise NotImplementedError


def make_batches(ids, batch_size: int, seq_len: int, seed: int = 0):
    """Yield (input_ids, target_ids) batches for next-token prediction."""
    raise NotImplementedError
