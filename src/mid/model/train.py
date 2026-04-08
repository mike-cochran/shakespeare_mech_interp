"""nanoGPT-style training loop for the HookedTransformer. Logs loss, perplexity, attention pattern snapshots.

Owner:
"""

from __future__ import annotations

from mid.config import ModelConfig, TrainConfig


def train(model_cfg: ModelConfig, train_cfg: TrainConfig, tokenizer_dir: str, out_dir: str):
    raise NotImplementedError
