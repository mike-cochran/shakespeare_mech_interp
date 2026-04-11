"""HookedTransformer factory for the Shakespeare model.

Owner:
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from mid.config import ModelConfig


def build_model(cfg: ModelConfig):
    """Construct an untrained HookedTransformer from a ModelConfig."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    hooked_cfg = HookedTransformerConfig(**cfg.to_dict())
    model = HookedTransformer(hooked_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def load_checkpoint(path: str):
    """Load a trained HookedTransformer from disk."""
    raise NotImplementedError
