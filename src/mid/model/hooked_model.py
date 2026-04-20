"""HookedTransformer factory for the Shakespeare model.

Owner: Mike C (build) & Cole S (load)
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path

from mid.config import ModelConfig


def build_model(cfg: ModelConfig):
    """Construct an untrained HookedTransformer from a ModelConfig."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    hooked_cfg = HookedTransformerConfig(**cfg.to_dict())
    model = HookedTransformer(hooked_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_checkpoint(path: str, model_cfg: ModelConfig):
    """Load a trained HookedTransformer from disk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_cfg)
    state_dict = torch.load(Path(path), map_location=device)
    # Safeguard for wrapped checkpoints
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # We can set to false if we want to load partial models or don't care for missing or unexpected keys.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"WARNING! Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"WARNING! Unexpected keys when loading checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model
