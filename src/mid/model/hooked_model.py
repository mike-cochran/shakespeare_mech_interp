"""HookedTransformer factory for the Shakespeare model.

Owner:
"""

from __future__ import annotations

from mid.config import ModelConfig


def build_model(cfg: ModelConfig):
    """Construct an untrained HookedTransformer from a ModelConfig."""
    raise NotImplementedError


def load_checkpoint(path: str):
    """Load a trained HookedTransformer from disk."""
    raise NotImplementedError
