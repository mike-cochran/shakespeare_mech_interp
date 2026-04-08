"""Cache HookedTransformer activations to disk for SAE training. Streams the Shakespeare corpus through the trained model and
saves per-layer activations at the configured hook points.

Owner:
"""

from __future__ import annotations


def cache_activations(
    checkpoint_path: str, hook_names: list[str], out_dir: str, batch_size: int = 32
):
    raise NotImplementedError
