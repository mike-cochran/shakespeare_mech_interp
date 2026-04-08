"""Activation patching and SAE feature steering.

Owner:
"""

from __future__ import annotations


def patch_activation(model, clean_input, corrupted_input, hook_name: str, position: int):
    raise NotImplementedError


def steer_with_feature(model, sae, feature_idx: int, coefficient: float, prompt: str):
    raise NotImplementedError
