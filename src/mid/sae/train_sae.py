"""SAELens training entry point. Consumes cached activations from `mid.sae.activations`.

Owner:
"""

from __future__ import annotations

from mid.config import SAEConfig


def train_sae(cfg: SAEConfig, activations_dir: str, out_dir: str):
    raise NotImplementedError
