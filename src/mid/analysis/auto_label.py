"""Auto-label SAE features by sending top-activating contexts to an LLM.

Owner:
"""

from __future__ import annotations


def top_activating_contexts(sae, activations, k: int = 20):
    raise NotImplementedError


def label_features(top_contexts, llm_client) -> dict[int, str]:
    raise NotImplementedError
