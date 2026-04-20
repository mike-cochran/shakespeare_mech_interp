"""Shared Anthropic client wrapper for analysis modules.

Owner: Areeb

Thin I/O layer. Callers (neuron_baseline.py for monosemanticity
scoring, auto_label.py for feature labeling) bring their own prompts
and parse the response themselves. Reads ANTHROPIC_API_KEY from the
environment.
"""

from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"


def call_anthropic(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8,
) -> str:
    """Send a single-message completion and return the stripped text."""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()
