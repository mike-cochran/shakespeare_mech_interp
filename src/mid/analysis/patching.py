"""Activation patching and SAE feature steering.

Owner: Areeb
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer


def patch_activation(
    model: HookedTransformer,
    clean_input: str,
    corrupted_input: str,
    hook_name: str,
    position: int,
) -> float:
    """Return patched_loss - clean_loss after patching one position.

    The activation at (hook_name, position) from the corrupted run is
    spliced into the clean run. Positive return values mean the patched
    position was load-bearing for the clean prediction.
    """
    clean_tokens = model.to_tokens(clean_input)
    corrupted_tokens = model.to_tokens(corrupted_input)

    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    corrupted_act = corrupted_cache[hook_name]

    clean_loss = model(clean_tokens, return_type="loss").item()

    def patch_hook(act, hook):  # noqa: ARG001
        act[:, position, :] = corrupted_act[:, position, :]
        return act

    patched_loss = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(hook_name, patch_hook)],
        return_type="loss",
    ).item()

    return patched_loss - clean_loss


def steer_with_feature(
    model: HookedTransformer,
    sae,
    feature_idx: int,
    coefficient: float,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Greedy-generate text with coefficient * sae.W_dec[feature_idx] added at the SAE hook.

    sae must expose .W_dec (shape [n_features, d_model]) and
    .cfg.metadata.hook_name (SAELens v6 StandardSAEConfig).
    """
    steering_dir = sae.W_dec[feature_idx].detach().clone()
    hook_name = sae.cfg.metadata.hook_name

    def steering_hook(act, hook):  # noqa: ARG001
        act[:, :, :] += coefficient * steering_dir.to(act.device)
        return act

    input_tokens = model.to_tokens(prompt)

    with torch.no_grad():
        tokens = input_tokens.clone()
        for _ in range(max_new_tokens):
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, steering_hook)],
                return_type="logits",
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)

    return model.to_string(tokens[0])


def compare_outputs(
    model: HookedTransformer,
    prompt: str,
    hook_name: str,
    patch_fn: Callable,
    top_k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Clean vs. patched top-k next-token predictions on prompt.

    patch_fn is a TransformerLens forward hook applied at hook_name.
    Returns {"clean": [(tok, p), ...], "patched": [(tok, p), ...]}.
    """
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        clean_logits = model(tokens, return_type="logits")
        patched_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, patch_fn)],
            return_type="logits",
        )

    result = {}
    for label, logits in [("clean", clean_logits), ("patched", patched_logits)]:
        probs = logits[0, -1, :].softmax(dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        result[label] = [
            (model.to_single_str_token(tid.item()), p.item()) for tid, p in zip(top_ids, top_probs)
        ]

    return result
