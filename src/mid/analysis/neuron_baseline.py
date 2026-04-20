"""Raw neuron-level interpretability baseline for comparison against SAE features.

Owner: Areeb
"""

from __future__ import annotations

import anthropic
import statistics
from collections import Counter

import torch
from transformer_lens import HookedTransformer


def top_activating_neurons(
    model: HookedTransformer,
    dataset_tokens: torch.Tensor,
    hook_name: str,
    k: int = 20,
    context_window: int = 8,
) -> dict[int, list[tuple[str, float]]]:
    """For each neuron at ``hook_name``, return the top-k activating contexts.

    ``dataset_tokens`` has shape ``(batch, seq)``. Returns
    ``{neuron_idx: [(snippet_text, activation_value), ...]}`` sorted by
    descending activation.
    """
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(dataset_tokens, names_filter=hook_name)
    acts = cache[hook_name]

    batch, seq, n_neurons = acts.shape
    flat = acts.reshape(-1, n_neurons)

    top_vals, top_pos = flat.topk(k, dim=0)

    result: dict[int, list[tuple[str, float]]] = {}
    for neuron_idx in range(n_neurons):
        contexts = []
        for rank in range(k):
            flat_idx = top_pos[rank, neuron_idx].item()
            val = top_vals[rank, neuron_idx].item()
            b, s = divmod(flat_idx, seq)
            start = max(0, s - context_window)
            end = min(seq, s + context_window + 1)
            snippet_tokens = dataset_tokens[b, start:end]
            snippet = model.to_string(snippet_tokens)
            contexts.append((snippet, val))
        result[neuron_idx] = contexts

    return result


# TODO: move the LLM path below into mid/analysis/llm.py once Mike's
# labeling pipeline solidifies so auto_label.py can share it.
def _score_via_llm(snippets: list[str], model: str = "claude-haiku-4-5") -> float:
    """Ask an LLM for a 0-1 monosemanticity rating on these contexts."""

    client = anthropic.Anthropic()
    prompt = (
        "You are rating how monosemantic a neuron or feature is, based on "
        "its top activating contexts from a small GPT trained on "
        "Shakespeare's plays.\n"
        "Monosemantic (score near 1): every context shares one clear "
        "concept, e.g. a single character's name (HAMLET, FALSTAFF), a "
        "punctuation role (end-of-line colons after speaker tags), stage "
        "directions, or a specific syntactic pattern.\n"
        "Polysemantic (score near 0): the contexts span unrelated concepts.\n"
        "Return a single float between 0 and 1, nothing else.\n\n"
        "Contexts:\n" + "\n---\n".join(s[:200] for s in snippets)
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return float(resp.content[0].text.strip())
    except (ValueError, IndexError, AttributeError):
        return 0.0


def score_monosemanticity(
    top_contexts: dict[int, list[tuple[str, float]]],
    llm_client=None,
) -> dict[int, float]:
    """Score each neuron's top contexts on a 0-1 monosemanticity scale.

    ``llm_client`` can be:
    - ``None`` (default): use a unique-token-ratio heuristic.
    - ``"anthropic"``: call the temporary inline Claude scorer.
    - any object with ``.score(snippets) -> float``.
    """
    scores: dict[int, float] = {}

    for neuron_idx, contexts in top_contexts.items():
        snippets = [text for text, _ in contexts]

        if llm_client == "anthropic":
            scores[neuron_idx] = _score_via_llm(snippets)
            continue
        if llm_client is not None:
            scores[neuron_idx] = float(llm_client.score(snippets))
            continue

        all_tokens = " ".join(snippets).split()
        if not all_tokens:
            scores[neuron_idx] = 0.0
            continue
        unique_ratio = len(set(all_tokens)) / len(all_tokens)
        scores[neuron_idx] = 1.0 - unique_ratio

    return scores


def compare_to_sae(
    neuron_scores: dict[int, float],
    feature_scores: dict[int, float],
    threshold: float = 0.5,
) -> dict[str, float | int | list[float]]:
    """Summary stats comparing neuron vs. SAE-feature monosemanticity scores."""
    n_vals = list(neuron_scores.values())
    f_vals = list(feature_scores.values())

    def frac_above(vals: list[float], t: float) -> float:
        return sum(1 for v in vals if v > t) / len(vals) if vals else 0.0

    return {
        "n_neurons": len(n_vals),
        "n_features": len(f_vals),
        "neuron_mean": statistics.mean(n_vals) if n_vals else 0.0,
        "feature_mean": statistics.mean(f_vals) if f_vals else 0.0,
        "neuron_median": statistics.median(n_vals) if n_vals else 0.0,
        "feature_median": statistics.median(f_vals) if f_vals else 0.0,
        "neuron_interpretable_frac": frac_above(n_vals, threshold),
        "feature_interpretable_frac": frac_above(f_vals, threshold),
        "neuron_scores_sorted": sorted(n_vals),
        "feature_scores_sorted": sorted(f_vals),
    }


def summarize_neuron(
    neuron_idx: int,
    top_contexts: dict[int, list[tuple[str, float]]],
    top_n_tokens: int = 5,
) -> dict[str, object]:
    """Return ``{neuron, top_tokens, contexts}`` for spot-checking one neuron."""
    contexts = top_contexts[neuron_idx]
    all_tokens = " ".join(text for text, _ in contexts).split()
    counter = Counter(all_tokens)
    return {
        "neuron": neuron_idx,
        "top_tokens": counter.most_common(top_n_tokens),
        "contexts": contexts,
    }
