"""Auto-label SAE features by sending top-activating contexts to an LLM.

Pipeline:
    1. `top_activating_contexts` streams cached activations through the SAE,
       keeps a running top-k per feature, and reconstructs a decoded text
       window around each peak token.
    2. `label_features` sends those windows to Claude and parses back a short label,
       description, and confidence score per feature.

Owner: Mike Cochran
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from mid.analysis.llm import call_anthropic

PEAK_OPEN = "\u00ab"  # «
PEAK_CLOSE = "\u00bb"  # »


def top_activating_contexts(
    sae,
    activations: torch.Tensor,
    metadata: dict,
    tokenizer,
    k: int = 20,
    window: int = 16,
    batch_size: int = 8192,
    device: str | None = None,
) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Find top-k activating contexts per SAE feature.

    Args:
        sae: the trained SAE
        activations: tensor of activations
        metadata: Dict with `token_ids`, `seq_idx`, `pos_idx`, `seq_len`
        tokenizer: Our BPE tokenizer
        k: Number of top contexts to keep per feature; 20 seems like a good balance
        window: # of tokens on each side of the peak to include
        batch_size: Tokens per SAE encode pass
        device: defaults to CUDA if available

    Returns:
        Tuple `(top_contexts, stats)`:
            top_contexts: `{feat_id: [{"activation", "seq_idx", "pos_idx",
                "peak_text", "context_text"}, ...]}`, entries sorted by
                activation descending.
            stats: `{feat_id: {"density", "max_act", "mean_act_nonzero"}}`.
                `density` is the fraction of tokens where the feature fires.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sae = sae.to(device).eval()
    activations = activations.to(torch.float32)
    n_tokens = activations.shape[0]
    d_sae = int(sae.cfg.d_sae)

    feat_idx = torch.arange(d_sae)
    feat_idx_dev = feat_idx.to(device)
    n_feats = feat_idx.numel()
    k = min(k, n_tokens)

    top_vals = torch.full((n_feats, k), float("-inf"))
    top_positions = torch.full((n_feats, k), -1, dtype=torch.long)

    nonzero_counts = torch.zeros(n_feats, dtype=torch.long)
    act_sums = torch.zeros(n_feats, dtype=torch.float64)
    max_acts = torch.full((n_feats,), float("-inf"))

    with torch.no_grad():
        for start in range(0, n_tokens, batch_size):
            stop = min(start + batch_size, n_tokens)
            x = activations[start:stop].to(device)
            feats = sae.encode(x).index_select(1, feat_idx_dev).cpu()

            nonzero_mask = feats > 0
            nonzero_counts += nonzero_mask.sum(dim=0)
            act_sums += feats.clamp_min(0).to(torch.float64).sum(dim=0)
            batch_max, _ = feats.max(dim=0)
            max_acts = torch.maximum(max_acts, batch_max)

            batch_positions = torch.arange(start, stop).unsqueeze(0).expand(n_feats, -1)
            cand_vals = torch.cat([top_vals, feats.T], dim=1)
            cand_pos = torch.cat([top_positions, batch_positions], dim=1)
            new_vals, new_idx = cand_vals.topk(k, dim=1)
            top_vals = new_vals
            top_positions = torch.gather(cand_pos, 1, new_idx)

    seq_ids = metadata["seq_idx"]
    pos_ids = metadata["pos_idx"]
    token_ids = metadata["token_ids"]
    seq_len = int(metadata["seq_len"])

    top_contexts: dict[int, list[dict]] = {}
    for row, feat_id in enumerate(feat_idx.tolist()):
        entries: list[dict] = []
        for rank in range(k):
            val = top_vals[row, rank].item()
            if val == float("-inf") or val <= 0.0:
                continue
            global_pos = int(top_positions[row, rank].item())
            seq = int(seq_ids[global_pos].item())
            pos_in_seq = int(pos_ids[global_pos].item())
            seq_start = seq * seq_len

            left = max(0, pos_in_seq - window)
            right = min(seq_len, pos_in_seq + window + 1)
            ctx_ids = token_ids[seq_start + left : seq_start + right].tolist()
            peak_offset = pos_in_seq - left

            left_text = tokenizer.decode(ctx_ids[:peak_offset]) if peak_offset > 0 else ""
            peak_text = tokenizer.decode([ctx_ids[peak_offset]])
            right_text = (
                tokenizer.decode(ctx_ids[peak_offset + 1 :])
                if peak_offset + 1 < len(ctx_ids)
                else ""
            )

            entries.append(
                {
                    "activation": float(val),
                    "seq_idx": seq,
                    "pos_idx": pos_in_seq,
                    "peak_text": peak_text,
                    "context_text": f"{left_text}{PEAK_OPEN}{peak_text}{PEAK_CLOSE}{right_text}",
                }
            )
        top_contexts[int(feat_id)] = entries

    density = nonzero_counts.to(torch.float64) / max(n_tokens, 1)
    denom = nonzero_counts.to(torch.float64).clamp_min(1)
    mean_nonzero = torch.where(nonzero_counts > 0, act_sums / denom, torch.zeros_like(act_sums))
    stats = {
        int(feat_id): {
            "density": float(density[row]),
            "max_act": float(max_acts[row].item()) if max_acts[row].isfinite() else 0.0,
            "mean_act_nonzero": float(mean_nonzero[row]),
        }
        for row, feat_id in enumerate(feat_idx.tolist())
    }

    return top_contexts, stats


_SYSTEM_PREAMBLE = (
    "You are analyzing a feature from a sparse autoencoder trained on a small "
    "Transformer that models the complete works of William Shakespeare. Below "
    "are text windows where this feature activated most strongly. The token "
    f"the feature fired on is wrapped in {PEAK_OPEN}guillemets{PEAK_CLOSE}. Identify the "
    "pattern the feature detects (syntactic, lexical, thematic, positional, "
    "etc.). Reply with exactly one JSON object on a single line and nothing "
    "else:\n"
    '{"label": "<=6 words", "description": "one sentence", "confidence": 0.0-1.0}'
)


def _build_prompt(entries: list[dict]) -> str:
    lines = []
    for i, e in enumerate(entries, 1):
        text = e["context_text"].replace("\n", " ").strip()
        lines.append(f"[{i}] act={e['activation']:.2f}  {text}")
    return _SYSTEM_PREAMBLE + "\n\nFeature contexts:\n" + "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if match is None:
        return {"label": text.strip()[:80], "description": "", "confidence": 0.0}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": text.strip()[:80], "description": "", "confidence": 0.0}
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": str(obj.get("label", "")).strip(),
        "description": str(obj.get("description", "")).strip(),
        "confidence": confidence,
    }


def label_features(
    top_contexts: dict[int, list[dict]],
    model: str | None = None,
    max_tokens: int = 200,
    stats: dict[int, dict] | None = None,
    min_density: float = 1e-5,
    max_density: float = 0.5,
    max_contexts: int = 12,
) -> dict[int, dict]:
    """Send top-activating contexts to an LLM and return parsed labels.

    Args:
        top_contexts: First element of `top_activating_contexts`.
        model: Model name. When None, the client's default is used.
        max_tokens: Cap on response tokens.
        stats: Second element of `top_activating_contexts`. When supplied,
            features with density outside `[min_density, max_density]` are
            skipped and tagged `dead` / `ubiquitous` with confidence 0.
        min_density, max_density: Density gates (see above).
        max_contexts: Truncate the context block sent to the LLM to keep the
            prompt compact.

    Returns:
        `{feat_id: {"label", "description", "confidence", "n_contexts"}}`.
    """

    out: dict[int, dict] = {}
    for feat_id, entries in top_contexts.items():
        if stats is not None:
            density = stats.get(feat_id, {}).get("density", 0.0)
            if density < min_density:
                out[feat_id] = {
                    "label": "dead",
                    "description": "feature rarely or never activates",
                    "confidence": 0.0,
                    "n_contexts": 0,
                }
                continue
            if density > max_density:
                out[feat_id] = {
                    "label": "ubiquitous",
                    "description": "feature fires on most tokens",
                    "confidence": 0.0,
                    "n_contexts": len(entries),
                }
                continue

        if not entries:
            out[feat_id] = {
                "label": "dead",
                "description": "no positive activations",
                "confidence": 0.0,
                "n_contexts": 0,
            }
            continue

        trimmed = entries[:max_contexts]
        prompt = _build_prompt(trimmed)
        kwargs: dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens}
        if model is not None:
            kwargs["model"] = model
        try:
            raw = call_anthropic(**kwargs)
        except Exception as exc:  # surface API erro
            out[feat_id] = {
                "label": "error",
                "description": f"{type(exc).__name__}: {exc}",
                "confidence": 0.0,
                "n_contexts": len(trimmed),
            }
            continue

        parsed = _parse_response(raw)
        parsed["n_contexts"] = len(trimmed)
        out[feat_id] = parsed

    return out
