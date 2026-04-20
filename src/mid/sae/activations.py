"""Cache HookedTransformer activations to disk for SAE training. Streams the Shakespeare corpus through the trained model and
saves per-layer activations at the configured hook points.

Owner: Cole S.
"""

from __future__ import annotations

import torch

from pathlib import Path
from mid.config import ModelConfig
from mid.model.dataset import load_token_arrays
from mid.model.hooked_model import load_checkpoint


def cache_activations(
    checkpoint_path: str,
    model_cfg: ModelConfig,
    hook_type: str,
    out_path: str,
    batch_size: int,
    use_split: bool,
):
    """
    Caches token-level activations and metadata for SAE training.  Processed data is stored in /data/sae_data/{model_type}_{model_size}_train.pt and /data/sae_data/{model_type}_{model_size}_val.pt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(out_path)
    # make the directory if it doesn't already exist
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_checkpoint(checkpoint_path, model_cfg)
    seq_len = model_cfg.n_ctx
    n_layers = model_cfg.n_layers

    # Load tokens
    train_ids, val_ids = load_token_arrays("../tokenizer_output")

    # Resolve hook names by type
    if hook_type == "mlp":
        hook_names = [f"blocks.{layer}.mlp.hook_post" for layer in range(n_layers)]
    elif hook_type == "attention":
        hook_names = [f"blocks.{layer}.attn.hook_z" for layer in range(n_layers)]
    elif hook_type == "stream":
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    else:
        print("WARNING: Hook type invalid, no data is saved.")
        return None

    # Decide whether to merge splits together or not
    if use_split:
        splits_to_process = [("train", train_ids), ("val", val_ids)]
    else:
        all_ids = (
            torch.cat(
                [
                    torch.as_tensor(train_ids, dtype=torch.long),
                    torch.as_tensor(val_ids, dtype=torch.long),
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )
        splits_to_process = [("all", all_ids)]

    # Process activations
    for split, token_ids in splits_to_process:
        # Prep sequences
        sequences = (len(token_ids) // seq_len) * seq_len
        token_ids = token_ids[:sequences]
        token_blocks = token_ids.reshape(-1, seq_len)

        # Store activations and metadata
        activations_by_layer = {layer: [] for layer in range(n_layers)}
        token_id_chunks = []
        seq_idx_chunks = []
        pos_idx_chunks = []

        # DEBUG: Total number of batches
        total_batches = (len(token_blocks) + batch_size - 1) // batch_size

        # Run the model and get the activations
        with torch.no_grad():
            # Get the batch
            for batch_idx in range(0, len(token_blocks), batch_size):
                batch = token_blocks[batch_idx : batch_idx + batch_size]
                batch_tensor = torch.from_numpy(batch).long().to(device)

                # Get the batch size in case it is different (partial sequence, only really plausible for the last batch)
                batch_size_real, seq_len_real = batch_tensor.shape

                # Get cache
                _, cache = model.run_with_cache(batch_tensor, names_filter=hook_names)

                # Metadata
                flat_ids = batch_tensor.detach().cpu().reshape(-1)
                token_id_chunks.append(flat_ids)

                seq_idxs = torch.arange(batch_idx, batch_idx + batch_size_real, dtype=torch.long)
                seq_idx_chunks.append(seq_idxs.unsqueeze(1).repeat(1, seq_len_real).reshape(-1))

                pos_idx_chunks.append(
                    torch.arange(seq_len_real, dtype=torch.long).repeat(batch_size_real)
                )

                # Get activations by layer
                for layer, hook_name in enumerate(hook_names):
                    act = cache[hook_name].detach().cpu()
                    # flatten depending on the hooks utilized
                    if hook_type in ("mlp", "stream"):
                        # Flatten from [batch, pos, width] to [num_tokens (bath*pos), width]
                        act = act.reshape(-1, act.shape[-1]).contiguous()
                    else:
                        # Flatten from [batch, pos, n_heads, d_heads] to [num_tokens (batch*pos), n_heads*d_heads]
                        act = act.reshape(-1, act.shape[-2] * act.shape[-1]).contiguous()

                    activations_by_layer[layer].append(act)

                # DEBUG: Prints progress every 50 batches
                if (batch_idx // batch_size) % 50 == 0:
                    print(
                        f"Processed batch {(batch_idx // batch_size) + 1} out of {total_batches} for {split}"
                    )

        # Consolidate metadata for split
        final_metadata = {
            "token_ids": torch.cat(token_id_chunks, dim=0),
            "seq_idx": torch.cat(seq_idx_chunks, dim=0),
            "pos_idx": torch.cat(pos_idx_chunks, dim=0),
            "split": split,
            "seq_len": seq_len,
            "hook_type": hook_type,
        }

        # Concatinate activations by layer
        final_activations_by_layer = {
            layer: torch.cat(chunks, dim=0) for layer, chunks in activations_by_layer.items()
        }

        # Save
        payload = {"metadata": final_metadata, "activations_by_layer": final_activations_by_layer}
        save_path = out_root / f"{split}.pt"
        torch.save(payload, save_path)
        print(f"Saved activations by layer to {save_path}")

    return None


def read_activations(path: str, layer_num: int = 0):
    """
    Loads the activations and metadata for a selected layer from the specified file.
    NOTE: You will have to manually move the data to the device for the SAE.  I have omitted this step as I cannot garuntee the device.
    Returns: activations, metadata, number of tokens, number of activations
    """
    payload = torch.load(path, map_location="cpu")
    activations = payload["activations_by_layer"][layer_num]
    metadata = payload["metadata"]

    number_of_tokens, number_of_activations = activations.shape

    return activations, metadata, number_of_tokens, number_of_activations
