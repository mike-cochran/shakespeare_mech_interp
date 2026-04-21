"""Load trained SAEs for downstream analysis.
Thin wrapper around the 'sae_lens.SAE.load_from_disk' so 03a/04 don't need
to know the SAELens import path directly. 
Owner: David Teklea
"""

from __future__ import annotations


def load_sae(path: str):
    """Load a trained SAE from disk. 
    Args: path ('out_dir' to 'train_sae', Should contain the
                files written by `TrainingSAE.save_inference_model` (weights +
                config JSON).), device (torch device where SAE is placed)
    Returns: A 'sae_lens.SAE' ready for '.encode()' / '.decode()' / '.forward()' 
    """

    from sae_lens import SAE
    return SAE.load_from_disk(path, device=device)
    raise NotImplementedError
