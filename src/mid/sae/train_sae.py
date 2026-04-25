"""SAELens training entry point. Consumes cached activations from `mid.sae.activations`.

Reads per-layer activations from mid.sae.activations.cache_activations,
trains a SAE model, and saves the trained model to disk.Does this using SAELens v6,
and agains tthe HookedTransformer checkpoint via 'override_model' (otherwise, SAELens
will try to pull model from the hub)

Format contracted from original:
    - One HF Dataset saved via 'Dataset.save_to_disk' (see mid.sae.activations.cache_activations):
    - One row per sequence, with columns as follows:
        - '{hook_name}': Array2d(shape=(contxt_size, d_input), dtype="float32")
        - 'token_ids': Sequence(Value("int32"), length=contxt_size)
    - At load time SAELens asserts 'features[hook_name].shape == (contxt_size, d_input)'


Owner:David Teklea
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from datasets import Array2D, Dataset, Features, Sequence, Value

from mid.config import ModelConfig, SAEConfig
from mid.model.hooked_model import load_checkpoint
from mid.sae.activations import read_activations


def _build_hf_cache(
    activations_pt_path: str,
    sae_cfg: SAEConfig,
    cache_dir: Path,
) -> Dataset:
    """Convert Cole's per-layer .pt payload into the SAELens-comptatible format HF Dataset

    Cole's cache stores activations flattened as `[n_seq * seq_len, d_in]` and
    token_ids flattened as `[n_seq * seq_len]` (contiguous, sequence-major).
    SAELens wants each row to be one full sequence, so we reshape back to
    `[n_seq, seq_len, d_in]` / `[n_seq, seq_len]` before writing.
    """

    hook_name = sae_cfg.hook_name()
    contxt_size = sae_cfg.context_len
    d_input = sae_cfg.dim_input
    # read the .pt file and extract activations and token_ids
    activations, metadata, n_tokens, n_activations = read_activations(
        activations_pt_path, layer_num=sae_cfg.layer
    )
    if n_activations != d_input:
        raise ValueError(
            f"Cached activation width ({n_activations}) != SAEConfig.d_in ({d_input}). "
            f"Check that hook_type={sae_cfg.hook_type!r} matches how 02a was run."
        )
    if metadata["seq_len"] != contxt_size:
        raise ValueError(
            f"Cached seq_len ({metadata['seq_len']}) != SAEConfig.contxt_size ({contxt_size})."
        )
    if metadata.get("hook_type") and metadata["hook_type"] != sae_cfg.hook_type:
        raise ValueError(
            f"Cache was built with hook_type={metadata['hook_type']!r} but SAEConfig "
            f"requests {sae_cfg.hook_type!r}. Re-run 02a or change sae_cfg.hook_type."
        )
    if n_tokens % contxt_size != 0:
        raise ValueError(
            f"Flat token count {n_tokens} is not divisible by contxt_size {contxt_size}."
        )
    n_seq = n_tokens // contxt_size
    activations_3d = activations.reshape(n_seq, contxt_size, d_input).to(torch.float32).contiguous()
    token_idexes_2d = metadata["token_ids"].reshape(n_seq, contxt_size).to(torch.int32).contiguous()

    features = Features(
        {
            hook_name: Array2D(shape=(contxt_size, d_input), dtype="float32"),
            "token_ids": Sequence(Value(dtype="int32"), length=contxt_size),
        }
    )

    ds = Dataset.from_dict(
        {hook_name: activations_3d.cpu().numpy(), "token_ids": token_idexes_2d.cpu().numpy()},
        features=features,
    )

    # save dataset to disk, otherwise, nuke and pave on re runs
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    ds.save_to_disk(str(cache_dir))
    print(
        f"Repacked cache from {n_seq} seqs x {contxt_size} tokens/seq x {d_input} d_in to HF Dataset at {cache_dir}."
        f"at hook {hook_name!r} -> {cache_dir}"
    )

    ## have to build a separate tokens-only dataset for the init check
    ## conducted by SAELens before training
    override_features = Features({"tokens": Sequence(Value(dtype="int64"), length=contxt_size)})
    override_ds = Dataset.from_dict(
        {"tokens": metadata["token_ids"].reshape(n_seq, contxt_size).to(torch.int64).numpy()},
        features=override_features,
    )
    return override_ds


def train_sae(
    sae_cfg: SAEConfig,
    model_cfg: ModelConfig,
    checkpoint_path: str,
    activations_pt_path: str,
    out_dir: str,
    device: str | None = None,
):
    """Train a single SAE against one layer of the cached activations"""

    from sae_lens import (
        LanguageModelSAERunnerConfig,
        LanguageModelSAETrainingRunner,
        LoggingConfig,
        StandardTrainingSAEConfig,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training SAE on device: {device}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir_path / "saelens_cache"

    # Step 1 - repack Cole's .pt as HF dataset SAELens can load
    hf_dataset = _build_hf_cache(activations_pt_path, sae_cfg, cache_dir)
    # Step 2 rebuild transformer and hand it to the runner via 'override_model' so it doesn't try to pull from the hub
    model = load_checkpoint(checkpoint_path, model_cfg)
    # Step 3 -build v6 nested config and run
    # Step 3 - build the v6 nested config and run.
    runner_cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(
            d_in=sae_cfg.dim_input,
            d_sae=sae_cfg.dim_sae,
            l1_coefficient=sae_cfg.l1_coeff,
            l1_warm_up_steps=sae_cfg.l1_warmup_steps,
            apply_b_dec_to_input=sae_cfg.apply_bias_decay_to_input,
            normalize_activations=sae_cfg.normalize_activations,
        ),
        # model_name is just an identifier in metadata; the actual weights come from override_model.
        model_name="shakespeare-decoder-small",
        model_class_name="HookedTransformer",
        hook_name=sae_cfg.hook_name(),
        dataset_path="shakespeare-cached",  # descriptive tag; not read when using cache
        is_dataset_tokenized=True,
        streaming=False,
        context_size=sae_cfg.context_len,
        use_cached_activations=True,
        cached_activations_path=str(cache_dir),
        prepend_bos=False,  # Cole's 02a does not prepend BOS
        train_batch_size_tokens=sae_cfg.batch_size,
        training_tokens=sae_cfg.training_tokens,
        lr=sae_cfg.lr,
        lr_warm_up_steps=sae_cfg.lr_warmup_steps,
        n_batches_in_buffer=sae_cfg.num_batches_in_buffer,
        store_batch_size_prompts=sae_cfg.store_batch_size_prompts,
        device=device,
        seed=sae_cfg.seed,
        logger=LoggingConfig(log_to_wandb=False),
        output_path=str(out_dir_path),
        n_checkpoints=0,
        save_final_checkpoint=False,
    )

    runner = LanguageModelSAETrainingRunner(
        cfg=runner_cfg,
        override_model=model,
        override_dataset=hf_dataset,
    )
    sae = runner.run()
    print(f"SAE training complete. Saved to {out_dir_path}")
    return sae
