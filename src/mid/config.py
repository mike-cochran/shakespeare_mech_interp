"""Dataclass configs for the transformer, SAE, and training loops. Loaded from YAML files in `configs/`.

Owner: Mike C.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class ModelConfig:
    d_model: int
    d_head: int
    n_heads: int
    n_layers: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: str
    tokenizer_name: str | None
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    total_steps: int
    eval_interval: int
    eval_batches: int

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass 
class SAEConfig:
    """ Config for the hyperparameters of the SAE via SAELens v6"""

    # Training Schedule 
    lr: float 
    batch_size: int
    training_tokens: int 
    context_len: int
    # Sparsity / reconstruction
    l1_coeff: float 
    l1_warmup_steps: int
    lr_warmup_steps: int
    apply_bias_decay_to_input: bool
    normalize_activations: str 
    # Architecture
    dim_input: int
    dim_sae: int
    hook_type: str  
    layer: int  
    # Buffer sizing for SAE 
    num_batches_in_buffer: int 
    store_batch_size_prompts: int 

    seed: int = 32

    def hook_name(self) -> str:
        """Resolve TransformerLens hook name for the SAE layer."""
        if self.hook_type == "mlp":
            return f"blocks.{self.layer}.mlp.hook_post"
        if self.hook_type == "attention":
            return f"blocks.{self.layer}.attn.hook_z"
        if self.hook_type == "stream": 
            return f"blocks.{self.layer}.hook_resid_post"
        raise ValueError(f"Invalid hook type: {self.hook_type}")
    def to_dict(self) -> dict:
        return asdict(self)

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs(path: str) -> tuple[ModelConfig, TrainConfig]:
    """Load a single YAML and split it into ModelConfig and TrainConfig."""
    raw = load_yaml(path)
    model_keys = {f.name for f in fields(ModelConfig)}
    train_keys = {f.name for f in fields(TrainConfig)}
    return (
        ModelConfig(**{k: v for k, v in raw.items() if k in model_keys}),
        TrainConfig(**{k: v for k, v in raw.items() if k in train_keys}),
    )

def load_sae_config(path: str) -> SAEConfig:
    """Load an SAE YAML into an SAEConfig."""
    raw = load_yaml(path)
    sae_keys = {f.name for f in fields(SAEConfig)}
    return SAEConfig(**{k: v for k, v in raw.items() if k in sae_keys})
