"""Dataclass configs for the transformer, SAE, and training loops. Loaded from YAML files in `configs/`.

Owner:
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
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