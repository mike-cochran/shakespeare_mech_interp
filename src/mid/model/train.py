"""nanoGPT-style training loop for the HookedTransformer. Logs loss, perplexity, attention pattern snapshots.

Owner:
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from mid.config import ModelConfig, TrainConfig
from mid.model.dataset import load_token_arrays, make_batches
from mid.model.hooked_model import build_model


# Define function to calculate cross entropy loss on predicted tokens
def estimate_loss(model, train_batches, val_batches, eval_batches: int, device: str):
    """Estimate train and val loss over several random batches using cross entropy loss."""
    model.eval()
    results = {}
    for name, batches in [("train", train_batches), ("val", val_batches)]:
        losses = []
        for _ in range(eval_batches):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        results[name] = np.mean(losses)
    model.train()
    return results


def train(
    model_cfg: ModelConfig,
    model_type,
    model_size,
    train_cfg: TrainConfig,
    tokenizer_dir: str,
    out_dir: str,
):
    """A simple training method to start"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} <-- Make sure 'cuda' if GPU is available")

    # Load data
    train_data, val_data = load_token_arrays(tokenizer_dir)
    seq_len = model_cfg.n_ctx
    train_batches = make_batches(train_data, train_cfg.batch_size, seq_len)
    val_batches = make_batches(val_data, train_cfg.batch_size, seq_len, seed=1)

    # Build model
    model = build_model(model_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    # Training loop
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.train()
    best_val_loss = float("inf")

    for step in range(1, train_cfg.total_steps + 1):
        x, y = next(train_batches)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate at set interval and save out the model with the lowest loss
        if step % train_cfg.eval_interval == 0:
            losses = estimate_loss(
                model, train_batches, val_batches, train_cfg.eval_batches, device
            )
            marker = ""
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                marker = " * New best model"
                torch.save(
                    model.state_dict(), out_path / f"{model_type}_{model_size}_best_model.pt"
                )
            print(f"Step {step:>6} | train {losses['train']:.3f} | val {losses['val']:.3f}{marker}")

    # Final save
    losses = estimate_loss(model, train_batches, val_batches, train_cfg.eval_batches, device)
    if losses["val"] < best_val_loss:
        torch.save(model.state_dict(), out_path / "best_model.pt")
    with open(out_path / "hooked_config.json", "w") as f:
        json.dump(model_cfg.to_dict(), f, indent=2)

    return model


def generate_sample(
    model, tokenizer_path: str, device: str, max_new_tokens=200, prompt="HAMLET:\nTo be, or not"
):
    """Generate a short sample to sanity-check the model."""
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path)))
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = input_ids[:, -model.cfg.n_ctx :]
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

    output_ids = input_ids[0].tolist()
    print(tokenizer.decode(output_ids))
