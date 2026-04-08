import json
import numpy as np
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer

OUTPUT_DIR = Path("tokenizer_output")
tokenizer = Tokenizer.from_file(str(OUTPUT_DIR / "tokenizer.json"))
train_ids = np.load(OUTPUT_DIR / "train.npy")

# Set up tokenizer colors
COLORS = [
    "\033[43m",   # yellow bg
    "\033[46m",   # cyan bg
    "\033[42m",   # green bg
    "\033[45m",   # magenta bg
    "\033[44m",   # blue bg
    "\033[41m",   # red bg
    "\033[47m",   # white bg
]
RESET = "\033[0m"

def show_tokenized(text: str, max_chars: int = 1000):
    """Print text with alternating background colors per token."""
    snippet = text[:max_chars]
    encoded = tokenizer.encode(snippet)

    print(f"\n{'═' * 65}")
    print("TOKENIZED VIEW (each color = one token)")
    print(f"{'═' * 65}\n")

    output = []
    for i, (tok_id, (start, end)) in enumerate(zip(encoded.ids, encoded.offsets)):
        color = COLORS[i % len(COLORS)]
        token_text = snippet[start:end]
        output.append(f"{color}{token_text}{RESET}")

    print("".join(output))
    print(f"\n\n({len(encoded.ids)} tokens shown)")


# Load corpus and show a sample
corpus = Path("data/shakespeare_combined.txt").read_text(encoding="utf-8")
show_tokenized(corpus, max_chars=1000)

# Show a different passage example
print("\n\n── Another passage ──")
show_tokenized(corpus[500000:], max_chars=1000)