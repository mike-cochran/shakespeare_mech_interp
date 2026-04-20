import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Config BPE params
CORPUS_FILE = Path("../data/shakespeare_combined.txt")
VOCAB_SIZE = 3000
MIN_FREQUENCY = 3  # token must appear at least this many times
TRAIN_TEST_SPLIT = 0.9
OUTPUT_DIR = Path("../tokenizer_output")

OUTPUT_DIR.mkdir(exist_ok=True)


# 1. Instantiate BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    show_progress=True,
)

print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE}...")
tokenizer.train([str(CORPUS_FILE)], trainer)
tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))


# 2. Encode the full text
text = CORPUS_FILE.read_text(encoding="utf-8")
encoded = tokenizer.encode(text)
token_ids = encoded.ids

print(f"Input text length: {len(text):,} chars → {len(token_ids):,} tokens")


# 3. Save out np arrays
ids = np.array(token_ids, dtype=np.uint16)  # uint16 supports up to 65k vocab

# Split tokenized data
split_idx = int(len(ids) * TRAIN_TEST_SPLIT)
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]

np.save(OUTPUT_DIR / "train.npy", train_ids)
np.save(OUTPUT_DIR / "val.npy", val_ids)


# 4. Save vocab lookup for interpretability
vocab = tokenizer.get_vocab()

# Invert: id → token string
id_to_token = {v: k for k, v in vocab.items()}

with open(OUTPUT_DIR / "vocab.json", "w") as f:
    json.dump(id_to_token, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/:")
print("  tokenizer.json  – full tokenizer (reload with Tokenizer.from_file)")
print("  train.npy       – training token ids")
print("  val.npy         – validation token ids")
print("  vocab.json      – id→token mapping for interpretability")


# 5. Check tokenization results
sample = token_ids[:50]
decoded = tokenizer.decode(sample)
N = 100
print(f"\n============ Sample Top {N} Tokens ============")
counts = Counter(train_ids.tolist())
vocab = tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

print(f"{'Rank':<6}{'ID':<8}{'Token':<30}{'Count':<10}{'% of corpus'}")
print("─" * 65)
for rank, (tok_id, count) in enumerate(counts.most_common(N), 1):
    token_str = id_to_token.get(tok_id, "???")
    display = token_str.replace("Ġ", "·").replace("Ċ", "↵")
    pct = count / len(train_ids) * 100
    print(f"{rank:<6}{tok_id:<8}{display:<30}{count:<10}{pct:.2f}%")
