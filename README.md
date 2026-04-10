# Shakespeare Mechanistic Interpretability

This repository contains a mechnistic interpretability study of a GPT-style model trained on the complete works of William Shakespeare as part of a group project in Georgia Tech's CS 7643 Deep Learning course.

## Team Members

- Areeb Amjad
- Cole Sheridan
- David Teklea
- Michael Cochran

## Project Structure

- `data/`: Folder containing raw and processed text data.
- `tokenizer_output/`: Trained BPE tokenizer and encoded datasets.
- `bpe_tokenize.py`: Script to train the BPE tokenizer.
- `inspect_tokenizer.py`: Utility to visualize how text is broken into tokens.

## Getting Started

### 1. Prerequisites

Create virtual environment: py -m venv .venv

Activate virtual environment: .venv\Scripts\activate

Set up requirements with: pip install -r requirements.txt

### 2. Data Preparation

If the combined text is not already present, you can download and process the raw Folger Shakespeare text with the following scripts:

```bash
# 1. Download the raw zip from Folger
python data/download_txt.py

# 2. Clean and combine texts into a single file
python data/prepare_txts.py
```

This will generate `data/shakespeare_combined.txt`.

### 3. BPE Tokenization

The project uses Byte-Pair Encoding (BPE) via the Hugging Face `tokenizers` library.

To train the tokenizer and encode the data for training:

```bash
python bpe_tokenize.py
```

**Outputs in `tokenizer_output/`:**
- `tokenizer.json`: The trained tokenizer model.
- `vocab.json`: A human-readable mapping of token IDs to strings (can use for interpretability).
- `train.npy` & `val.npy`: Tokenized data split into training and validation sets.

### 4. Inspecting Tokenization (Optional)

To see examples of how the tokenizer handles specific passages (with color-coded tokens):

```bash
python inspect_tokenizer.py
```

