from pathlib import Path
import json
import sys

import numpy as np


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_file_exists(path: Path) -> bool:
    if not path.exists():
        print(f"[MISSING] {path}")
        return False

    if path.stat().st_size == 0:
        print(f"[EMPTY]   {path}")
        return False

    print(f"[OK]      {path}")
    print(f"          size: {path.stat().st_size:,} bytes")
    return True


def check_text_file(path: Path) -> bool:
    print_header("Checking combined Shakespeare text")

    if not check_file_exists(path):
        return False

    try:
        text = path.read_text(encoding="utf-8")
        preview = text[:200].replace("\n", "\\n")
        print(f"          characters: {len(text):,}")
        print(f"          preview: {preview}")
        return True
    # Gotta exit gracefully if file is huge or cannot be read
    except Exception as e:
        print(f"[ERROR]   Failed to read text file: {e}")
        return False

# Check JSON files (tokenizer and vocab)
def check_json_file(path: Path) -> bool:
    print_header(f"Checking JSON file: {path.name}")

    if not check_file_exists(path):
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"          json type: {type(data).__name__}")

        if isinstance(data, dict):
            print(f"          entries: {len(data):,}")
            sample_keys = list(data.keys())[:5]
            print(f"          sample keys: {sample_keys}")
        # Might be a list of tokens or something else 
        elif isinstance(data, list):
            print(f"          entries: {len(data):,}")
        else:
            print("          loaded successfully")

        return True
    except Exception as e:
        print(f"[ERROR]   Failed to load JSON: {e}")
        return False

# Check NumPy .npy files (train and val token ids)
def check_npy_file(path: Path) -> bool:
    print_header(f"Checking NumPy file: {path.name}")

    if not check_file_exists(path):
        return False

    try:
        arr = np.load(path)
        print(f"          shape: {arr.shape}")
        print(f"          dtype: {arr.dtype}")

        if arr.ndim == 0:
            print("          scalar array")
        else:
            print(f"          total elements: {arr.size:,}")
            preview = arr[:10] if arr.size >= 10 else arr
            print(f"          first tokens: {preview}")

        return True
    except Exception as e:
        print(f"[ERROR]   Failed to load NumPy file: {e}")
        return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    combined_text_path = root / "data" / "shakespeare_combined.txt"
    tokenizer_json_path = root / "tokenizer_output" / "tokenizer.json"
    vocab_json_path = root / "tokenizer_output" / "vocab.json"
    train_npy_path = root / "tokenizer_output" / "train.npy"
    val_npy_path = root / "tokenizer_output" / "val.npy"

    results = []

    results.append(check_text_file(combined_text_path))
    results.append(check_json_file(tokenizer_json_path))
    results.append(check_json_file(vocab_json_path))
    results.append(check_npy_file(train_npy_path))
    results.append(check_npy_file(val_npy_path))

    passed = sum(results)
    failed = len(results) - passed

    print_header("Final Summary")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed == 0:
        print("All artifact checks passed.")
        return 0

    print("Some artifact checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())