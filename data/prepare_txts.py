import re
from pathlib import Path

INPUT_DIR = Path("folger/shakespeares-works_TXT_FolgerShakespeare")
OUTPUT_FILE = Path("shakespeare_combined.txt")


def clean_text(text: str) -> str:
    """Cleans up the text for tokenization, focusing on removing unnecessary:
    - metadata at the top of the file
    - character lists at the beginning of the play
    - removing stage directions
    - Act and Scene headers with = signs
    - Sonnet numbers
    - leading and trailing tabs
    - extra carriage returns
    """

    # 1. Cut everything before ACT 1 (header + character list)
    act1_match = re.search(r"^ACT 1\s*$", text, re.MULTILINE)
    if act1_match:
        text = text[act1_match.start() :]
    else:  # Text is a play so cut header from that
        # 2. Cut poem header
        header_match = re.search(r"^.*FDT version.*$", text, re.MULTILINE)
        if header_match:
            text = text[header_match.end() :]

    # 3. Remove stage directions: [Enter Bertram...], [She exits.], etc.
    text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

    # 4. Remove ACT/Scene headers and their decoration lines
    text = re.sub(r"^ACT \d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Scene \d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^=+\s*$", "", text, flags=re.MULTILINE)  # decorative "====" lines

    # 5. Remove sonnet numbers (bare digit lines)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # 6. Strip trailing tabs/spaces from lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # 7. Convert leading tabs (indented verse) to spaces
    text = re.sub(r"^\t+", lambda m: "  " * len(m.group()), text, flags=re.MULTILINE)

    # 8. Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    all_text = []

    for filepath in sorted(INPUT_DIR.glob("*.txt")):
        print(f"Processing: {filepath.name}")
        raw = filepath.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        all_text.append(cleaned)

    combined = "\n\n".join(all_text)
    OUTPUT_FILE.write_text(combined, encoding="utf-8")
    print(f"Wrote {len(combined):,} characters to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
