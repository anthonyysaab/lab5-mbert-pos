"""
Analyze the mismatch between UD word tokenization and mBERT WordPiece tokenization.

Outputs:
- outputs/tokenization_stats.csv
- outputs/tokenization_examples.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoTokenizer

from config import MODEL_CHECKPOINT
from load_conllu import read_conllu


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")


def subtokens_for_words(tokenizer, words: List[str]) -> List[List[str]]:
    """
    Return the mBERT subtokens corresponding to each UD word.
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    word_ids = encoding.word_ids()

    grouped: List[List[str]] = [[] for _ in words]

    for token, word_id in zip(tokens, word_ids):
        if word_id is None:
            continue
        grouped[word_id].append(token)

    return grouped


def analyze_file(tokenizer, path: Path) -> dict:
    sentences = read_conllu(path)

    word_count = 0
    subtoken_count = 0
    split_word_count = 0
    max_subtokens_for_word = 0

    for words, _ in sentences:
        grouped = subtokens_for_words(tokenizer, words)

        for pieces in grouped:
            word_count += 1
            subtoken_count += len(pieces)

            if len(pieces) > 1:
                split_word_count += 1

            max_subtokens_for_word = max(max_subtokens_for_word, len(pieces))

    return {
        "language": path.parent.name,
        "split": path.stem,
        "sentences": len(sentences),
        "words": word_count,
        "subtokens": subtoken_count,
        "subtokens_per_word": subtoken_count / word_count if word_count else 0.0,
        "split_words": split_word_count,
        "split_word_rate": split_word_count / word_count if word_count else 0.0,
        "max_subtokens_for_word": max_subtokens_for_word,
    }


def write_examples(tokenizer, output_path: Path, max_sentences_per_language: int = 3) -> None:
    lines: List[str] = []

    for train_path in sorted(RAW_DIR.glob("*/train.conllu")):
        language = train_path.parent.name
        sentences = read_conllu(train_path)

        lines.append("=" * 80)
        lines.append(f"LANGUAGE: {language}")
        lines.append("=" * 80)

        for sentence_index, (words, tags) in enumerate(sentences[:max_sentences_per_language], start=1):
            grouped = subtokens_for_words(tokenizer, words)

            lines.append(f"\nSentence {sentence_index}")
            lines.append("-" * 80)

            for word, tag, pieces in zip(words, tags, grouped):
                lines.append(f"{word:25s} {tag:8s} -> {' '.join(pieces)}")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    rows = []
    for path in sorted(RAW_DIR.rglob("*.conllu")):
        rows.append(analyze_file(tokenizer, path))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tokenization_stats.csv", index=False, encoding="utf-8")

    write_examples(tokenizer, OUTPUT_DIR / "tokenization_examples.txt")

    print("Wrote outputs/tokenization_stats.csv")
    print("Wrote outputs/tokenization_examples.txt")
    print()
    print(
        df[
            [
                "language",
                "split",
                "words",
                "subtokens",
                "subtokens_per_word",
                "split_word_rate",
                "max_subtokens_for_word",
            ]
        ]
    )


if __name__ == "__main__":
    main()
