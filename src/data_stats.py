"""
Create reusable corpus statistics for the Lab 5 report.

Outputs:
- outputs/corpus_stats.csv
- outputs/upos_distribution.csv
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from load_conllu import read_conllu


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    corpus_rows = []
    upos_rows = []

    for path in sorted(RAW_DIR.rglob("*.conllu")):
        language = path.parent.name
        split = path.stem

        sentences = read_conllu(path)

        sentence_count = len(sentences)
        token_count = sum(len(tokens) for tokens, _ in sentences)
        tag_counter = Counter(tag for _, tags in sentences for tag in tags)

        corpus_rows.append(
            {
                "language": language,
                "split": split,
                "file": str(path),
                "sentences": sentence_count,
                "tokens": token_count,
                "upos_label_count": len(tag_counter),
                "upos_labels": " ".join(sorted(tag_counter.keys())),
            }
        )

        for tag, count in sorted(tag_counter.items()):
            upos_rows.append(
                {
                    "language": language,
                    "split": split,
                    "upos": tag,
                    "count": count,
                    "proportion": count / token_count if token_count else 0.0,
                }
            )

    corpus_df = pd.DataFrame(corpus_rows)
    upos_df = pd.DataFrame(upos_rows)

    corpus_df.to_csv(OUTPUT_DIR / "corpus_stats.csv", index=False, encoding="utf-8")
    upos_df.to_csv(OUTPUT_DIR / "upos_distribution.csv", index=False, encoding="utf-8")

    print("Wrote outputs/corpus_stats.csv")
    print("Wrote outputs/upos_distribution.csv")
    print()
    print(corpus_df[["language", "split", "sentences", "tokens", "upos_label_count"]])


if __name__ == "__main__":
    main()
