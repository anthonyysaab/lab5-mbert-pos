"""
Load and inspect Universal Dependencies .conllu files.

This script extracts:
- sentence tokens
- UPOS labels
- corpus statistics
- UPOS label distributions

It ignores empty nodes and keeps real UD word tokens only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from conllu import parse_incr


Sentence = Tuple[List[str], List[str]]


def is_real_token(token_id) -> bool:
    """
    UD token ids can be:
    - int: normal token
    - tuple like (1, "-", 2): multiword token range
    - tuple like (3, ".", 1): empty node

    For POS tagging, we keep only normal integer-token lines.
    """
    return isinstance(token_id, int)


def read_conllu(path: Path) -> List[Sentence]:
    """
    Read one .conllu file and return a list of:
    (tokens, upos_tags)
    """
    sentences: List[Sentence] = []

    with path.open("r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words: List[str] = []
            tags: List[str] = []

            for token in tokenlist:
                if not is_real_token(token["id"]):
                    continue

                form = token["form"]
                upos = token["upos"]

                if form is None or upos is None:
                    continue

                words.append(form)
                tags.append(upos)

            if words:
                sentences.append((words, tags))

    return sentences


def inspect_treebank(path: Path) -> Dict[str, object]:
    """
    Return basic statistics for one .conllu file.
    """
    sentences = read_conllu(path)

    token_count = sum(len(words) for words, _ in sentences)
    tag_counter = Counter(tag for _, tags in sentences for tag in tags)

    return {
        "file": str(path),
        "sentences": len(sentences),
        "tokens": token_count,
        "upos_labels": sorted(tag_counter.keys()),
        "upos_distribution": dict(tag_counter.most_common()),
    }


def inspect_all_conllu(raw_dir: Path) -> List[Dict[str, object]]:
    """
    Inspect all .conllu files under data/raw.
    """
    paths = sorted(raw_dir.rglob("*.conllu"))

    if not paths:
        raise FileNotFoundError(
            f"No .conllu files found under {raw_dir}. "
            "Put UD treebank files in data/raw first."
        )

    return [inspect_treebank(path) for path in paths]


def main() -> None:
    raw_dir = Path("data/raw")
    results = inspect_all_conllu(raw_dir)

    print("\n=== UD treebank inspection ===\n")

    for result in results:
        print(f"File: {result['file']}")
        print(f"Sentences: {result['sentences']}")
        print(f"Tokens: {result['tokens']}")
        print(f"UPOS labels: {', '.join(result['upos_labels'])}")
        print("Top UPOS distribution:")

        for tag, count in list(result["upos_distribution"].items())[:10]:
            print(f"  {tag:6s} {count}")

        print()


if __name__ == "__main__":
    main()
