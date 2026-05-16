"""
Load and inspect Universal Dependencies .conllu files.

This script extracts:
- sentence tokens
- UPOS labels
- corpus statistics
- UPOS label distributions

Important Lab 5 decision:
Universal Dependencies sometimes represents a surface token as a multiword token.
Example:

    1-2    au      _
    1      à       ADP
    2      le      DET

For this lab, the surface token is kept as one token and its label is the
concatenation of the component UPOS tags:

    au  ->  ADP+DET

This follows the tokenization-alignment rule required in the assignment.
Empty nodes are ignored.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from conllu import parse_incr


Sentence = Tuple[List[str], List[str]]


def is_real_token_id(token_id) -> bool:
    """
    Return True for normal UD word-token ids.

    In CoNLL-U / conllu:
    - int: normal token
    - tuple like (1, "-", 2): multiword-token range
    - tuple like (3, ".", 1): empty node
    """
    return isinstance(token_id, int)


def is_multiword_token_id(token_id) -> bool:
    """
    Return True for UD multiword-token range ids, e.g. (1, "-", 2).
    """
    return (
        isinstance(token_id, tuple)
        and len(token_id) == 3
        and token_id[1] == "-"
        and isinstance(token_id[0], int)
        and isinstance(token_id[2], int)
    )


def clean_surface_token(form: str) -> str:
    """
    UD guidelines allow token forms containing spaces in rare cases.
    The assignment asks us to remove such spaces.
    """
    return "".join(str(form).split())


def read_conllu(path: Path) -> List[Sentence]:
    """
    Read one .conllu file and return a list of:

        (tokens, upos_tags)

    Multiword tokens are kept as surface tokens and labelled with the
    concatenation of their component UPOS tags.
    """
    sentences: List[Sentence] = []

    with path.open("r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words: List[str] = []
            tags: List[str] = []

            normal_tokens = {
                token["id"]: token
                for token in tokenlist
                if is_real_token_id(token["id"])
            }

            consumed_token_ids: set[int] = set()

            for token in tokenlist:
                token_id = token["id"]

                if is_multiword_token_id(token_id):
                    start_id, _, end_id = token_id
                    form = token["form"]

                    component_tags: List[str] = []
                    component_ids = list(range(start_id, end_id + 1))

                    for component_id in component_ids:
                        component = normal_tokens.get(component_id)
                        if component is None:
                            continue

                        upos = component.get("upos")
                        if upos is None:
                            continue

                        component_tags.append(upos)

                    if form is not None and component_tags:
                        words.append(clean_surface_token(form))
                        tags.append("+".join(component_tags))
                        consumed_token_ids.update(component_ids)

                    continue

                if is_real_token_id(token_id):
                    if token_id in consumed_token_ids:
                        continue

                    form = token["form"]
                    upos = token["upos"]

                    if form is None or upos is None:
                        continue

                    words.append(clean_surface_token(form))
                    tags.append(upos)

                    continue

                # Empty nodes and other non-standard ids are ignored.

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
            print(f"  {tag:12s} {count}")

        print()


if __name__ == "__main__":
    main()