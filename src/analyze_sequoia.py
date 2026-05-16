"""
Analyze the French Sequoia UD test set for Lab 5 warm-up questions.

This script answers the Sequoia-specific parts of the assignment:

1. What is the UPOS label distribution in the Sequoia test set?
2. What multiword tokens occur, and how are they annotated?
3. Are there token forms containing spaces?
4. How does a French sentence differ under UD tokenization and mBERT tokenization?

Input:
- data/sequoia/fr_sequoia-ud-test.conllu

Outputs:
- outputs/sequoia_standard_upos_distribution.csv
- outputs/sequoia_lab_label_distribution.csv
- outputs/sequoia_multiword_tokens.csv
- outputs/sequoia_space_tokens.csv
- outputs/sequoia_tokenization_example.txt
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from conllu import parse_incr
from transformers import AutoTokenizer

from config import MODEL_CHECKPOINT
from load_conllu import read_conllu


SEQUOIA_PATH = Path("data/sequoia/fr_sequoia-ud-test.conllu")
OUTPUT_DIR = Path("outputs")


def is_real_token_id(token_id) -> bool:
    """
    Return True for normal UD word-token ids.
    """
    return isinstance(token_id, int)


def is_multiword_token_id(token_id) -> bool:
    """
    Return True for UD multiword-token range ids, e.g. (13, "-", 14).
    """
    return (
        isinstance(token_id, tuple)
        and len(token_id) == 3
        and token_id[1] == "-"
        and isinstance(token_id[0], int)
        and isinstance(token_id[2], int)
    )


def load_tokenlists(path: Path):
    """
    Load the Sequoia .conllu file as conllu TokenList objects.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Sequoia file: {path}\n"
            "Download it first into data/sequoia/fr_sequoia-ud-test.conllu"
        )

    with path.open("r", encoding="utf-8") as f:
        return list(parse_incr(f))


def standard_upos_distribution(tokenlists) -> pd.DataFrame:
    """
    Standard UD distribution:
    count only normal integer-indexed syntactic tokens.
    Multiword range lines are not counted here.
    """
    counter: Counter[str] = Counter()

    for tokenlist in tokenlists:
        for token in tokenlist:
            if not is_real_token_id(token["id"]):
                continue

            upos = token.get("upos")
            if upos:
                counter[upos] += 1

    total = sum(counter.values())

    rows = [
        {
            "upos": tag,
            "count": count,
            "percentage": round(100 * count / total, 4) if total else 0.0,
        }
        for tag, count in counter.most_common()
    ]

    return pd.DataFrame(rows)


def lab_label_distribution(path: Path) -> pd.DataFrame:
    """
    Lab-normalized distribution using our project loader:
    multiword tokens are kept as surface forms and labelled ADP+DET, etc.
    """
    sentences = read_conllu(path)
    counter: Counter[str] = Counter(tag for _, tags in sentences for tag in tags)
    total = sum(counter.values())

    rows = [
        {
            "label": tag,
            "count": count,
            "percentage": round(100 * count / total, 4) if total else 0.0,
        }
        for tag, count in counter.most_common()
    ]

    return pd.DataFrame(rows)


def multiword_tokens(tokenlists) -> pd.DataFrame:
    """
    Extract Sequoia multiword-token range lines and their component UPOS labels.
    """
    rows: List[Dict[str, object]] = []

    for sentence_index, tokenlist in enumerate(tokenlists, start=1):
        sentence_text = tokenlist.metadata.get("text", "")

        normal_tokens = {
            token["id"]: token
            for token in tokenlist
            if is_real_token_id(token["id"])
        }

        for token in tokenlist:
            token_id = token["id"]

            if not is_multiword_token_id(token_id):
                continue

            start_id, _, end_id = token_id
            component_ids = list(range(start_id, end_id + 1))

            component_forms: List[str] = []
            component_upos: List[str] = []

            for component_id in component_ids:
                component = normal_tokens.get(component_id)
                if component is None:
                    continue

                component_forms.append(str(component.get("form", "")))
                component_upos.append(str(component.get("upos", "")))

            rows.append(
                {
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "surface_form": token.get("form"),
                    "range": f"{start_id}-{end_id}",
                    "component_forms": "+".join(component_forms),
                    "component_upos": "+".join(component_upos),
                    "lab_label": "+".join(component_upos),
                }
            )

    return pd.DataFrame(rows)


def tokens_containing_spaces(tokenlists) -> pd.DataFrame:
    """
    Find token forms that contain spaces.

    The Lab 5 instructions ask us to check whether Sequoia contains such tokens.
    In our main loader, spaces are removed from token forms before model input.
    """
    rows: List[Dict[str, object]] = []

    for sentence_index, tokenlist in enumerate(tokenlists, start=1):
        sentence_text = tokenlist.metadata.get("text", "")

        for token in tokenlist:
            form = token.get("form")
            if form is None:
                continue

            form = str(form)

            if " " not in form:
                continue

            rows.append(
                {
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "token_id": str(token["id"]),
                    "form": form,
                    "upos": token.get("upos"),
                }
            )

    return pd.DataFrame(rows)


def write_tokenization_example(output_path: Path) -> None:
    """
    Compare UD-style tokenization and mBERT WordPiece tokenization
    for the sentence requested in the assignment.

    Unicode escapes are used for accented characters to avoid Windows / PowerShell
    copy-paste encoding corruption.
    """
    raw_sentence = (
        "Pouvez-vous donner les m\u00eames garanties "
        "au sein de l'Union Europ\u00e9enne"
    )

    # UD-style analysis:
    # - pouvez-vous is split into Pouvez - vous
    # - au is a multiword token corresponding to à + le
    # - l'Union is segmented as l' + Union
    ud_original_tokens = [
        "Pouvez",
        "-",
        "vous",
        "donner",
        "les",
        "m\u00eames",
        "garanties",
        "\u00e0",
        "le",
        "sein",
        "de",
        "l'",
        "Union",
        "Europ\u00e9enne",
    ]

    # Lab-normalized tokenization:
    # keep multiword surface tokens such as au, and label them ADP+DET.
    ud_lab_tokens = [
        "Pouvez",
        "-",
        "vous",
        "donner",
        "les",
        "m\u00eames",
        "garanties",
        "au",
        "sein",
        "de",
        "l'",
        "Union",
        "Europ\u00e9enne",
    ]

    ud_lab_labels = [
        "VERB",
        "PUNCT",
        "PRON",
        "VERB",
        "DET",
        "ADJ",
        "NOUN",
        "ADP+DET",
        "NOUN",
        "ADP",
        "DET",
        "PROPN",
        "PROPN",
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    raw_mbert_tokens = tokenizer.tokenize(raw_sentence)

    encoding = tokenizer(
        ud_lab_tokens,
        is_split_into_words=True,
        padding=False,
        truncation=False,
    )

    aligned_mbert_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    word_ids = encoding.word_ids()

    lines: List[str] = []

    lines.append("RAW SENTENCE")
    lines.append("=" * 80)
    lines.append(raw_sentence)
    lines.append("")

    lines.append("UD ORIGINAL TOKENIZATION")
    lines.append("=" * 80)
    lines.append(" ".join(ud_original_tokens))
    lines.append("")
    lines.append("This reflects the UD components: au = \u00e0 + le.")
    lines.append("")

    lines.append("UD LAB TOKENIZATION")
    lines.append("=" * 80)
    lines.append(" ".join(ud_lab_tokens))
    lines.append("")

    lines.append("UD LAB LABELS")
    lines.append("=" * 80)
    for token, label in zip(ud_lab_tokens, ud_lab_labels):
        lines.append(f"{token:15s} -> {label}")
    lines.append("")

    lines.append("mBERT TOKENIZATION OF RAW STRING")
    lines.append("=" * 80)
    lines.append(" ".join(raw_mbert_tokens))
    lines.append("")

    lines.append("mBERT TOKENIZATION ALIGNED TO UD LAB TOKENS")
    lines.append("=" * 80)
    lines.append(
        "Only the first subtoken of each UD token receives the POS label. "
        "Continuation subtokens, [CLS], and [SEP] receive <pad> and are ignored."
    )
    lines.append("")

    previous_word_id = None

    for mbert_token, word_id in zip(aligned_mbert_tokens, word_ids):
        if word_id is None:
            ud_token = "<special>"
            training_label = "<pad>"
        elif word_id != previous_word_id:
            ud_token = ud_lab_tokens[word_id]
            training_label = ud_lab_labels[word_id]
        else:
            ud_token = ud_lab_tokens[word_id]
            training_label = "<pad>"

        lines.append(
            f"{mbert_token:15s} "
            f"word_id={str(word_id):4s} "
            f"ud={ud_token:15s} "
            f"training_label={training_label}"
        )

        previous_word_id = word_id

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenlists = load_tokenlists(SEQUOIA_PATH)

    standard_df = standard_upos_distribution(tokenlists)
    lab_df = lab_label_distribution(SEQUOIA_PATH)
    mwt_df = multiword_tokens(tokenlists)
    space_df = tokens_containing_spaces(tokenlists)

    standard_df.to_csv(
        OUTPUT_DIR / "sequoia_standard_upos_distribution.csv",
        index=False,
        encoding="utf-8",
    )
    lab_df.to_csv(
        OUTPUT_DIR / "sequoia_lab_label_distribution.csv",
        index=False,
        encoding="utf-8",
    )
    mwt_df.to_csv(
        OUTPUT_DIR / "sequoia_multiword_tokens.csv",
        index=False,
        encoding="utf-8",
    )
    space_df.to_csv(
        OUTPUT_DIR / "sequoia_space_tokens.csv",
        index=False,
        encoding="utf-8",
    )

    write_tokenization_example(OUTPUT_DIR / "sequoia_tokenization_example.txt")

    print("Sequoia analysis complete.")
    print()
    print(f"Sentences: {len(tokenlists)}")
    print(f"Standard UPOS labels: {len(standard_df)}")
    print(f"Lab-normalized labels: {len(lab_df)}")
    print(f"Multiword tokens: {len(mwt_df)}")
    print(f"Tokens containing spaces: {len(space_df)}")
    print()
    print("Top standard UPOS labels:")
    print(standard_df.head(10).to_string(index=False))
    print()
    print("Top lab-normalized labels:")
    print(lab_df.head(10).to_string(index=False))
    print()
    print("First multiword-token examples:")

    if len(mwt_df) > 0:
        print(
            mwt_df[
                ["surface_form", "component_forms", "component_upos", "lab_label"]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("No multiword tokens found.")

    print()
    print("Wrote:")
    print("  outputs/sequoia_standard_upos_distribution.csv")
    print("  outputs/sequoia_lab_label_distribution.csv")
    print("  outputs/sequoia_multiword_tokens.csv")
    print("  outputs/sequoia_space_tokens.csv")
    print("  outputs/sequoia_tokenization_example.txt")


if __name__ == "__main__":
    main()