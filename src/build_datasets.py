"""
Build HuggingFace POS-tagging datasets for mBERT.

This script:
- reads UD .conllu files
- builds a shared UPOS label vocabulary
- tokenizes UD words with mBERT WordPiece tokenization
- aligns one POS label to the first subtoken of each word
- assigns -100 to special tokens and continuation subtokens
- saves one DatasetDict per language

Outputs:
- data/processed/hf_pos/{language}/
- outputs/label_vocab.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import IGNORE_INDEX, MODEL_CHECKPOINT
from load_conllu import read_conllu


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/hf_pos")
OUTPUT_DIR = Path("outputs")

MAX_LENGTH = 128


def collect_label_vocab() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = set()

    for path in sorted(RAW_DIR.rglob("*.conllu")):
        sentences = read_conllu(path)
        for _, tags in sentences:
            labels.update(tags)

    label_list = sorted(labels)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    return label_list, label2id, id2label


def align_labels_with_subtokens(
    word_ids: List[int | None],
    word_labels: List[str],
    label2id: Dict[str, int],
) -> List[int]:
    """
    Align word-level UPOS labels to subword-level mBERT tokens.

    Rule:
    - [CLS], [SEP], [PAD] get -100
    - first subtoken of a word gets the word POS label
    - later subtokens of the same word get -100
    """
    aligned_labels: List[int] = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            aligned_labels.append(label2id[word_labels[word_id]])
        else:
            aligned_labels.append(IGNORE_INDEX)

        previous_word_id = word_id

    return aligned_labels


def build_split_dataset(
    tokenizer,
    path: Path,
    label2id: Dict[str, int],
) -> Dataset:
    sentences = read_conllu(path)

    rows = {
        "tokens": [],
        "upos_tags": [],
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for words, tags in sentences:
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = encoding.word_ids()
        aligned_labels = align_labels_with_subtokens(word_ids, tags, label2id)

        rows["tokens"].append(words)
        rows["upos_tags"].append(tags)
        rows["input_ids"].append(encoding["input_ids"])
        rows["attention_mask"].append(encoding["attention_mask"])
        rows["labels"].append(aligned_labels)

    return Dataset.from_dict(rows)


def build_language_dataset(
    tokenizer,
    language_dir: Path,
    label2id: Dict[str, int],
) -> DatasetDict:
    datasets = {}

    for split in ["train", "dev", "test"]:
        path = language_dir / f"{split}.conllu"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        hf_split_name = "validation" if split == "dev" else split
        datasets[hf_split_name] = build_split_dataset(tokenizer, path, label2id)

    return DatasetDict(datasets)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    label_list, label2id, id2label = collect_label_vocab()

    label_vocab_path = OUTPUT_DIR / "label_vocab.json"
    label_vocab_path.write_text(
        json.dumps(
            {
                "label_list": label_list,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "ignore_index": IGNORE_INDEX,
                "max_length": MAX_LENGTH,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("Shared UPOS label vocabulary:")
    print(label_list)
    print()

    for language_dir in sorted(path for path in RAW_DIR.iterdir() if path.is_dir()):
        language = language_dir.name
        dataset = build_language_dataset(tokenizer, language_dir, label2id)

        output_path = PROCESSED_DIR / language
        dataset.save_to_disk(output_path)

        print(f"Saved {language} dataset to {output_path}")
        print(dataset)
        print()

    print(f"Wrote {label_vocab_path}")


if __name__ == "__main__":
    main()
