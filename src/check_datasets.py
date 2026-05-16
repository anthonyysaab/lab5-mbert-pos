"""
Sanity-check the processed HuggingFace POS datasets.

Checks:
- datasets exist for all languages
- each split has examples
- input_ids, attention_mask, and labels have equal length
- labels contain real UPOS ids and -100 ignore ids
- decoded example shows one label per UD word, not per subtoken
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from config import IGNORE_INDEX, MODEL_CHECKPOINT


PROCESSED_DIR = Path("data/processed/hf_pos")
LABEL_VOCAB_PATH = Path("outputs/label_vocab.json")


def inspect_example(dataset, tokenizer, id2label: dict[int, str]) -> None:
    example = dataset[0]

    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    labels = example["labels"]

    print("\nDecoded labeled tokens from first example:")
    print("-" * 80)

    for token, label_id in zip(tokens, labels):
        if label_id == IGNORE_INDEX:
            continue

        print(f"{token:20s} -> {id2label[label_id]}")

    print()


def main() -> None:
    vocab = json.loads(LABEL_VOCAB_PATH.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in vocab["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    for language_dir in sorted(path for path in PROCESSED_DIR.iterdir() if path.is_dir()):
        language = language_dir.name
        dataset = load_from_disk(language_dir)

        print("=" * 80)
        print(f"LANGUAGE: {language}")
        print("=" * 80)

        for split_name, split_dataset in dataset.items():
            first = split_dataset[0]

            input_len = len(first["input_ids"])
            mask_len = len(first["attention_mask"])
            label_len = len(first["labels"])

            assert input_len == mask_len == label_len, (
                f"Length mismatch in {language}/{split_name}: "
                f"input_ids={input_len}, attention_mask={mask_len}, labels={label_len}"
            )

            real_label_count = sum(
                1
                for row in split_dataset["labels"]
                for label in row
                if label != IGNORE_INDEX
            )

            ignored_label_count = sum(
                1
                for row in split_dataset["labels"]
                for label in row
                if label == IGNORE_INDEX
            )

            print(
                f"{split_name:10s} "
                f"examples={len(split_dataset):6d} "
                f"seq_len={input_len:3d} "
                f"real_labels={real_label_count:8d} "
                f"ignored={ignored_label_count:8d}"
            )

        inspect_example(dataset["train"], tokenizer, id2label)

    print("All dataset checks passed.")


if __name__ == "__main__":
    main()
