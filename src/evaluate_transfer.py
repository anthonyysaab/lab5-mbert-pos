"""
Evaluate one fine-tuned mBERT POS model on all language test sets.

Example:
python src/evaluate_transfer.py --checkpoint checkpoints/en_ewt_smoke --train-language en_ewt --max-examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import MODEL_CHECKPOINT
from train_pos import compute_metrics_builder


PROCESSED_DIR = Path("data/processed/hf_pos")
OUTPUT_DIR = Path("outputs")


def load_label_vocab():
    vocab = json.loads(Path("outputs/label_vocab.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in vocab["id2label"].items()}
    label2id = {k: int(v) for k, v in vocab["label2id"].items()}
    return label2id, id2label


def maybe_select_subset(dataset, max_examples: int | None, seed: int):
    if max_examples is None or max_examples <= 0 or max_examples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_examples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-language", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    label2id, id2label = load_label_vocab()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(Path("checkpoints") / "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    rows = []

    for language_dir in sorted(path for path in PROCESSED_DIR.iterdir() if path.is_dir()):
        target_language = language_dir.name
        dataset = load_from_disk(language_dir)
        eval_dataset = maybe_select_subset(dataset[args.split], args.max_examples, args.seed)

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        row = {
            "train_language": args.train_language,
            "target_language": target_language,
            "split": args.split,
            "examples": len(eval_dataset),
            "eval_loss": metrics.get("eval_loss"),
            "accuracy": metrics.get("eval_accuracy"),
            "macro_f1": metrics.get("eval_macro_f1"),
            "runtime": metrics.get("eval_runtime"),
        }

        rows.append(row)

        print(
            f"{args.train_language} -> {target_language}: "
            f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
        )

    df = pd.DataFrame(rows)

    safe_checkpoint_name = Path(args.checkpoint).name
    output_path = OUTPUT_DIR / f"transfer_{safe_checkpoint_name}_{args.split}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print()
    print(f"Wrote {output_path}")
    print(df)


if __name__ == "__main__":
    main()
