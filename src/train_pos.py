"""
Fine-tune mBERT for UPOS tagging on one UD language.

Default mode is a small smoke test:
- trains on a subset
- evaluates on validation
- saves metrics

Full 5x5 multilingual evaluation will be added after this works.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import IGNORE_INDEX, MODEL_CHECKPOINT


PROCESSED_DIR = Path("data/processed/hf_pos")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")


def load_label_vocab() -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab = json.loads(Path("outputs/label_vocab.json").read_text(encoding="utf-8"))

    label_list = vocab["label_list"]
    label2id = {label: int(idx) for label, idx in vocab["label2id"].items()}
    id2label = {int(idx): label for idx, label in vocab["id2label"].items()}

    return label_list, label2id, id2label


def maybe_select_subset(dataset, max_examples: int | None, seed: int):
    if max_examples is None or max_examples <= 0 or max_examples >= len(dataset):
        return dataset

    return dataset.shuffle(seed=seed).select(range(max_examples))


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels: List[int] = []
        true_predictions: List[int] = []

        for pred_row, label_row in zip(predictions, labels):
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == IGNORE_INDEX:
                    continue

                true_labels.append(int(label_id))
                true_predictions.append(int(pred_id))

        accuracy = accuracy_score(true_labels, true_predictions)
        macro_f1 = f1_score(true_labels, true_predictions, average="macro", zero_division=0)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    return compute_metrics


def make_training_args(
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainingArguments:
    """
    Build TrainingArguments while supporting both older and newer Transformers names.
    Some versions use evaluation_strategy, newer ones may use eval_strategy.
    """
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    args = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "max_steps": max_steps,
        "logging_steps": 10,
        "eval_steps": 25,
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "fp16": False,
    }

    if "evaluation_strategy" in params:
        args["evaluation_strategy"] = "steps"
    else:
        args["eval_strategy"] = "steps"

    return TrainingArguments(**args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en_ewt")
    parser.add_argument("--max-train-examples", type=int, default=500)
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    label_list, label2id, id2label = load_label_vocab()

    dataset_path = PROCESSED_DIR / args.language
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_from_disk(dataset_path)

    train_dataset = maybe_select_subset(dataset["train"], args.max_train_examples, args.seed)
    eval_dataset = maybe_select_subset(dataset["validation"], args.max_eval_examples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_dir = CHECKPOINT_DIR / f"{args.language}_smoke"

    training_args = make_training_args(
        output_dir=output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training language: {args.language}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print(f"Max steps: {args.max_steps}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(output_dir)

    metrics = {
        "language": args.language,
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "max_steps": args.max_steps,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }

    metrics_path = OUTPUT_DIR / f"{args.language}_smoke_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(f"Saved model to {output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()
