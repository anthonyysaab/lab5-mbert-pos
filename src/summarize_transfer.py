"""
Summarize multilingual transfer results into 5x5 matrices.

Inputs:
- outputs/transfer_*_smoke_test.csv

Outputs:
- outputs/transfer_all_results.csv
- outputs/transfer_accuracy_matrix.csv
- outputs/transfer_macro_f1_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")

LANGUAGE_ORDER = ["ar_padt", "en_ewt", "es_gsd", "fr_gsd", "it_isdt"]


def main() -> None:
    paths = sorted(OUTPUT_DIR.glob("transfer_*_smoke_test.csv"))

    if not paths:
        raise FileNotFoundError("No transfer CSV files found in outputs/")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    df["train_language"] = pd.Categorical(
        df["train_language"],
        categories=LANGUAGE_ORDER,
        ordered=True,
    )
    df["target_language"] = pd.Categorical(
        df["target_language"],
        categories=LANGUAGE_ORDER,
        ordered=True,
    )

    df = df.sort_values(["train_language", "target_language"])

    accuracy_matrix = df.pivot(
        index="train_language",
        columns="target_language",
        values="accuracy",
    )

    f1_matrix = df.pivot(
        index="train_language",
        columns="target_language",
        values="macro_f1",
    )

    df.to_csv(OUTPUT_DIR / "transfer_all_results.csv", index=False, encoding="utf-8")
    accuracy_matrix.to_csv(OUTPUT_DIR / "transfer_accuracy_matrix.csv", encoding="utf-8")
    f1_matrix.to_csv(OUTPUT_DIR / "transfer_macro_f1_matrix.csv", encoding="utf-8")

    print("Accuracy matrix")
    print(accuracy_matrix.round(4))
    print()

    print("Macro-F1 matrix")
    print(f1_matrix.round(4))
    print()

    print("Wrote:")
    print("outputs/transfer_all_results.csv")
    print("outputs/transfer_accuracy_matrix.csv")
    print("outputs/transfer_macro_f1_matrix.csv")


if __name__ == "__main__":
    main()
