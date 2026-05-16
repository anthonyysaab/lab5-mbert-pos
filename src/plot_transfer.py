"""
Create heatmaps for multilingual POS transfer results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")


def plot_matrix(csv_name: str, title: str, output_name: str) -> None:
    df = pd.read_csv(OUTPUT_DIR / csv_name, index_col=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(df.values)

    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    ax.set_xlabel("Test language")
    ax.set_ylabel("Training language")
    ax.set_title(title)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.iloc[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_matrix(
        "transfer_accuracy_matrix.csv",
        "Cross-lingual POS tagging accuracy",
        "transfer_accuracy_heatmap.png",
    )

    plot_matrix(
        "transfer_macro_f1_matrix.csv",
        "Cross-lingual POS tagging macro-F1",
        "transfer_macro_f1_heatmap.png",
    )

    print("Wrote outputs/transfer_accuracy_heatmap.png")
    print("Wrote outputs/transfer_macro_f1_heatmap.png")


if __name__ == "__main__":
    main()
