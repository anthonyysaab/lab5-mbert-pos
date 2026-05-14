"""
Configuration for Lab 5 — mBERT POS tagging.
"""

MODEL_CHECKPOINT = "bert-base-multilingual-cased"

PAD_LABEL = "<pad>"
IGNORE_INDEX = -100

DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs"
