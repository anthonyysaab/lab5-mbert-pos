# Lab 5 — How Multilingual is Multilingual BERT?

This repository contains the code and report for Lab 5.

Goal:
Fine-tune multilingual BERT (`bert-base-multilingual-cased`) for POS tagging using Universal Dependencies treebanks, then evaluate cross-lingual transfer across five languages.

Main tasks:
- Load `.conllu` UD treebanks
- Extract tokens and UPOS labels
- Align UD tokenization with mBERT subword tokenization
- Build HuggingFace datasets
- Fine-tune mBERT POS taggers
- Evaluate a 5 × 5 multilingual transfer matrix
