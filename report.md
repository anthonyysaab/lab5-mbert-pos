# Lab 5 — How Multilingual is Multilingual BERT?

Anthony Saab

## 1. Introduction

This lab investigates how multilingual BERT (`bert-base-multilingual-cased`) behaves on part-of-speech tagging across several languages. The experiment fine-tunes mBERT on Universal Dependencies treebanks and evaluates both same-language performance and cross-lingual transfer.

The five selected languages are:

| Code | Language / Treebank |
|---|---|
| `ar_padt` | Arabic PADT |
| `en_ewt` | English EWT |
| `es_gsd` | Spanish GSD |
| `fr_gsd` | French GSD |
| `it_isdt` | Italian ISDT |

This selection includes one Semitic language, English, and three Romance languages. It therefore allows us to test whether cross-lingual transfer is stronger between typologically and genealogically closer languages.

---

## 2. Universal Dependencies data

The data comes from Universal Dependencies `.conllu` files. Each `.conllu` file contains sentences annotated with linguistic information such as token form, lemma, UPOS tag, XPOS tag, morphological features, dependency head, and dependency relation.

For this lab, I extracted:

```text
FORM  → word/token surface form
UPOS  → universal part-of-speech label
```

Multiword-token range lines and empty nodes were ignored. I kept only normal integer-indexed tokens, because the task is word-level UPOS tagging.

The shared UPOS inventory contains 17 labels:

```text
ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
```

Some splits contain fewer than 17 labels. This does not mean the annotation scheme changes. It only means that rare categories such as `INTJ`, `SYM`, or `PART` do not appear in every dev/test split.

---

## 3. Corpus statistics

The corpus statistics were saved in:

```text
outputs/corpus_stats.csv
outputs/upos_distribution.csv
```

Summary of the main training sets:

| Language | Train sentences | Train tokens | UPOS labels |
|---|---:|---:|---:|
| Arabic PADT | 6,075 | 223,881 | 17 |
| English EWT | 12,544 | 204,578 | 17 |
| Spanish GSD | 14,186 | 382,327 | 17 |
| French GSD | 14,450 | 354,647 | 16 |
| Italian ISDT | 13,121 | 276,014 | 17 |

The most frequent labels across the treebanks are generally `NOUN`, `ADP`, `DET`, `PUNCT`, and `VERB`. This creates a class imbalance: accuracy can be high because frequent labels dominate, while macro-F1 is more sensitive to rare labels.

---

## 4. UD tokenization vs mBERT tokenization

UD tokenization is word-level: each word has one UPOS label.

mBERT uses WordPiece tokenization. One UD word may become one or several subtokens. This creates the central preprocessing problem of the lab:

```text
UD word-level labels must be aligned with mBERT subword-level input.
```

The alignment rule used was:

```text
First subtoken of a word       → receives the UPOS label
Continuation subtokens         → receive -100
Special tokens / padding       → receive -100
```

The value `-100` is ignored by PyTorch/HuggingFace during loss computation.

Example:

```text
word:      unbelievable
subtokens: un ##bel ##ievable
labels:    ADJ -100 -100
```

This means the model only learns from the first subtoken of each UD word.

---

## 5. Tokenization analysis

The tokenization analysis was saved in:

```text
outputs/tokenization_stats.csv
outputs/tokenization_examples.txt
```

Important observations:

| Language | Approx. subtokens per word | Split-word rate |
|---|---:|---:|
| Arabic PADT | ~1.76 | ~40% |
| English EWT | ~1.22–1.31 | ~13–15% |
| Spanish GSD | ~1.25–1.27 | ~16–17% |
| French GSD | ~1.31 | ~21% |
| Italian ISDT | ~1.33 | ~22–23% |

Arabic is split much more heavily by mBERT than the European languages. This matters because heavier segmentation increases the mismatch between the linguistic word-level annotation and the model’s subword input representation.

---

## 6. Dataset creation

The processed HuggingFace datasets contain:

```text
input_ids
attention_mask
labels
```

The datasets were saved under:

```text
data/processed/hf_pos/
```

The shared label vocabulary was saved as:

```text
outputs/label_vocab.json
```

All sequences were padded/truncated to length 128. This avoids overly long examples and keeps training feasible on the available GPU.

---

## 7. Model and training setup

The model used was:

```text
bert-base-multilingual-cased
```

For each language, I loaded mBERT as a token-classification model with 17 UPOS labels.

Reduced experimental setup:

| Parameter | Value |
|---|---:|
| Train examples per language | 2,000 |
| Validation examples per language | 500 |
| Max steps | 300 |
| Batch size | 4 |
| Max sequence length | 128 |
| Device | CUDA |
| GPU | Quadro RTX 3000, 6 GB VRAM |

When loading the model, HuggingFace reported missing `classifier.weight` and `classifier.bias`. This is expected: the pretrained mBERT checkpoint does not contain a POS-tagging classifier head. A new classification layer is initialized and trained during fine-tuning.

It also reported unexpected pretraining weights such as `cls.predictions.*` and `cls.seq_relationship.*`. These belong to BERT’s original pretraining heads and are not needed for POS tagging.

---

## 8. Same-language validation results

| Training language | Validation accuracy | Validation macro-F1 |
|---|---:|---:|
| Arabic PADT | 0.9416 | 0.7891 |
| English EWT | 0.9237 | 0.7619 |
| Spanish GSD | 0.9548 | 0.7706 |
| French GSD | 0.9694 | 0.7918 |
| Italian ISDT | 0.9675 | 0.7279 |

These results show that the model successfully learns POS tagging from each treebank, even with a reduced training setup.

---

## 9. Cross-lingual transfer experiment

Each model was trained on one language and evaluated on the test sets of all five languages. This gives a 5 × 5 multilingual transfer matrix.

Rows represent the training language. Columns represent the test language.

---

## 10. Accuracy matrix

| Train \ Test | ar_padt | en_ewt | es_gsd | fr_gsd | it_isdt |
|---|---:|---:|---:|---:|---:|
| ar_padt | 0.9428 | 0.5441 | 0.6000 | 0.5962 | 0.6102 |
| en_ewt | 0.5541 | 0.9257 | 0.8594 | 0.8781 | 0.9082 |
| es_gsd | 0.6495 | 0.8219 | 0.9498 | 0.9335 | 0.9468 |
| fr_gsd | 0.6820 | 0.8177 | 0.9148 | 0.9619 | 0.9453 |
| it_isdt | 0.6668 | 0.8337 | 0.9210 | 0.9387 | 0.9673 |

![Cross-lingual POS tagging accuracy](outputs/transfer_accuracy_heatmap.png)

---

## 11. Macro-F1 matrix

| Train \ Test | ar_padt | en_ewt | es_gsd | fr_gsd | it_isdt |
|---|---:|---:|---:|---:|---:|
| ar_padt | 0.8220 | 0.3903 | 0.4008 | 0.4266 | 0.4576 |
| en_ewt | 0.3987 | 0.7579 | 0.6162 | 0.6543 | 0.7161 |
| es_gsd | 0.4951 | 0.6417 | 0.7639 | 0.7732 | 0.8255 |
| fr_gsd | 0.4753 | 0.6224 | 0.6589 | 0.7747 | 0.7914 |
| it_isdt | 0.4600 | 0.6351 | 0.6685 | 0.7464 | 0.8239 |

![Cross-lingual POS tagging macro-F1](outputs/transfer_macro_f1_heatmap.png)

---

## 12. Interpretation of the results

The strongest scores are on the diagonal. This means that training and testing on the same language gives the best performance. This is expected because the model sees the same treebank conventions, vocabulary distribution, morphology, and syntax during training.

The Romance languages transfer very well to one another. Spanish, French, and Italian produce high cross-lingual accuracy scores:

```text
Spanish → French:  0.9335
Spanish → Italian: 0.9468
French → Italian:  0.9453
Italian → French:  0.9387
```

This supports the idea that mBERT has shared multilingual representations that are especially useful when languages are typologically and genealogically close.

English transfers reasonably well to the Romance languages:

```text
English → Spanish: 0.8594
English → French:  0.8781
English → Italian: 0.9082
```

However, English transfers much less well to Arabic:

```text
English → Arabic: 0.5541
```

Arabic is the weakest cross-lingual target overall. This is probably caused by several factors: different script, richer morphology, different subword segmentation behavior, and greater typological distance from the other selected languages.

Macro-F1 is lower than accuracy. This is because POS labels are imbalanced. Common tags such as `NOUN`, `ADP`, `DET`, `VERB`, and `PUNCT` dominate accuracy, while rare labels such as `INTJ`, `SYM`, `PART`, and `X` affect macro-F1 more strongly.

---

## 13. Is mBERT multilingual?

The answer is yes, but with limits.

mBERT is multilingual because a model trained on one language can still perform above random baseline on other languages. The transfer is especially strong between related languages, such as Spanish, French, and Italian.

However, mBERT is not equally multilingual for all language pairs. Transfer performance decreases when the target language is more distant in script, morphology, tokenization behavior, and syntactic distribution.

The results therefore support a nuanced conclusion:

```text
mBERT learns shared multilingual representations, but transfer quality depends strongly on linguistic distance and tokenization compatibility.
```

---

## 14. Conclusion

This lab implemented a full POS-tagging pipeline with multilingual BERT:

```text
.conllu files
→ UD words and UPOS labels
→ mBERT WordPiece tokenization
→ label alignment
→ HuggingFace Dataset creation
→ mBERT fine-tuning
→ same-language and cross-lingual evaluation
```

The experiment shows that mBERT can transfer POS-tagging knowledge across languages, especially between related Romance languages. The model performs best in same-language evaluation, but cross-lingual transfer remains strong when the source and target languages are close.

Arabic behaves differently from the European languages. It has heavier mBERT subword segmentation and weaker transfer results, suggesting that multilinguality is affected by script, morphology, and tokenizer coverage.

Overall, mBERT is multilingual, but its multilinguality is structured rather than uniform.