# Lab 5 — How Multilingual is Multilingual BERT?

Anthony Saab

## 1. Introduction

This lab investigates how multilingual BERT (`bert-base-multilingual-cased`) behaves on part-of-speech tagging across several languages. The experiment fine-tunes mBERT on Universal Dependencies treebanks and evaluates both same-language performance and cross-lingual transfer.

The five selected languages for the multilingual experiment are:

| Code | Language / Treebank |
|---|---|
| `ar_padt` | Arabic PADT |
| `en_ewt` | English EWT |
| `es_gsd` | Spanish GSD |
| `fr_gsd` | French GSD |
| `it_isdt` | Italian ISDT |

This selection includes one Semitic language, English, and three Romance languages. It therefore allows us to test whether cross-lingual transfer is stronger between typologically and genealogically closer languages.

Before the multilingual experiment, I also analyzed the French Sequoia test set. Sequoia is used as a warm-up corpus in the assignment because it illustrates several Universal Dependencies issues: label distribution, multiword tokens, tokens containing spaces, and the mismatch between UD tokenization and mBERT WordPiece tokenization.

---

## 2. Sequoia warm-up analysis

The French Sequoia test set contains 456 sentences. With standard UD counting, where only normal integer-indexed syntactic tokens are counted, the file contains 15 UPOS labels. The most frequent labels are:

| UPOS | Count | Percentage |
|---|---:|---:|
| NOUN | 2125 | 21.1569 |
| ADP | 1630 | 16.2286 |
| DET | 1486 | 14.7949 |
| PUNCT | 1084 | 10.7925 |
| VERB | 781 | 7.7758 |
| ADJ | 636 | 6.3321 |
| PROPN | 480 | 4.7790 |
| ADV | 417 | 4.1517 |
| PRON | 398 | 3.9626 |
| AUX | 345 | 3.4349 |

The distribution is clearly imbalanced. Nominal and functional categories such as `NOUN`, `ADP`, and `DET` dominate the corpus, while other categories occur much less often. This matters for evaluation: accuracy can be high because the model predicts frequent classes well, while macro-F1 is more sensitive to rare labels.

The Sequoia test set contains 310 multiword tokens. Most of them are French contractions such as `des`, `du`, `au`, `Aux`, and `desdites`. In UD, these are represented as one surface token plus separate grammatical components:

```text
des      = de + les       -> ADP+DET
du       = de + le        -> ADP+DET
Aux      = À + les        -> ADP+DET
au       = à + le         -> ADP+DET
desdites = de + lesdites  -> ADP+DET
```

For this lab, I kept the surface token and assigned it a concatenated label such as `ADP+DET`. This preserves the surface form seen by mBERT while still encoding the grammatical analysis given by UD.

The Sequoia test set also contains 13 tokens whose form contains spaces. Since the assignment states that such spaces should be removed, the loader normalizes token forms by deleting internal whitespace before tokenization.

For the sentence:

```text
Pouvez-vous donner les mêmes garanties au sein de l'Union Européenne
```

UD tokenization and mBERT tokenization differ. A UD-style grammatical tokenization is:

```text
Pouvez - vous donner les mêmes garanties au sein de l' Union Européenne
```

with labels such as:

```text
Pouvez      -> VERB
-           -> PUNCT
vous        -> PRON
garanties   -> NOUN
au          -> ADP+DET
Européenne  -> PROPN
```

mBERT uses WordPiece subword tokenization:

```text
Po ##uve ##z - vous donner les mêmes gara ##nties au sein de l ' Union Euro ##pée ##nne
```

This creates a label-alignment problem: UD gives one POS label per UD token, but mBERT may split one token into several subtokens. The solution is to assign the POS label only to the first subtoken and assign `<pad>` / `-100` to continuation subtokens and special tokens:

```text
Po       -> VERB
##uve    -> <pad>
##z      -> <pad>

gara     -> NOUN
##nties  -> <pad>

Euro     -> PROPN
##pée    -> <pad>
##nne    -> <pad>
```

This ensures that training loss and evaluation metrics are computed only on real UD-token positions, not on artificial WordPiece fragments.

---

## 3. Universal Dependencies data

The main multilingual data comes from Universal Dependencies `.conllu` files. Each `.conllu` file contains sentences annotated with linguistic information such as token form, lemma, UPOS tag, XPOS tag, morphological features, dependency head, and dependency relation.

For this lab, I extracted:

```text
FORM  → word/token surface form
UPOS  → universal part-of-speech label
```

Empty nodes were ignored. Multiword-token range lines were handled specially: the surface token was kept, and the UPOS labels of its components were concatenated. For example, a French contraction such as `au`, whose components are `à` and `le`, is represented as:

```text
au -> ADP+DET
```

This gives mBERT the surface token that actually appears in the text while preserving the grammatical information from the UD annotation.

The ordinary UPOS inventory contains 17 universal labels:

```text
ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
```

After multiword-token normalization, the project label vocabulary also includes composite labels such as `ADP+DET`, `VERB+PRON`, or `AUX+PART`. These composite labels are not new UPOS categories; they encode the grammatical composition of a single surface token.

---

## 4. Corpus statistics

The corpus statistics were saved in:

```text
outputs/corpus_stats.csv
outputs/upos_distribution.csv
```

Summary of the main training sets after multiword-token normalization:

| Language | Train sentences | Train tokens | Label count |
|---|---:|---:|---:|
| Arabic PADT | 6,075 | 191,869 | 111 |
| English EWT | 12,544 | 201,963 | 38 |
| Spanish GSD | 14,186 | 375,031 | 21 |
| French GSD | 14,450 | 344,961 | 18 |
| Italian ISDT | 13,121 | 257,612 | 25 |

The label count is larger than 17 for some languages because multiword tokens can produce composite labels. Arabic has the largest label inventory because its treebank contains many cliticized forms and morphologically complex surface tokens.

The most frequent labels across the treebanks are generally `NOUN`, `ADP`, `DET`, `PUNCT`, and `VERB`. This creates a class imbalance: accuracy can be high because frequent labels dominate, while macro-F1 is more sensitive to rare labels and composite labels.

---

## 5. UD tokenization vs mBERT tokenization

UD tokenization is word-level or grammatical-token-level: each UD token receives one label.

mBERT uses WordPiece tokenization. One UD token may become one or several subtokens. This creates the central preprocessing problem of the lab:

```text
UD token-level labels must be aligned with mBERT subword-level input.
```

The alignment rule used was:

```text
First subtoken of a UD token     → receives the POS label
Continuation subtokens           → receive -100
Special tokens / padding         → receive -100
```

The value `-100` is ignored by PyTorch/HuggingFace during loss computation.

Example:

```text
word:      garanties
subtokens: gara ##nties
labels:    NOUN -100
```

This means the model only learns from the first subtoken of each UD token. Continuation subtokens are still part of the input representation, but they do not contribute directly to the loss or evaluation metrics.

---

## 6. Tokenization analysis

The tokenization analysis was saved in:

```text
outputs/tokenization_stats.csv
outputs/tokenization_examples.txt
```

Important observations after multiword-token normalization:

| Language | Approx. subtokens per word | Split-word rate |
|---|---:|---:|
| Arabic PADT | ~1.97–2.02 | ~50–51% |
| English EWT | ~1.24–1.32 | ~13–15% |
| Spanish GSD | ~1.26–1.28 | ~17–18% |
| French GSD | ~1.32–1.33 | ~21–22% |
| Italian ISDT | ~1.36 | ~24–25% |

Arabic is split much more heavily by mBERT than the European languages. This matters because heavier segmentation increases the mismatch between the linguistic token-level annotation and the model’s subword input representation.

The Romance languages show similar behavior, with French and Italian having slightly higher split rates than Spanish. English has the lowest split rate among the European treebanks. These differences are relevant to cross-lingual transfer because mBERT does not represent all languages with the same degree of tokenization compatibility.

---

## 7. Dataset creation

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

All sequences were padded/truncated to length 128. This avoids overly long examples and keeps training feasible on the available GPU. Truncation means that if a sentence produces more than 128 WordPiece tokens, only the first 128 positions are kept. This is a practical compromise: it makes training cheaper, but a small amount of sentence-final material may be lost for very long examples.

---

## 8. Model and training setup

The model used was:

```text
bert-base-multilingual-cased
```

For each language, I loaded mBERT as a token-classification model. The classifier head was initialized for the project’s POS label vocabulary.

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

## 9. Same-language validation results

| Training language | Validation accuracy | Validation macro-F1 |
|---|---:|---:|
| Arabic PADT | 0.9416 | 0.7891 |
| English EWT | 0.9237 | 0.7619 |
| Spanish GSD | 0.9548 | 0.7706 |
| French GSD | 0.9694 | 0.7918 |
| Italian ISDT | 0.9675 | 0.7279 |

These results show that the model successfully learns POS tagging from each treebank, even with a reduced training setup.

Accuracy is consistently higher than macro-F1 because the POS distribution is imbalanced. Frequent labels such as `NOUN`, `ADP`, `DET`, `PUNCT`, and `VERB` dominate the token counts, while rarer labels and composite labels are harder to learn reliably.

---

## 10. Cross-lingual transfer experiment

Each model was trained on one language and evaluated on the test sets of all five languages. This gives a 5 × 5 multilingual transfer matrix.

Rows represent the training language. Columns represent the test language.

This setup directly tests whether mBERT’s learned POS representations transfer across languages. If mBERT is strongly multilingual, a model trained on one language should still perform well on other languages. If transfer depends heavily on language similarity, then related languages should transfer better than distant languages.

---

## 11. Accuracy matrix

| Train \ Test | ar_padt | en_ewt | es_gsd | fr_gsd | it_isdt |
|---|---:|---:|---:|---:|---:|
| ar_padt | 0.9428 | 0.5441 | 0.6000 | 0.5962 | 0.6102 |
| en_ewt | 0.5541 | 0.9257 | 0.8594 | 0.8781 | 0.9082 |
| es_gsd | 0.6495 | 0.8219 | 0.9498 | 0.9335 | 0.9468 |
| fr_gsd | 0.6820 | 0.8177 | 0.9148 | 0.9619 | 0.9453 |
| it_isdt | 0.6668 | 0.8337 | 0.9210 | 0.9387 | 0.9673 |

![Cross-lingual POS tagging accuracy](outputs/transfer_accuracy_heatmap.png)

---

## 12. Macro-F1 matrix

| Train \ Test | ar_padt | en_ewt | es_gsd | fr_gsd | it_isdt |
|---|---:|---:|---:|---:|---:|
| ar_padt | 0.8220 | 0.3903 | 0.4008 | 0.4266 | 0.4576 |
| en_ewt | 0.3987 | 0.7579 | 0.6162 | 0.6543 | 0.7161 |
| es_gsd | 0.4951 | 0.6417 | 0.7639 | 0.7732 | 0.8255 |
| fr_gsd | 0.4753 | 0.6224 | 0.6589 | 0.7747 | 0.7914 |
| it_isdt | 0.4600 | 0.6351 | 0.6685 | 0.7464 | 0.8239 |

![Cross-lingual POS tagging macro-F1](outputs/transfer_macro_f1_heatmap.png)

---

## 13. Interpretation of the results

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

Macro-F1 is lower than accuracy. This is because POS labels are imbalanced. Common tags such as `NOUN`, `ADP`, `DET`, `VERB`, and `PUNCT` dominate accuracy, while rare labels such as `INTJ`, `SYM`, `PART`, `X`, and composite multiword labels affect macro-F1 more strongly.

---

## 14. Is mBERT multilingual?

The answer is yes, but with limits.

mBERT is multilingual because a model trained on one language can still perform above a random baseline on other languages. The transfer is especially strong between related languages, such as Spanish, French, and Italian.

However, mBERT is not equally multilingual for all language pairs. Transfer performance decreases when the target language is more distant in script, morphology, tokenization behavior, and syntactic distribution.

The results therefore support a nuanced conclusion:

```text
mBERT learns shared multilingual representations, but transfer quality depends strongly on linguistic distance and tokenization compatibility.
```

This is also visible in the tokenization analysis. Arabic has much heavier WordPiece segmentation than the European languages, which means the model must map more subword fragments back to UD token-level labels. The Romance languages, by contrast, have more similar tokenization behavior and stronger transfer scores.

---

## 15. Conclusion

This lab implemented a full POS-tagging pipeline with multilingual BERT:

```text
.conllu files
→ UD tokens and UPOS labels
→ multiword-token normalization
→ mBERT WordPiece tokenization
→ label alignment
→ HuggingFace Dataset creation
→ mBERT fine-tuning
→ same-language and cross-lingual evaluation
```

The Sequoia warm-up analysis showed why tokenization is a central issue in this lab. UD may represent French contractions such as `au`, `du`, and `des` as multiword tokens with grammatical components like `ADP+DET`, while mBERT may further split surface tokens into WordPiece subtokens. This requires an explicit alignment strategy: keep the label on the first subtoken and ignore continuation subtokens with `-100`.

The multilingual experiment shows that mBERT can transfer POS-tagging knowledge across languages, especially between related Romance languages. The model performs best in same-language evaluation, but cross-lingual transfer remains strong when the source and target languages are close.

Arabic behaves differently from the European languages. It has heavier mBERT subword segmentation and weaker transfer results, suggesting that multilinguality is affected by script, morphology, and tokenizer coverage.

Overall, mBERT is multilingual, but its multilinguality is structured rather than uniform.