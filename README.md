# Pashto Topic Classification Dataset

A paragraph-level topic classification dataset for Pashto, one of the most under-resourced major languages in NLP. It contains **27,437 labeled paragraphs across 12 topics**, collected from five Pashto news outlets.

Pashto is spoken by tens of millions of people, yet public labeled datasets for it are still rare — and topic classification datasets are almost nonexistent. This dataset was built to close that gap: real published text, labels grounded in how editors themselves categorized the articles, and enough volume per class to fine-tune modern models.

## What's inside

| File                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `pashto_topic_dataset.csv` | The full dataset: `id`, `text`, `topic`, `topic_ps` |

Every row is one paragraph of published news text with two labels — `topic` in English and `topic_ps` in Pashto — so the dataset drops into either an English-keyed or Pashto-keyed pipeline without any mapping work.

## The 12 topics

| topic         | topic_ps       | rows  |
| ------------- | -------------- | ----- |
| agriculture   | کرنه           | 2,490 |
| politics      | سیاست          | 2,469 |
| sports        | لوبې           | 2,447 |
| education     | ښوونه او روزنه | 2,425 |
| health        | روغتیا         | 2,412 |
| security      | امنیت          | 2,400 |
| technology    | ټکنالوژي       | 2,387 |
| world         | نړۍ            | 2,364 |
| economy       | اقتصاد         | 2,334 |
| entertainment | کلتور او هنر   | 2,175 |
| religion      | دین            | 2,175 |
| environment   | چاپېریال       | 1,359 |

## Where the text comes from

Paragraphs were collected from the public websites of five Pashto news outlets:

- **Pajhwok Afghan News**
- **Bakhtar News Agency**
- **BBC Pashto**
- **Shamshad News**
- **Nunn Taki Asia (نن ټکی اسیا)**

Each article was collected from a specific **editorial section** of its outlet (for example Pajhwok's کرنه section, or BBC Pashto's ساینس او ټکنالوژي topic page), and its paragraphs inherit that section as their label. In other words, the labeling was done by the professional editors who filed each story — we then verified and refined it. After automatic collection, the corpus went through a full cleaning pass: deduplication (exact and near-duplicate), removal of non-prose fragments, and a manual review of every paragraph the label-verification model disagreed with.

## Text quality and orthography

A deliberate decision in this dataset: **the Pashto text is kept exactly as its publishers wrote it.**

- No letter normalization of any kind — ي/ی, ك/ک, ۀ/ه distinctions are preserved as published. Pashto orthography varies across writers and regions, and collapsing those variants destroys information that real-world text will always contain.
- ZWNJ (U+200C, نيم‌فاصله) is preserved — it is part of correct Pashto compound writing (زده‌کړې، مرګ‌ژوبله).
- What _was_ standardized is everything that isn't language: whitespace, digit script (۰-۹ throughout), quotation marks (standard `"`), and stray invisible characters.

Paragraphs run 9–121 words (median 36) — long enough to carry a clear topic signal, short enough to train on cheaply.

## How to use it

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("pashto_topic_dataset.csv")

train, tmp = train_test_split(df, test_size=0.20, stratify=df["topic"], random_state=42)
val, test  = train_test_split(tmp, test_size=0.50, stratify=tmp["topic"], random_state=42)

X_train, y_train = train["text"], train["topic"]   # or train["topic_ps"]
```

Stratified sampling keeps every class at the same proportion in train, validation and test. A fixed `random_state` makes results reproducible and comparable across experiments.

## Baseline results

Three classic ML models with TF-IDF features, using the stratified 80/10/10 split above:

| Model                                                  | Test accuracy | Test macro-F1 |
| ------------------------------------------------------ | ------------- | ------------- |
| Linear SVM (tuned: word+char TF-IDF, balanced weights) | **85.0%**     | **0.849**     |
| Linear SVM                                             | 83.2%         | 0.831         |
| Logistic Regression                                    | 81.2%         | 0.812         |
| Multinomial Naive Bayes                                | 77.4%         | 0.767         |

A full training notebook (data checks, all three models, per-class results, confusion matrix, and a tuning section) is included. The linear baselines leave clear headroom — fine-tuning a multilingual transformer such as XLM-R is the natural next step, and we'd love to see what numbers you reach.

## Related work

This dataset is a companion to the **Pashto Sentiment Analysis Corpus** by the same author (10K+ labeled sentences, three sentiment classes). The two datasets share no text, so they can be combined freely in multi-task setups.

## License and citation

Released for research and educational use. The underlying text remains the property of the respective publishers; this dataset compiles short excerpts for research purposes with attribution to the source outlets.

If you use this dataset in your research, please cite it:

```
Khairullah. Pashto Topic Classification Dataset: 27K labeled paragraphs
across 12 topics from Pashto news media. 2026.
```

Questions, corrections, or results to share — please open a discussion. Contributions that grow the smaller classes (especially environment) are particularly welcome.
