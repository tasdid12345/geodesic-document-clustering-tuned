# Geodesic Distance Clustering of News Documents: Fixed-k vs Validation-Tuned

Motivation: Geodesic distances for web document clustering (2011) argues that shortest-path distances over a nearest-neighbour graph can capture the manifold structure of document collections better than direct similarity measures. This project checks whether tuning the neighbourhood size on held-out data, and using cosine-based edge weights, improves that approach on a public newsgroup corpus.

Question: Does a validation-tuned geodesic (kNN-graph shortest-path) distance clustering beat a fixed-k geodesic baseline at grouping news documents by topic?

Dataset: 20 Newsgroups, four categories (sci.space, rec.sport.hockey, comp.graphics, talk.politics.misc), with headers, footers and quotes removed. It is loaded through scikit-learn's `fetch_20newsgroups` (downloaded from figshare, no login). This is not the paper's own dataset.

Baseline: geodesic clustering with a fixed neighbourhood size and Euclidean edge weights on TF-IDF vectors, followed by average-linkage clustering.

Improved model: the neighbourhood size is chosen from a small grid by adjusted Rand index on a validation split inside the development data, and edges are weighted by cosine distance. Everything else is identical.

Metrics: adjusted Rand index on the held-out test split (main), normalised mutual information (secondary).

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and a chart to figures/).

## Results (latest run)

On the 20 Newsgroups 4-category test split (1,121 documents), adjusted Rand index rose from 0.0001 for a standard geodesic-distance clustering with a fixed neighbourhood size to 0.488 for the validation-tuned version (k chosen from {5,10,15,20}, cosine edges), from a single run on one holdout.

| Metric | Value |
|---|---|
| baseline_score | 0.0001027 |
| improved_score | 0.4882 |
| improvement | 0.4881 |
| baseline_nmi | 0.005391 |
| improved_nmi | 0.4736 |
| improved_selected_k | 20 |
| validation_ari_k5 | 0.3277 |
| validation_ari_k10 | 0.3672 |
| validation_ari_k15 | 0.3458 |
| validation_ari_k20 | 0.4121 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Clustering is deterministic given the split, so a single run is reported; seed 42 controls all splits and subsampling. The test split is one 30% holdout, so differences of a few points may be split noise. k is chosen by ARI on a validation split inside the development set, never on the test set. The graph is built over the test documents only (transductive). fetch_20newsgroups does not expose a request timeout, so the retry/backoff wrapper is the only download safeguard. The paper's own result is not stated and it does not use this dataset.

## Reference

This project is motivated by the following paper:

> S Tekir, F Mansmann, D Keim (2011). *Geodesic distances for web document clustering*. 2011 IEEE Symposium on. https://ieeexplore.ieee.org/abstract/document/5949449/

Paper author: Prof. Daniel Keim, Universität Konstanz (daniel.keim@uni-konstanz.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
