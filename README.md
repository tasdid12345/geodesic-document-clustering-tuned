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

## Reference

This project is motivated by the following paper:

> S Tekir, F Mansmann, D Keim (2011). *Geodesic distances for web document clustering*. 2011 IEEE Symposium on. https://ieeexplore.ieee.org/abstract/document/5949449/

Paper author: Prof. Daniel Keim, Universität Konstanz (daniel.keim@uni-konstanz.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
