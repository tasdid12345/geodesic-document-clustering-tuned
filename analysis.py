"""Geodesic (Isomap-style) clustering of 20 Newsgroups: fixed-k baseline vs validation-tuned cosine-weighted variant."""
import json, os, sys, time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse.csgraph import shortest_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
CATS = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.misc"]
SOURCE = "https://ndownloader.figshare.com/files/5975967"
MAX_ROWS = 200_000          # subsampling cap to keep runtime small
K_BASELINE = 10             # fixed neighbourhood size of the baseline
K_GRID = [5, 10, 15, 20]    # neighbourhood sizes searched by the improved model


def load_data():
    """Load the 4 categories; retry with exponential backoff if the download fails."""
    for attempt in range(5):
        try:
            b = fetch_20newsgroups(subset="all", categories=CATS,
                                   remove=("headers", "footers", "quotes"),
                                   shuffle=True, random_state=SEED)
            return list(b.data), np.asarray(b.target)
        except Exception as e:  # urllib HTTPError (429/5xx) or network errors
            wait = 5 * 2 ** attempt
            print(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("could not download 20 Newsgroups")


def geodesic_cluster(X, k, metric, n_clusters):
    """kNN graph -> shortest-path (geodesic) distances -> average-linkage clustering."""
    G = kneighbors_graph(X, n_neighbors=k, mode="distance", metric=metric)
    G.data += 1e-9                       # keep zero-distance edges (duplicate docs) as edges
    G = G.maximum(G.T)                   # symmetrise the graph
    D = shortest_path(G, directed=False)
    # disconnected components give inf; replace with the largest finite geodesic distance
    D[np.isinf(D)] = D[np.isfinite(D)].max()
    return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                   linkage="average").fit_predict(D)


def main():
    texts, y = load_data()
    if len(texts) > MAX_ROWS:
        idx = np.random.RandomState(SEED).choice(len(texts), MAX_ROWS, replace=False)
        texts, y = [texts[i] for i in idx], y[idx]
    print(f"loaded {len(texts)} documents from {len(CATS)} categories")

    # Split ONCE: 70/30 stratified development vs test
    dev_txt, te_txt, y_dev, y_te = train_test_split(texts, y, test_size=0.3,
                                                    stratify=y, random_state=SEED)
    # Inside the development set: 70/30 stratified fit/validation split, used only to pick k
    fit_txt, va_txt, _, y_va = train_test_split(dev_txt, y_dev, test_size=0.3,
                                                stratify=y_dev, random_state=SEED)
    n_cls = len(CATS)

    # Model selection on the validation split only (vectoriser fitted on the fit part)
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, max_features=20000)
    vec.fit(fit_txt)
    X_va = vec.transform(va_txt)
    val_ari = {}
    for k in K_GRID:
        pred = geodesic_cluster(X_va, k, "cosine", n_cls)
        val_ari[k] = adjusted_rand_score(y_va, pred)
        print(f"validation ARI, k={k}: {val_ari[k]:.4f}")
    best_k = max(K_GRID, key=lambda k: (val_ari[k], -k))  # ties -> smaller k
    print(f"selected k = {best_k}")

    # Final evaluation: vectoriser fitted on the full development set, applied once to test
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, max_features=20000)
    vec.fit(dev_txt)
    X_te = vec.transform(te_txt)
    # Baseline: fixed k, Euclidean edge weights on L2-normalised TF-IDF
    pred_b = geodesic_cluster(X_te, K_BASELINE, "euclidean", n_cls)
    # Improved: validation-selected k, cosine-distance edge weights
    pred_i = geodesic_cluster(X_te, best_k, "cosine", n_cls)

    b_ari, i_ari = adjusted_rand_score(y_te, pred_b), adjusted_rand_score(y_te, pred_i)
    b_nmi = normalized_mutual_info_score(y_te, pred_b)
    i_nmi = normalized_mutual_info_score(y_te, pred_i)
    vals = [b_ari, i_ari, b_nmi, i_nmi]
    if not all(np.isfinite(vals)):
        sys.exit("non-finite metrics; failing")
    print(f"test ARI  baseline={b_ari:.4f} improved={i_ari:.4f}")
    print(f"test NMI  baseline={b_nmi:.4f} improved={i_nmi:.4f}")

    results = {
        "question": "Does a validation-tuned geodesic (kNN-graph shortest-path) distance clustering "
                    "beat a fixed-k geodesic baseline at grouping news documents by topic?",
        "method": "TF-IDF -> kNN graph -> shortest-path distances -> average-linkage clustering "
                  "with the true number of classes; test split evaluated once.",
        "dataset": {
            "name": "20 Newsgroups (4 categories, headers/footers/quotes removed)",
            "source_url": SOURCE,
            "n_rows": int(len(texts)),
            "n_train": int(len(dev_txt)),
            "n_test": int(len(te_txt)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "adjusted_rand_index (test)",
        "higher_is_better": True,
        "baseline": f"Geodesic clustering with fixed k={K_BASELINE} and Euclidean edge weights.",
        "improved": "Geodesic clustering with k selected from {5,10,15,20} on a validation split "
                    "of the development set, and cosine-distance edge weights.",
        "metrics": {
            "baseline_score": float(b_ari),
            "improved_score": float(i_ari),
            "improvement": float(i_ari - b_ari),
            "baseline_nmi": float(b_nmi),
            "improved_nmi": float(i_nmi),
            "improved_selected_k": int(best_k),
            **{f"validation_ari_k{k}": float(v) for k, v in val_ari.items()},
        },
        "notes": ("Clustering is deterministic given the split, so a single run is reported; "
                  "seed 42 controls all splits and subsampling. The test split is one 30% holdout, "
                  "so differences of a few points may be split noise. k is chosen by ARI on a "
                  "validation split inside the development set, never on the test set. "
                  "The graph is built over the test documents only (transductive). "
                  "fetch_20newsgroups does not expose a request timeout, so the retry/backoff "
                  "wrapper is the only download safeguard. The paper's own result is not stated "
                  "and it does not use this dataset."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optional chart: baseline vs improved
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = np.arange(2)
    ax.bar(xs - 0.2, [b_ari, b_nmi], 0.4, label="baseline")
    ax.bar(xs + 0.2, [i_ari, i_nmi], 0.4, label="improved")
    ax.set_xticks(xs, ["ARI (test)", "NMI (test)"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Geodesic clustering on 20 Newsgroups")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    print("wrote results.json and figures/baseline_vs_improved.png")


if __name__ == "__main__":
    main()
