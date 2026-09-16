"""Q4 -- Offline evaluation harness.

Consumes the retrieval results Q2/Q3 already produced (outputs/<dataset>/retrieval_results_val.parquet)
-- no index is rebuilt here. For both BM25 and embedding methods:
  1. AUC, MRR, nDCG@5, nDCG@10 -- computed per impression against that impression's own
     official candidate list (the only place labels exist), averaged across impressions.
  2. Beyond-accuracy: intra-list diversity, novelty, coverage -- computed on each method's
     own top-10 FULL-CORPUS retrieval (i.e. what it would actually recommend), not a
     re-ranking of the officially-provided candidates.
  3. Slicing: cold-start (history_len <= threshold) vs warm users, and head vs tail
     articles (by training-set click popularity).
  4. Bootstrap 95% CI for every accuracy metric.

Usage:
    python pipeline/evaluate.py --dataset mind
    python pipeline/evaluate.py --dataset ebnerd
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.common.embeddings import EmbeddingIndex
from pipeline.common.metrics import coverage, evaluate_impressions, intra_list_diversity, novelty, summarize

COLD_START_THRESHOLD = 5   # history_len <= this => cold-start (assignment's own example threshold)
HEAD_FRACTION = 0.2        # top 20% most-clicked (in TRAIN) articles = "head"


def _train_popularity(fs) -> dict:
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["candidates", "labels"])
    counter = Counter()
    for cands, labels in zip(train["candidates"], train["labels"]):
        for c, l in zip(cands, labels):
            if l == 1:
                counter[c] += 1
    return dict(counter)


def _head_tail_mask(clicked_ids: list, popularity: dict, head_frac: float) -> np.ndarray:
    """True = impression's clicked article is a 'head' (popular) article."""
    if not popularity:
        return np.zeros(len(clicked_ids), dtype=bool)
    cutoff = np.quantile(list(popularity.values()), 1 - head_frac)
    return np.array([any(popularity.get(c, 0) >= cutoff for c in clicked) for clicked in clicked_ids])


def _metrics_for_mask(labels_all, scores_all, mask) -> dict:
    labels_sub = [l for l, m in zip(labels_all, mask) if m]
    scores_sub = [s for s, m in zip(scores_all, mask) if m]
    return summarize(evaluate_impressions(labels_sub, scores_sub), n_boot=1000, seed=config.RANDOM_SEED)


def run(dataset: str):
    fs = config.FEATURE_STORE / dataset
    out_dir = config.OUTPUTS / dataset

    results = pd.read_parquet(out_dir / "retrieval_results_val.parquet")
    articles = pd.read_parquet(fs / "articles.parquet")
    article_vecs = np.load(fs / "article_embeddings.npy")
    emb_index = EmbeddingIndex(articles["article_id"].tolist(), article_vecs)
    popularity = _train_popularity(fs)
    total_clicks = sum(popularity.values())
    catalog_size = len(articles)

    labels_all = [np.array(l) for l in results["labels"]]
    clicked_ids = [[c for c, l in zip(cands, labs) if l == 1]
                   for cands, labs in zip(results["candidates"], results["labels"])]

    warm_mask = (results["history_len"] > COLD_START_THRESHOLD).to_numpy()
    cold_mask = ~warm_mask
    head_mask = _head_tail_mask(clicked_ids, popularity, HEAD_FRACTION)
    tail_mask = ~head_mask

    report = {"dataset": dataset, "n_impressions": len(results),
              "cold_start_threshold": COLD_START_THRESHOLD, "head_fraction": HEAD_FRACTION,
              "n_cold": int(cold_mask.sum()), "n_warm": int(warm_mask.sum()),
              "n_head_click": int(head_mask.sum()), "n_tail_click": int(tail_mask.sum())}

    for method, score_col, top10_col in [("bm25", "bm25_scores", "bm25_top10"),
                                          ("embedding", "embed_scores", "embed_top10")]:
        scores_all = [np.array(s) for s in results[score_col]]

        overall = summarize(evaluate_impressions(labels_all, scores_all), n_boot=1000, seed=config.RANDOM_SEED)
        cold = _metrics_for_mask(labels_all, scores_all, cold_mask)
        warm = _metrics_for_mask(labels_all, scores_all, warm_mask)
        head = _metrics_for_mask(labels_all, scores_all, head_mask)
        tail = _metrics_for_mask(labels_all, scores_all, tail_mask)

        top10_lists = results[top10_col].tolist()
        div = intra_list_diversity(top10_lists, emb_index)  # content-space diversity, same embedding space for both
        nov = novelty(top10_lists, popularity, total_clicks, catalog_size)
        cov = coverage(top10_lists, catalog_size)

        report[method] = {
            "overall": overall,
            "slices": {"cold_start": cold, "warm": warm, "head_articles": head, "tail_articles": tail},
            "beyond_accuracy": {
                "intra_list_diversity_mean": float(np.nanmean(div)),
                "novelty_mean": float(np.nanmean(nov)),
                "coverage": cov,
            },
        }
        print(f"[{dataset}] {method}: AUC={overall['auc']['mean']:.4f} "
              f"[{overall['auc']['ci_low']:.4f},{overall['auc']['ci_high']:.4f}]  "
              f"MRR={overall['mrr']['mean']:.4f}  nDCG@5={overall['ndcg5']['mean']:.4f}  "
              f"nDCG@10={overall['ndcg10']['mean']:.4f}  diversity={report[method]['beyond_accuracy']['intra_list_diversity_mean']:.3f}  "
              f"novelty={report[method]['beyond_accuracy']['novelty_mean']:.3f}  coverage={cov:.4f}")
        print(f"[{dataset}] {method} cold-start AUC={cold['auc']['mean']:.4f} (n={report['n_cold']}) "
              f"vs warm AUC={warm['auc']['mean']:.4f} (n={report['n_warm']})")

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{dataset}] saved outputs/{dataset}/eval_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Q4: offline evaluation harness.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    targets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        run(name)


if __name__ == "__main__":
    main()
