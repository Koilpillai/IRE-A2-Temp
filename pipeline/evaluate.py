"""Q4/Q5 -- Offline evaluation harness for the FULL two-stage pipeline.

Two independent evaluations live here, over two different row sets, because A2's Q2
fix (pipeline/features.py) changed the reranker's candidate pool to A1's own
retrieval-fused set rather than the dataset's officially-provided candidate list --
the two are no longer the same rows:

  1. BM25 / embedding (retrieval-only): consumes outputs/<dataset>/retrieval_results_val.parquet
     -- each impression's OFFICIAL candidate list, the only place A1's own BM25/
     embedding scores were computed against. AUC/MRR/nDCG@5/nDCG@10, slicing
     (cold-start/warm, head/tail), beyond-accuracy (diversity/novelty/coverage on
     each method's own top-10 full-corpus retrieval), all bootstrap 95% CI.
  2. GBDT re-ranker (the actual "full two-stage pipeline" Q5 asks for): consumes
     outputs/<dataset>/rerank_val_predictions.parquet joined against
     feature_store/<dataset>/behavioral_features_val.parquet for slicing columns --
     the GBDT's own retrieval-fused candidate rows, not the official list. Same
     accuracy metrics, same two slice dimensions, plus bootstrap CI on the
     beyond-accuracy metrics too (diversity/novelty/coverage over the GBDT's own
     top-10 per impression, not reused from BM25/embedding's full-corpus retrieval,
     since the GBDT never does full-corpus retrieval itself -- see module note below).

No index is rebuilt here for the BM25/embedding path; the GBDT path reuses the
already-trained/persisted model and feature tables from pipeline/rerank.py.

Usage:
    python -m pipeline.evaluate --dataset mind
    python -m pipeline.evaluate --dataset ebnerd

Requires pipeline/rerank.py to have already been run (needs rerank_val_predictions.parquet
and behavioral_features_val.parquet) for the GBDT section; falls back to reporting only
the BM25/embedding section with a warning if those aren't present yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.common.embeddings import EmbeddingIndex
from pipeline.common.metrics import coverage, evaluate_impressions, intra_list_diversity, novelty, summarize
from pipeline.common.popularity import train_popularity as _train_popularity

COLD_START_THRESHOLD = 5   # history_len <= this => cold-start (assignment's own example threshold)
HEAD_FRACTION = 0.2        # top 20% most-clicked (in TRAIN) articles = "head"


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


def _bootstrap_scalar_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for a plain per-impression scalar array (diversity/novelty) --
    same percentile-of-resampled-means method as pipeline.common.metrics.bootstrap_ci,
    reused here directly for the beyond-accuracy metrics Q5 also asks for a CI on."""
    from pipeline.common.metrics import bootstrap_ci
    mean, lo, hi = bootstrap_ci(values[~np.isnan(values)], n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int((~np.isnan(values)).sum())}


def _bootstrap_coverage_ci(top_n_id_lists: list[list], catalog_size: int, n_boot: int = 1000, seed: int = 42) -> dict:
    """Coverage is a single set-union statistic over ALL impressions, not a per-
    impression mean, so its bootstrap resamples IMPRESSIONS (with replacement) and
    recomputes the union-coverage of each resampled set of lists."""
    rng = np.random.default_rng(seed)
    n = len(top_n_id_lists)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    point = coverage(top_n_id_lists, catalog_size)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = coverage([top_n_id_lists[j] for j in idx], catalog_size)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def _gbdt_evaluation(dataset: str, fs: Path, out_dir: Path, popularity: dict, total_clicks: int,
                      catalog_size: int, emb_index: EmbeddingIndex) -> dict | None:
    """The GBDT's own retrieval-fused candidate rows (pipeline/rerank.py's output),
    NOT the official candidate list -- Q2's fix changed the candidate pool, so this
    section evaluates the actual "full two-stage pipeline" Q5 asks for, on its own
    correct row set, rather than reusing the BM25/embedding masks/lists above (which
    are keyed to a different, smaller row set)."""
    preds_path = out_dir / "rerank_val_predictions.parquet"
    feat_path = fs / "behavioral_features_val.parquet"
    if not preds_path.exists() or not feat_path.exists():
        print(f"[{dataset}] skipping GBDT section: {preds_path if not preds_path.exists() else feat_path} "
              f"not found -- run `python -m pipeline.rerank --dataset {dataset}` first")
        return None

    preds = pd.read_parquet(preds_path)
    feats = pd.read_parquet(feat_path, columns=["impression_id", "candidate_id", "hist_len"])
    # hist_len is constant per impression (computed once per impression, not per
    # candidate) -- one row per impression is enough for the cold/warm mask.
    hist_len_by_imp = feats.drop_duplicates("impression_id").set_index("impression_id")["hist_len"]

    ordered = preds.sort_values(["impression_id", "candidate_position"])
    grouped = ordered.groupby("impression_id", sort=False)
    imp_ids = list(grouped.groups.keys())
    labels_all = grouped["label"].apply(lambda s: s.to_numpy()).tolist()
    scores_all = grouped["gbdt_score"].apply(lambda s: s.to_numpy()).tolist()
    cand_ids_all = grouped["candidate_id"].apply(lambda s: s.to_numpy()).tolist()

    hist_len_arr = hist_len_by_imp.reindex(imp_ids).fillna(0).to_numpy()
    warm_mask = hist_len_arr > COLD_START_THRESHOLD
    cold_mask = ~warm_mask

    clicked_ids = [[c for c, l in zip(cands, labs) if l == 1]
                   for cands, labs in zip(cand_ids_all, labels_all)]
    head_mask = _head_tail_mask(clicked_ids, popularity, HEAD_FRACTION)
    tail_mask = ~head_mask

    overall = summarize(evaluate_impressions(labels_all, scores_all), n_boot=1000, seed=config.RANDOM_SEED)
    cold = _metrics_for_mask(labels_all, scores_all, cold_mask)
    warm = _metrics_for_mask(labels_all, scores_all, warm_mask)
    head = _metrics_for_mask(labels_all, scores_all, head_mask)
    tail = _metrics_for_mask(labels_all, scores_all, tail_mask)

    # Beyond-accuracy: the GBDT's own top-10 PER IMPRESSION (it only ever ranks the
    # candidates it was given, never does full-corpus retrieval itself -- unlike
    # BM25/embedding's top10_col above, which comes from an actual full-corpus scan).
    top10_lists = []
    for cands, scores in zip(cand_ids_all, scores_all):
        order = np.argsort(-scores)[:10]
        top10_lists.append([cands[i] for i in order])

    div = intra_list_diversity(top10_lists, emb_index)
    nov = novelty(top10_lists, popularity, total_clicks, catalog_size)

    return {
        "overall": overall,
        "slices": {"cold_start": cold, "warm": warm, "head_articles": head, "tail_articles": tail},
        "n_cold": int(cold_mask.sum()), "n_warm": int(warm_mask.sum()),
        "n_head_click": int(head_mask.sum()), "n_tail_click": int(tail_mask.sum()),
        "beyond_accuracy": {
            "intra_list_diversity": _bootstrap_scalar_ci(div, seed=config.RANDOM_SEED),
            "novelty": _bootstrap_scalar_ci(nov, seed=config.RANDOM_SEED),
            "coverage": _bootstrap_coverage_ci(top10_lists, catalog_size, seed=config.RANDOM_SEED),
        },
        "note": "candidate pool is A1's retrieval-fused set (pipeline/features.py), "
                "not the dataset's officially-provided candidate list -- not directly "
                "row-for-row comparable to the bm25/embedding sections above.",
    }


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

        report[method] = {
            "overall": overall,
            "slices": {"cold_start": cold, "warm": warm, "head_articles": head, "tail_articles": tail},
            "beyond_accuracy": {
                "intra_list_diversity": _bootstrap_scalar_ci(div, seed=config.RANDOM_SEED),
                "novelty": _bootstrap_scalar_ci(nov, seed=config.RANDOM_SEED),
                "coverage": _bootstrap_coverage_ci(top10_lists, catalog_size, seed=config.RANDOM_SEED),
            },
        }
        cov_mean = report[method]["beyond_accuracy"]["coverage"]["mean"]
        print(f"[{dataset}] {method}: AUC={overall['auc']['mean']:.4f} "
              f"[{overall['auc']['ci_low']:.4f},{overall['auc']['ci_high']:.4f}]  "
              f"MRR={overall['mrr']['mean']:.4f}  nDCG@5={overall['ndcg5']['mean']:.4f}  "
              f"nDCG@10={overall['ndcg10']['mean']:.4f}  "
              f"diversity={report[method]['beyond_accuracy']['intra_list_diversity']['mean']:.3f}  "
              f"novelty={report[method]['beyond_accuracy']['novelty']['mean']:.3f}  coverage={cov_mean:.4f}")
        print(f"[{dataset}] {method} cold-start AUC={cold['auc']['mean']:.4f} (n={report['n_cold']}) "
              f"vs warm AUC={warm['auc']['mean']:.4f} (n={report['n_warm']})")

    gbdt_report = _gbdt_evaluation(dataset, fs, out_dir, popularity, total_clicks, catalog_size, emb_index)
    if gbdt_report is not None:
        report["gbdt"] = gbdt_report
        ov = gbdt_report["overall"]
        ba = gbdt_report["beyond_accuracy"]
        print(f"[{dataset}] gbdt (full two-stage pipeline): AUC={ov['auc']['mean']:.4f} "
              f"[{ov['auc']['ci_low']:.4f},{ov['auc']['ci_high']:.4f}]  MRR={ov['mrr']['mean']:.4f}  "
              f"nDCG@5={ov['ndcg5']['mean']:.4f}  nDCG@10={ov['ndcg10']['mean']:.4f}  "
              f"diversity={ba['intra_list_diversity']['mean']:.3f}  novelty={ba['novelty']['mean']:.3f}  "
              f"coverage={ba['coverage']['mean']:.4f}")

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{dataset}] saved outputs/{dataset}/eval_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Q4/Q5: offline evaluation harness (retrieval + full two-stage pipeline).")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    targets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        run(name)


if __name__ == "__main__":
    main()
