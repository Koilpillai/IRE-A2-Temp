"""A2 Q2 -- Two-stage retrieve-then-rank: GBDT re-ranker over Q1's behavioural features.

Stage 1 (retrieve) is done inside pipeline/features.py's build_features, per Q2.1
("Use Assignment 1's candidate generator to retrieve top-K candidates (K ~ 100-200)"):
each row's candidate pool is the union of A1's own BM25 and Word2Vec+ANN top-
config.RETRIEVAL_CANDIDATE_K retrieval over the full article corpus, not the dataset's
raw officially-provided candidate list (see pipeline/features.py's module docstring
for the query construction and the positive-label-preservation policy). Stage 2
(rank) is this module: a pointwise GBDT trained on Q1's engineered behavioural
features, predicting P(click) per (impression, candidate).

Option A (GBDT) vs Option B (a neural ranker, e.g. NRMS) per Q2.2: Option B needs
`torch`, and A1's README already documents that `torch` couldn't be installed in the
original dev sandbox. It's installable here (see requirements.txt / the design note),
but re-deriving NRMS's user/news encoders from scratch is a much larger undertaking
than this assignment's Q2 scope calls for, so GBDT (scikit-learn's
HistGradientBoostingClassifier -- same algorithm family as LightGBM/XGBoost) is the
one actually implemented; see pipeline/common/reranker.py.

"Before" reranking (Q2.4) = A1's own fused BM25+embedding retrieval score on the
EXACT SAME candidate rows the GBDT scores -- since `candidate_position` is already
each candidate's rank under that fused retrieval score (see pipeline/features.py),
`-candidate_position` is an exactly order-preserving stand-in for it, so AUC/MRR/
nDCG computed from it are identical to computing them from the raw fused score. This
keeps "before" and "after" on the identical candidate universe, unlike a naive reuse
of outputs/<dataset>/retrieval_results_val.parquet, which scores the dataset's
officially-provided candidate list -- a different (and differently-sized) row set now
that Q2.1's candidate generation is real retrieval, not that list. That file's
BM25-only/embedding-only numbers are still reported separately below as an additional
reference point (the "how good is retrieval alone over the platform's own curated
list" question retrieval_eval.py already answers), just not treated as the Q2.4
before/after pair.

Usage:
    python -m pipeline.rerank --dataset mind
    python -m pipeline.rerank --dataset ebnerd
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.common.metrics import evaluate_impressions, summarize
from pipeline.common.reranker import feature_columns, group_by_impression, load_features, to_xy, train_gbdt


def run(dataset: str) -> dict:
    fs = config.FEATURE_STORE / dataset
    out_dir = config.OUTPUTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_features(fs, "train")
    val_df = load_features(fs, "val")
    feat_cols = feature_columns(train_df)
    print(f"[{dataset}] train rows={len(train_df):,}  val rows={len(val_df):,}  "
          f"features={len(feat_cols)} ({feat_cols})")

    X_train, y_train = to_xy(train_df, feat_cols)
    X_val, y_val = to_xy(val_df, feat_cols)

    t0 = time.time()
    model = train_gbdt(X_train, y_train, seed=config.RANDOM_SEED)
    print(f"[{dataset}] GBDT trained in {time.time()-t0:.1f}s "
          f"(n_iter={model.n_iter_}, train clickthrough rate={y_train.mean():.4f})")

    val_df = val_df.copy()
    val_df["gbdt_score"] = model.predict_proba(X_val)[:, 1]
    # -candidate_position: candidate_position is this row's rank under A1's own fused
    # BM25+embedding retrieval score (see pipeline/features.py), so its negation is an
    # exactly order-preserving stand-in for that score -- on the SAME candidate rows
    # gbdt_score was just computed over.
    val_df["retrieval_score"] = -val_df["candidate_position"]

    model_path = fs / "reranker_model.joblib"
    joblib.dump({"model": model, "feature_cols": feat_cols}, model_path)

    preds_path = out_dir / "rerank_val_predictions.parquet"
    val_df[["impression_id", "candidate_id", "candidate_position", "label", "gbdt_score"]].to_parquet(
        preds_path, index=False)

    # Primary Q2.4 "before" vs "after": identical candidate rows, only the score differs.
    retrieval_labels, retrieval_scores = group_by_impression(val_df, "retrieval_score")
    gbdt_labels, gbdt_scores = group_by_impression(val_df, "gbdt_score")

    # Supplementary reference point: A1's BM25-only/embedding-only scores over the
    # dataset's officially-provided candidate list (a different, smaller row set --
    # see the module docstring), not part of the apples-to-apples before/after pair.
    retrieval_official = pd.read_parquet(out_dir / "retrieval_results_val.parquet")
    official_labels_all = [np.asarray(l) for l in retrieval_official["labels"]]
    bm25_scores_all = [np.asarray(s) for s in retrieval_official["bm25_scores"]]
    embed_scores_all = [np.asarray(s) for s in retrieval_official["embed_scores"]]

    report = {"dataset": dataset, "n_train_rows": len(train_df), "n_val_rows": len(val_df),
              "feature_columns": feat_cols, "gbdt_n_iter": int(model.n_iter_)}
    for name, labels, scores in [
        ("retrieval_before", retrieval_labels, retrieval_scores),
        ("gbdt_after", gbdt_labels, gbdt_scores),
        ("bm25_official_candidates_reference", official_labels_all, bm25_scores_all),
        ("embedding_official_candidates_reference", official_labels_all, embed_scores_all),
    ]:
        summary = summarize(evaluate_impressions(labels, scores), n_boot=1000, seed=config.RANDOM_SEED)
        report[name] = summary
        print(f"[{dataset}] {name:40s}  AUC={summary['auc']['mean']:.4f}  "
              f"MRR={summary['mrr']['mean']:.4f}  nDCG@5={summary['ndcg5']['mean']:.4f}  "
              f"nDCG@10={summary['ndcg10']['mean']:.4f}")

    with open(out_dir / "rerank_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{dataset}] saved {model_path}, {preds_path}, and rerank_metrics.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="A2 Q2: train + evaluate the GBDT re-ranker.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    for ds in (["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]):
        run(ds)


if __name__ == "__main__":
    main()
