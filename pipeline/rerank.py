"""A2 Q2 -- Two-stage retrieve-then-rank: GBDT re-ranker over Q1's behavioural features.

Stage 1 (retrieve) is already done -- A1's own offline-eval protocol for MIND/EB-NeRD
scores each impression's OWN officially-provided candidate list (Q2.4/Q4's AUC/MRR/nDCG
convention), which for a bounded 5-300 candidate pool per impression already plays the
role of "retrieve top-K candidates" (the K~100-200 the assignment names is well within
these lists' size range). Stage 2 (rank) is this module: a pointwise GBDT trained on
Q1's engineered behavioural features, predicting P(click) per (impression, candidate).

Option A (GBDT) vs Option B (a neural ranker, e.g. NRMS) per Q2.2: Option B needs
`torch`, and A1's README already documents that `torch` couldn't be installed in the
original dev sandbox. It's installable here (see requirements.txt / the design note),
but re-deriving NRMS's user/news encoders from scratch is a much larger undertaking
than this assignment's Q2 scope calls for, so GBDT (scikit-learn's
HistGradientBoostingClassifier -- same algorithm family as LightGBM/XGBoost) is the
one actually implemented; see pipeline/common/reranker.py.

"Before" reranking = A1's own BM25 and embedding scores on the same candidates
(outputs/<dataset>/retrieval_results_val.parquet, already computed). "After" = this
module's GBDT score on the same candidates. All three get the identical AUC/MRR/
nDCG@5/nDCG@10 treatment via pipeline.common.metrics, so the comparison is apples to
apples (Q2.4).

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

    model_path = fs / "reranker_model.joblib"
    joblib.dump({"model": model, "feature_cols": feat_cols}, model_path)

    preds_path = out_dir / "rerank_val_predictions.parquet"
    val_df[["impression_id", "candidate_id", "candidate_position", "label", "gbdt_score"]].to_parquet(
        preds_path, index=False)

    # "before": A1's own BM25/embedding scores on the exact same candidates.
    retrieval = pd.read_parquet(out_dir / "retrieval_results_val.parquet")
    labels_all = [np.asarray(l) for l in retrieval["labels"]]
    bm25_scores_all = [np.asarray(s) for s in retrieval["bm25_scores"]]
    embed_scores_all = [np.asarray(s) for s in retrieval["embed_scores"]]

    # "after": this module's GBDT, re-grouped back into per-impression lists.
    gbdt_labels, gbdt_scores = group_by_impression(val_df, "gbdt_score")

    report = {"dataset": dataset, "n_train_rows": len(train_df), "n_val_rows": len(val_df),
              "feature_columns": feat_cols, "gbdt_n_iter": int(model.n_iter_)}
    for name, labels, scores in [
        ("bm25_before", labels_all, bm25_scores_all),
        ("embedding_before", labels_all, embed_scores_all),
        ("gbdt_after", gbdt_labels, gbdt_scores),
    ]:
        summary = summarize(evaluate_impressions(labels, scores), n_boot=1000, seed=config.RANDOM_SEED)
        report[name] = summary
        print(f"[{dataset}] {name:17s}  AUC={summary['auc']['mean']:.4f}  "
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
