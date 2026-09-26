"""A2 Q9 -- Anti-gaming: metrics with and without features unavailable at serving time.

Q9 asks explicitly: "Report metrics with and without features unavailable at serving
time." The feature most at risk of quietly encoding the label rather than genuine
signal is `candidate_position` / `candidate_position_norm` (see pipeline/features.py's
module docstring, "Candidate generation"): every row's candidate pool is the union of
A1's own BM25+embedding full-corpus retrieval, and since that retrieval misses the true
click ~98%+ of the time (outputs/<dataset>/recall_at_k.json), the true positive is
force-inserted at a UNIFORMLY RANDOM position among the genuinely-retrieved candidates
whenever retrieval misses it. That randomization is exactly the anti-gaming fix for a
previously-caught leak (a fixed insertion point made `candidate_position` an almost
perfect giveaway of the label -- the tell was an implausible ~0.978 val AUC). This
module re-validates that fix quantitatively rather than trusting the randomization by
argument alone: it retrains the reranker with `candidate_position`/
`candidate_position_norm` excluded and reports the SAME metrics both ways. If the
randomization is doing its job, the position-excluded model should perform close to
(not dramatically worse than) the full model -- a large gap would mean the position
features still carry serving-time-unavailable signal despite the randomization.

This is a genuine "unavailable at serving time" comparison because, unlike the fused
BM25+embedding retrieval SCORE (which is available before any click, and is what
`candidate_position` is a rank of -- see rerank.py), the row's FINAL position after
the random positive-insertion step depends on whether the label happened to require an
insertion at all -- information that is a function of the label itself, not of
anything a serving-time system would know before scoring.

Usage:
    python -m pipeline.anti_gaming --dataset mind
    python -m pipeline.anti_gaming --dataset ebnerd

Requires feature_store/<dataset>/behavioral_features_{train,val}.parquet to already
exist (run `python -m pipeline.features --dataset <dataset>` first).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pipeline import config
from pipeline.common.metrics import evaluate_impressions, summarize
from pipeline.common.reranker import feature_columns, group_by_impression, load_features, to_xy, train_gbdt

POSITION_COLUMNS = {"candidate_position", "candidate_position_norm"}


def run(dataset: str) -> dict:
    fs = config.FEATURE_STORE / dataset
    out_dir = config.OUTPUTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_features(fs, "train")
    val_df = load_features(fs, "val").copy()

    full_cols = feature_columns(train_df)
    no_position_cols = [c for c in full_cols if c not in POSITION_COLUMNS]
    if len(no_position_cols) == len(full_cols):
        raise SystemExit(f"{dataset}: expected {POSITION_COLUMNS} in {full_cols}")

    X_train_full, y_train = to_xy(train_df, full_cols)
    X_val_full, _ = to_xy(val_df, full_cols)
    t0 = time.time()
    full_model = train_gbdt(X_train_full, y_train, seed=config.RANDOM_SEED)
    print(f"[{dataset}] WITH position features ({full_cols}) trained in {time.time()-t0:.1f}s")
    val_df["with_position_score"] = full_model.predict_proba(X_val_full)[:, 1]

    X_train_np, _ = to_xy(train_df, no_position_cols)
    X_val_np, _ = to_xy(val_df, no_position_cols)
    t0 = time.time()
    no_position_model = train_gbdt(X_train_np, y_train, seed=config.RANDOM_SEED)
    print(f"[{dataset}] WITHOUT position features ({no_position_cols}) trained in {time.time()-t0:.1f}s")
    val_df["without_position_score"] = no_position_model.predict_proba(X_val_np)[:, 1]

    with_labels, with_scores = group_by_impression(val_df, "with_position_score")
    without_labels, without_scores = group_by_impression(val_df, "without_position_score")

    report = {
        "dataset": dataset,
        "full_features": full_cols,
        "features_excluded_as_serving_unavailable": sorted(POSITION_COLUMNS),
        "no_position_features": no_position_cols,
    }
    for name, labels, scores in [
        ("with_position_features", with_labels, with_scores),
        ("without_position_features", without_labels, without_scores),
    ]:
        summary = summarize(evaluate_impressions(labels, scores), n_boot=1000, seed=config.RANDOM_SEED)
        report[name] = summary
        print(f"[{dataset}] {name:28s}  AUC={summary['auc']['mean']:.4f}  "
              f"MRR={summary['mrr']['mean']:.4f}  nDCG@5={summary['ndcg5']['mean']:.4f}  "
              f"nDCG@10={summary['ndcg10']['mean']:.4f}")

    auc_gap = report["with_position_features"]["auc"]["mean"] - report["without_position_features"]["auc"]["mean"]
    report["auc_gap_with_minus_without"] = auc_gap
    # A large gap here would mean the position features still carry serving-time-
    # unavailable (label-derived) signal despite the random-insertion anti-leak fix --
    # this threshold is a diagnostic flag for the report, not a hard pass/fail gate.
    report["large_gap_flag"] = bool(abs(auc_gap) > 0.05)
    print(f"[{dataset}] AUC gap (with - without position features) = {auc_gap:+.4f}"
          f"{'  [FLAG: large gap -- check for residual leakage]' if report['large_gap_flag'] else ''}")

    with open(out_dir / "anti_gaming.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{dataset}] saved outputs/{dataset}/anti_gaming.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="A2 Q9: metrics with/without serving-unavailable features.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    for ds in (["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]):
        run(ds)


if __name__ == "__main__":
    main()
