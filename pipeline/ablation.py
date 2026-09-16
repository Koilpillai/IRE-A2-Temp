"""A2 Q3 -- Baseline reproduced, then beaten: ablation + paired bootstrap CI.

Q3.1 asks to reproduce "the official/starter baseline (e.g., NRMS from the ebnerd-
benchmark repo, or the MIND baseline)". NRMS is a neural ranker built on `torch`;
pipeline/rerank.py's docstring already explains why re-deriving NRMS's user/news
encoders from scratch is out of scope here even though `torch` itself is now
installed (see the design note for the full reasoning). The baseline reproduced here
instead is a deliberately minimal, non-personalized GBDT -- popularity, freshness, and
shown position only, none of Q1's click-history/session engineering -- trained with
the exact same algorithm, hyperparameters, training data and random seed as Q2's
reranker. That isolates the ablation to exactly one variable (the feature set), which
is what Q3.3 actually asks the ablation to isolate.

"Improved" = Q2's already-trained full-feature reranker (pipeline/rerank.py), reused
here rather than retrained, for the same one-variable-at-a-time reason.

Statistical significance (Q3.4): for each metric, this computes the PAIRED per-
impression delta (full's score minus baseline's score, same impression, same labels)
and bootstraps that delta array directly -- a paired bootstrap, not two independent
ones -- then checks whether the resulting 95% CI excludes zero.

Usage:
    python -m pipeline.ablation --dataset mind
    python -m pipeline.ablation --dataset ebnerd

Requires pipeline/rerank.py to have already been run for the same dataset (it needs
feature_store/<dataset>/reranker_model.joblib).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from pipeline import config
from pipeline.common.metrics import bootstrap_ci, mrr_score, ndcg_at_k
from pipeline.common.reranker import FEATURE_COLS_MINIMAL, load_features, to_xy, train_gbdt

METRICS = [
    ("auc", lambda y, s: roc_auc_score(y, s), True),
    ("mrr", mrr_score, False),
    ("ndcg5", lambda y, s: ndcg_at_k(y, s, 5), False),
    ("ndcg10", lambda y, s: ndcg_at_k(y, s, 10), False),
]


def run(dataset: str) -> dict:
    fs = config.FEATURE_STORE / dataset
    out_dir = config.OUTPUTS / dataset
    model_path = fs / "reranker_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found -- run `python -m pipeline.rerank --dataset {dataset}` first")

    train_df = load_features(fs, "train")
    val_df = load_features(fs, "val").copy()

    X_train_min, y_train = to_xy(train_df, FEATURE_COLS_MINIMAL)
    X_val_min, _ = to_xy(val_df, FEATURE_COLS_MINIMAL)
    t0 = time.time()
    baseline_model = train_gbdt(X_train_min, y_train, seed=config.RANDOM_SEED)
    print(f"[{dataset}] baseline (minimal-feature: {FEATURE_COLS_MINIMAL}) GBDT trained in {time.time()-t0:.1f}s")
    val_df["baseline_score"] = baseline_model.predict_proba(X_val_min)[:, 1]

    bundle = joblib.load(model_path)
    full_model, full_feat_cols = bundle["model"], bundle["feature_cols"]
    X_val_full, _ = to_xy(val_df, full_feat_cols)
    val_df["full_score"] = full_model.predict_proba(X_val_full)[:, 1]

    # One shared grouping for both score columns -- guarantees the two lists are
    # aligned impression-for-impression (labels are identical either way, so the
    # per-metric validity filter below drops the same impressions from both).
    ordered = val_df.sort_values(["impression_id", "candidate_position"])
    grouped = ordered.groupby("impression_id", sort=False)
    labels_g = grouped["label"].apply(lambda s: s.to_numpy()).tolist()
    baseline_g = grouped["baseline_score"].apply(lambda s: s.to_numpy()).tolist()
    full_g = grouped["full_score"].apply(lambda s: s.to_numpy()).tolist()

    report = {"dataset": dataset, "baseline_features": FEATURE_COLS_MINIMAL, "full_features": full_feat_cols}
    for metric_name, metric_fn, needs_both_classes in METRICS:
        baseline_vals, full_vals = [], []
        for y_true, b_s, f_s in zip(labels_g, baseline_g, full_g):
            y_true = np.asarray(y_true)
            valid = (0 < y_true.sum() < len(y_true)) if needs_both_classes else (y_true.sum() > 0)
            if not valid:
                continue
            baseline_vals.append(metric_fn(y_true, np.asarray(b_s, dtype=np.float64)))
            full_vals.append(metric_fn(y_true, np.asarray(f_s, dtype=np.float64)))
        baseline_vals = np.array(baseline_vals)
        full_vals = np.array(full_vals)
        delta = full_vals - baseline_vals  # paired: same impressions, same order

        b_mean, b_lo, b_hi = bootstrap_ci(baseline_vals, seed=config.RANDOM_SEED)
        f_mean, f_lo, f_hi = bootstrap_ci(full_vals, seed=config.RANDOM_SEED)
        d_mean, d_lo, d_hi = bootstrap_ci(delta, seed=config.RANDOM_SEED)
        excludes_zero = bool(d_lo > 0 or d_hi < 0)

        report[metric_name] = {
            "baseline": {"mean": b_mean, "ci_low": b_lo, "ci_high": b_hi, "n": int(len(baseline_vals))},
            "full": {"mean": f_mean, "ci_low": f_lo, "ci_high": f_hi, "n": int(len(full_vals))},
            "paired_delta": {"mean": d_mean, "ci_low": d_lo, "ci_high": d_hi, "excludes_zero": excludes_zero},
        }
        print(f"[{dataset}] {metric_name:7s} baseline={b_mean:.4f}  full={f_mean:.4f}  "
              f"delta={d_mean:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]  significant={excludes_zero}")

    with open(out_dir / "ablation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{dataset}] saved outputs/{dataset}/ablation.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="A2 Q3: baseline-vs-improved ablation with paired bootstrap CI.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    for ds in (["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]):
        run(ds)


if __name__ == "__main__":
    main()
