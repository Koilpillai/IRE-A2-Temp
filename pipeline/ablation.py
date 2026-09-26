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

Q3.3's "ablation study isolating the contribution of your improvement" is two things
here, not one: the baseline-vs-full comparison above (does the FULL feature set beat a
non-personalized baseline), AND a per-feature-group INCREMENTAL ablation below (which
specific feature group -- click-history, session, or position -- actually drives that
gain, added one at a time on top of the baseline). A single "minimal vs everything at
once" comparison can't tell those apart; the incremental version can.

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
from pipeline.common.reranker import FEATURE_COLS_MINIMAL, feature_columns, load_features, to_xy, train_gbdt

METRICS = [
    ("auc", lambda y, s: roc_auc_score(y, s), True),
    ("mrr", mrr_score, False),
    ("ndcg5", lambda y, s: ndcg_at_k(y, s, 5), False),
    ("ndcg10", lambda y, s: ndcg_at_k(y, s, 10), False),
]

# Incremental feature groups added on top of FEATURE_COLS_MINIMAL, one at a time, to
# isolate which group actually drives the gain over the baseline (Q3.3). Built from
# pipeline/features.py's own four sub-requirement groupings (module docstring
# points 1-4); position features are already in FEATURE_COLS_MINIMAL, so "position" is
# the starting group, not an addition.
GROUP_CLICK_HISTORY = ["hist_len", "hist_category_match_frac", "hist_category_match_recency", "hist_embed_sim"]
GROUP_SESSION = ["session_position"]  # user_avg_read_time/scroll_pct handled per-dataset below
GROUP_ARTICLE_CONTEXT = ["category_match"]  # popularity/freshness/position already in the baseline


def _incremental_feature_sets(dataset: str, available_cols: list[str]) -> list[tuple[str, list[str]]]:
    """(stage_name, feature_columns) pairs, each stage = baseline + all groups up to
    and including this one. Only includes columns that actually exist for this
    dataset (EB-NeRD's two dwell-time columns don't exist for MIND)."""
    session_cols = list(GROUP_SESSION)
    if dataset == "ebnerd":
        session_cols += [c for c in ("user_avg_read_time", "user_avg_scroll_pct") if c in available_cols]

    stages = [("baseline_position_only", list(FEATURE_COLS_MINIMAL))]
    running = list(FEATURE_COLS_MINIMAL)
    for stage_name, group in [
        ("+click_history", GROUP_CLICK_HISTORY),
        ("+session", session_cols),
        ("+article_context", GROUP_ARTICLE_CONTEXT),
    ]:
        running = running + [c for c in group if c in available_cols and c not in running]
        stages.append((stage_name, list(running)))
    return stages


def _grouped_scores(val_df, score_col: str):
    ordered = val_df.sort_values(["impression_id", "candidate_position"])
    grouped = ordered.groupby("impression_id", sort=False)
    labels_g = grouped["label"].apply(lambda s: s.to_numpy()).tolist()
    scores_g = grouped[score_col].apply(lambda s: s.to_numpy()).tolist()
    return labels_g, scores_g


def run_incremental(dataset: str, train_df, val_df) -> dict:
    """Q3.3 per-feature-group ablation: baseline -> +click-history -> +session ->
    +article-context (== full model), reporting AUC at each stage so the report can
    say which group contributes what, rather than only "baseline vs. everything"."""
    available = feature_columns(train_df)
    stages = _incremental_feature_sets(dataset, available)

    stage_report = {}
    prev_auc = None
    for stage_name, cols in stages:
        X_train, y_train = to_xy(train_df, cols)
        X_val, _ = to_xy(val_df, cols)
        model = train_gbdt(X_train, y_train, seed=config.RANDOM_SEED)
        val_df[f"_stage_{stage_name}"] = model.predict_proba(X_val)[:, 1]
        labels_g, scores_g = _grouped_scores(val_df, f"_stage_{stage_name}")

        auc_vals = np.array([
            roc_auc_score(y, s) for y, s in zip(labels_g, scores_g)
            if 0 < np.asarray(y).sum() < len(y)
        ])
        mean, lo, hi = bootstrap_ci(auc_vals, seed=config.RANDOM_SEED)
        delta_from_prev = None if prev_auc is None else mean - prev_auc
        stage_report[stage_name] = {
            "features": cols, "auc_mean": mean, "auc_ci_low": lo, "auc_ci_high": hi,
            "auc_delta_from_previous_stage": delta_from_prev,
        }
        print(f"[{dataset}] ablation stage {stage_name:22s} n_feat={len(cols):2d}  "
              f"AUC={mean:.4f} [{lo:.4f},{hi:.4f}]"
              + (f"  (+{delta_from_prev:+.4f} vs prev stage)" if delta_from_prev is not None else ""))
        prev_auc = mean
        val_df.drop(columns=[f"_stage_{stage_name}"], inplace=True)
    return stage_report


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

    print(f"[{dataset}] -- Q3.3 per-feature-group incremental ablation --")
    report["incremental_feature_group_ablation"] = run_incremental(dataset, train_df, val_df)

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
