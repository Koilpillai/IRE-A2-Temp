"""Q1 -- Reproducible data pipeline: raw files -> unified schema -> feature store.

One-command rebuild:
    python pipeline/build_pipeline.py --dataset mind
    python pipeline/build_pipeline.py --dataset ebnerd
    python pipeline/build_pipeline.py --dataset all

Only the LABELED splits (train/val) are materialized into the feature store -- they're
small and reused across Q2/Q3/Q4. The large unlabeled Codabench test sets (2.37M / 13.5M
rows, no ground truth) are streamed directly from raw files at Q5 submission time instead
of being duplicated into the store (see generate_submission.py).
"""
from __future__ import annotations

import argparse
import json
import time

from pipeline import config
from pipeline.adapters import ebnerd as ebnerd_adapter
from pipeline.adapters import mind as mind_adapter
from pipeline.common.split import carve_internal_validation


def build_mind() -> dict:
    out_dir = config.FEATURE_STORE / "mind"
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = mind_adapter.load_articles(
        config.MIND_TRAIN_DIR / "news.tsv",
        config.MIND_DEV_DIR / "news.tsv",
        config.MIND_TEST_DIR / "news.tsv",
    )
    articles.to_parquet(out_dir / "articles.parquet", index=False)

    train = mind_adapter.load_behaviors(config.MIND_TRAIN_DIR / "behaviors.tsv", has_labels=True)
    val = mind_adapter.load_behaviors(config.MIND_DEV_DIR / "behaviors.tsv", has_labels=True)

    # History is constant per user within a split (verified empirically -- see design note),
    # so pull it into its own small per-user table instead of repeating it on every impression
    # row (fit/tune reuse train's table, since they're time-ordered subsets of the same users).
    mind_adapter.extract_user_history(train).to_parquet(out_dir / "user_history_train.parquet", index=False)
    mind_adapter.extract_user_history(val).to_parquet(out_dir / "user_history_val.parquet", index=False)
    train = train.drop(columns=["history"])
    val = val.drop(columns=["history"])

    fit, tune = carve_internal_validation(train, "timestamp", config.INTERNAL_TUNE_HOLDOUT_FRACTION)

    train.to_parquet(out_dir / "behaviors_train.parquet", index=False)
    fit.to_parquet(out_dir / "behaviors_train_fit.parquet", index=False)
    tune.to_parquet(out_dir / "behaviors_train_tune.parquet", index=False)
    val.to_parquet(out_dir / "behaviors_val.parquet", index=False)

    return {
        "dataset": "mind",
        "n_articles": len(articles),
        "n_train": len(train), "n_train_fit": len(fit), "n_train_tune": len(tune),
        "n_val": len(val),
        "train_time_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "internal_tune_cutoff": str(fit["timestamp"].max()),
        "val_time_range": [str(val["timestamp"].min()), str(val["timestamp"].max())],
    }


def build_ebnerd() -> dict:
    out_dir = config.FEATURE_STORE / "ebnerd"
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = ebnerd_adapter.load_articles(
        config.EBNERD_SMALL_DIR / "articles.parquet",
        config.EBNERD_TESTSET_DIR / "articles.parquet",
    )
    articles.to_parquet(out_dir / "articles.parquet", index=False)

    train = ebnerd_adapter.load_behaviors(
        config.EBNERD_SMALL_DIR / "train" / "behaviors.parquet",
        config.EBNERD_SMALL_DIR / "train" / "history.parquet", has_labels=True,
    )
    val = ebnerd_adapter.load_behaviors(
        config.EBNERD_SMALL_DIR / "validation" / "behaviors.parquet",
        config.EBNERD_SMALL_DIR / "validation" / "history.parquet", has_labels=True,
    )
    # EB-NeRD ships history as its own per-split, per-user file already -- just carry it
    # into the feature store as-is (fit/tune reuse train's table; see build_mind for why).
    ebnerd_adapter.load_user_history(config.EBNERD_SMALL_DIR / "train" / "history.parquet") \
        .to_parquet(out_dir / "user_history_train.parquet", index=False)
    ebnerd_adapter.load_user_history(config.EBNERD_SMALL_DIR / "validation" / "history.parquet") \
        .to_parquet(out_dir / "user_history_val.parquet", index=False)

    fit, tune = carve_internal_validation(train, "timestamp", config.INTERNAL_TUNE_HOLDOUT_FRACTION)

    train.to_parquet(out_dir / "behaviors_train.parquet", index=False)
    fit.to_parquet(out_dir / "behaviors_train_fit.parquet", index=False)
    tune.to_parquet(out_dir / "behaviors_train_tune.parquet", index=False)
    val.to_parquet(out_dir / "behaviors_val.parquet", index=False)

    return {
        "dataset": "ebnerd",
        "n_articles": len(articles),
        "n_train": len(train), "n_train_fit": len(fit), "n_train_tune": len(tune),
        "n_val": len(val),
        "train_time_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "internal_tune_cutoff": str(fit["timestamp"].max()),
        "val_time_range": [str(val["timestamp"].min()), str(val["timestamp"].max())],
    }


def main():
    parser = argparse.ArgumentParser(description="Q1: build the feature store from raw files.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    builders = {"mind": build_mind, "ebnerd": build_ebnerd}
    targets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]

    for name in targets:
        t0 = time.time()
        print(f"[build_pipeline] building feature store for {name} ...")
        summary = builders[name]()
        summary["build_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(summary, indent=2))
        (config.OUTPUTS / name).mkdir(parents=True, exist_ok=True)
        with open(config.OUTPUTS / name / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
