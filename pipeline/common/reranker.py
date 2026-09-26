"""Shared GBDT reranker helpers -- used by pipeline/rerank.py (Q2), pipeline/ablation.py
(Q3), pipeline/anti_gaming.py (Q9) and pipeline/generate_submission.py (Q5), so they
don't duplicate the train/predict/group logic.

GBDT (not a neural ranker) is the deliberate choice here, per Q2.2's "Option A" --
see pipeline/rerank.py's module docstring for why Option B (NRMS/an MLP) isn't on the
table in this environment. Trained with XGBoost's histogram GBDT on the GPU (CUDA)
when one is available, CPU otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pipeline.common.embeddings import DEVICE

GBDT_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"

# Every engineered column from pipeline/features.py except the id/label columns.
# EB-NeRD gets two extra columns (dwell time) that MIND's raw files simply don't
# carry -- see pipeline/features.py's module docstring -- so the reranker is always
# trained per-dataset with whatever columns that dataset's feature table actually has.
NON_FEATURE_COLUMNS = {
    "impression_id", "user_id", "candidate_id", "label",
    # diagnostic-only, unclamped freshness kept for test_no_leakage.py -- see
    # pipeline/features.py's CORE_COLUMNS comment. Never fed to the model.
    "freshness_hours_raw",
}

# Q3's "reproduced baseline": a deliberately minimal, non-personalized feature set
# (popularity + freshness + where it was shown) -- no click-history or session signal
# at all. See pipeline/ablation.py for why this stands in for the "official/starter
# baseline" Q3.1 asks for.
FEATURE_COLS_MINIMAL = ["popularity_log", "freshness_hours", "candidate_position_norm"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def load_features(fs: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(fs / f"behavioral_features_{split}.parquet")


def to_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # float32: histogram GBDT bins features anyway, and it halves host RAM + GPU
    # transfer for the 30-50M-row retrieval-fused training tables.
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def train_gbdt(X: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    """XGBoost histogram GBDT (Q2.2's Option A), on the GPU when CUDA is available.

    Hyperparameters carried over 1:1 from the earlier scikit-learn
    HistGradientBoostingClassifier setup (300 rounds, lr 0.08, depth 6, L2 1.0), and so
    is its early-stopping scheme: a stratified 10% holdout of the training rows,
    stopping after 10 rounds without log-loss improvement."""
    X_fit, X_es, y_fit, y_es = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6, reg_lambda=1.0,
        tree_method="hist", device=GBDT_DEVICE, random_state=seed,
        eval_metric="logloss", early_stopping_rounds=10,
        # verbosity=0 also silences the per-call "mismatched devices" notice when a
        # GPU-trained model scores CPU-resident numpy rows at predict time.
        verbosity=0,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


def n_boosting_rounds(model: XGBClassifier) -> int:
    return int(model.best_iteration) + 1


def group_by_impression(df: pd.DataFrame, score_col: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-impression (labels, scores) lists, ordered by candidate_position within
    each impression -- the shape pipeline.common.metrics.evaluate_impressions expects.
    Cross-impression order doesn't matter: metrics are computed per impression and
    then aggregated, so groupby's own group ordering is fine as-is."""
    ordered = df.sort_values(["impression_id", "candidate_position"])
    labels_list = ordered.groupby("impression_id", sort=False)["label"].apply(lambda s: s.to_numpy())
    scores_list = ordered.groupby("impression_id", sort=False)[score_col].apply(lambda s: s.to_numpy())
    return labels_list.tolist(), scores_list.tolist()
