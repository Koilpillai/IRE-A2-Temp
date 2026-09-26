"""Shared GBDT reranker helpers -- used by both pipeline/rerank.py (Q2) and
pipeline/ablation.py (Q3), so the two don't duplicate the train/predict/group logic.

GBDT (not a neural ranker) is the deliberate choice here, per Q2.2's "Option A" --
see pipeline/rerank.py's module docstring for why Option B (NRMS/an MLP) isn't on the
table in this environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def train_gbdt(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    """scikit-learn's own histogram GBDT -- functionally the same algorithm family as
    LightGBM/XGBoost (Q2.2's Option A), chosen because it ships with scikit-learn
    (already a hard requirement) instead of needing a separate native-wheel install."""
    model = HistGradientBoostingClassifier(
        random_state=seed, max_iter=300, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    )
    model.fit(X, y)
    return model


def group_by_impression(df: pd.DataFrame, score_col: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-impression (labels, scores) lists, ordered by candidate_position within
    each impression -- the shape pipeline.common.metrics.evaluate_impressions expects.
    Cross-impression order doesn't matter: metrics are computed per impression and
    then aggregated, so groupby's own group ordering is fine as-is."""
    ordered = df.sort_values(["impression_id", "candidate_position"])
    labels_list = ordered.groupby("impression_id", sort=False)["label"].apply(lambda s: s.to_numpy())
    scores_list = ordered.groupby("impression_id", sort=False)[score_col].apply(lambda s: s.to_numpy())
    return labels_list.tolist(), scores_list.tolist()
