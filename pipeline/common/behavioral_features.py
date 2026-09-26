"""Shared behavioural-feature formulas -- used by both pipeline/features.py (Q1,
building the labeled train/val feature tables) and pipeline/generate_submission.py
(Q5, scoring the unlabeled Codabench test candidates with the trained reranker).

Pulled out of pipeline/features.py so the two call sites can't drift into computing
"the same" feature two different ways (a real train/serving-skew risk), not because
either site needed a new abstraction on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def rank_decay_weights(n: int, decay: float) -> np.ndarray:
    """History list is oldest -> most-recent; the most recent item gets weight 1."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return decay ** np.arange(n - 1, -1, -1, dtype=np.float64)


def time_decay_weights(item_times: np.ndarray, ref_time, half_life_hours: float) -> np.ndarray:
    if len(item_times) == 0:
        return np.zeros(0, dtype=np.float64)
    # np.datetime64(...) explicitly, not the bare pd.Timestamp -- numpy 2.x's ufunc
    # dispatch no longer auto-converts a Timestamp scalar against a datetime64 array
    # (raises _UFuncBinaryResolutionError on dtype('O') vs dtype('<M8[us]')).
    delta_hours = (np.datetime64(ref_time) - item_times) / np.timedelta64(1, "h")
    delta_hours = np.clip(delta_hours.astype(np.float64), 0.0, None)
    return 0.5 ** (delta_hours / half_life_hours)


def weighted_mean_vec(rows: list[int], weights: np.ndarray, article_vecs: np.ndarray) -> np.ndarray:
    dim = article_vecs.shape[1]
    if not rows or weights.sum() <= 0:
        return np.zeros(dim, dtype=np.float32)
    w = weights[:len(rows)]
    return (article_vecs[rows] * w[:, None]).sum(axis=0) / w.sum()


def safe_nanmean(arr: np.ndarray) -> float:
    """np.nanmean, but 0.0 (not NaN + a RuntimeWarning) for empty/all-NaN input --
    EB-NeRD's read_time_fixed/scroll_percentage_fixed do contain per-item NaNs."""
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def cosine(a: np.ndarray | None, b: np.ndarray) -> float:
    if a is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Session bucketing (causal: position among STRICTLY PRIOR impressions only)
# ---------------------------------------------------------------------------

def session_position_mind(behaviors: pd.DataFrame, gap_minutes: float = 30.0) -> pd.Series:
    """MIND has no session_id -- bucket by a 30-minute inactivity gap per user, a
    standard session-boundary heuristic (see design note for the choice of threshold)."""
    df = behaviors[["impression_id", "user_id", "timestamp"]].sort_values(["user_id", "timestamp"])
    gap = df.groupby("user_id")["timestamp"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    session_key = new_session.groupby(df["user_id"]).cumsum()
    position = df.groupby([df["user_id"], session_key]).cumcount()
    return pd.Series(position.values, index=df["impression_id"]).reindex(behaviors["impression_id"]).reset_index(drop=True)


def session_position_ebnerd(behaviors: pd.DataFrame, session_ids: pd.DataFrame) -> pd.Series:
    df = behaviors[["impression_id", "user_id", "timestamp"]].merge(session_ids, on="impression_id", how="left")
    df = df.sort_values(["user_id", "session_id", "timestamp"])
    position = df.groupby(["user_id", "session_id"]).cumcount()
    return pd.Series(position.values, index=df["impression_id"]).reindex(behaviors["impression_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Freshness lookups (train-only where a proxy is needed)
# ---------------------------------------------------------------------------

def mind_first_appearance(fs: Path) -> dict:
    """Article -> earliest TRAIN-split timestamp it appears in any candidate list --
    MIND's freshness proxy (see pipeline/features.py's module docstring). Train-only,
    reused unchanged for train, val, and test-time serving alike."""
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["timestamp", "candidates"])
    first_seen: dict = {}
    for ts, cands in zip(train["timestamp"], train["candidates"]):
        for c in cands:
            prev = first_seen.get(c)
            if prev is None or ts < prev:
                first_seen[c] = ts
    return first_seen


def fallback_freshness_hours(fs: Path, lookup: dict, seed: int) -> float:
    """Median freshness observed on a TRAIN sample -- used only for the rare candidate
    with no known publish/first-appearance time, so a missing lookup doesn't silently
    become "brand new" (freshness_hours=0)."""
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["timestamp", "candidates"])
    sample = train.sample(n=min(20_000, len(train)), random_state=seed)
    vals = []
    for ts, cands in zip(sample["timestamp"], sample["candidates"]):
        for c in cands:
            t0 = lookup.get(c)
            if t0 is not None:
                vals.append((ts - t0) / np.timedelta64(1, "h"))
    return float(np.median(vals)) if vals else 0.0
