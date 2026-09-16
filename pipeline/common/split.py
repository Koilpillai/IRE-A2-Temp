"""Temporal splitting utilities.

Per the assignment: interaction data must never be split randomly. Every split here
is a cutoff on a timestamp column, so anything after the cutoff is strictly in the
future relative to anything before it.
"""
from __future__ import annotations

import pandas as pd


def temporal_cutoff(df: pd.DataFrame, time_col: str, holdout_fraction: float):
    """Return the timestamp such that `holdout_fraction` of rows fall strictly after it."""
    n = len(df)
    idx = min(max(int(round(n * (1 - holdout_fraction))), 0), n - 1)
    return df[time_col].sort_values().iloc[idx]


def split_by_cutoff(df: pd.DataFrame, time_col: str, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (before, after) the cutoff. `after` is the held-out (future) slice."""
    mask = df[time_col] <= cutoff
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)


def carve_internal_validation(
    df: pd.DataFrame, time_col: str, holdout_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a time-ordered tuning slice out of a train set's own tail.

    Used so hyperparameters (BM25 k1/b, embedding dim, etc.) can be picked without ever
    touching the official validation/dev split, which stays reserved for reported metrics.
    Returns (fit, tune).
    """
    cutoff = temporal_cutoff(df, time_col, holdout_fraction)
    return split_by_cutoff(df, time_col, cutoff)
