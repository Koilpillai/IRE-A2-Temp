"""Shared helpers for Q5 Codabench submission generation (rank-list format).

Both MIND and EB-NeRD use the identical line format once you supply the right
candidate-list column and output filename:
    impression_id [rank_of_candidate_1,rank_of_candidate_2,...,rank_of_candidate_N]
where rank_of_candidate_i is that candidate's rank (1 = most likely to be clicked)
among the SAME impression's candidates, positionally aligned to the order the
candidates were given in (verified against both datasets' Codabench "Evaluation" /
"Submission Guidelines" pages -- see README).

`hybrid_score` is NOT the final ranking function anymore -- pipeline/generate_
submission.py now ranks candidates with the trained Q2 GBDT (Q5 must reflect "the
full two-stage pipeline", not Stage 1 alone). It's kept because the GBDT's own
`candidate_position` feature was defined at training time as a candidate's rank
under this exact fused score (see pipeline/features.py), so serving still calls it
to compute that one feature consistently with training.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def hybrid_score(bm25_scores: np.ndarray, embed_scores: np.ndarray) -> np.ndarray:
    """Simple, training-free rank fusion: mean of per-impression min-max normalized scores."""
    return 0.5 * min_max_normalize(np.asarray(bm25_scores, dtype=np.float64)) \
        + 0.5 * min_max_normalize(np.asarray(embed_scores, dtype=np.float64))


def scores_to_ranks(scores: np.ndarray) -> list[int]:
    """1-indexed ranks, rank 1 = highest score; ties broken deterministically by position."""
    return rankdata(-np.asarray(scores, dtype=np.float64), method="ordinal").astype(int).tolist()


def format_line(impression_id, ranks: list[int]) -> str:
    return f"{impression_id} [{','.join(str(r) for r in ranks)}]\n"


def write_submission_zip(txt_path: Path, zip_path: Path, arcname: str) -> None:
    """Zip containing *only* `arcname` at the root -- no subfolders, no __MACOSX junk."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=arcname)
