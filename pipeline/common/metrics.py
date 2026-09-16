"""Offline evaluation harness: accuracy metrics + beyond-accuracy metrics + bootstrap CIs.

Accuracy metrics (AUC, MRR, nDCG@5, nDCG@10) follow the official MIND/RecSys-challenge
convention: computed per impression against that impression's own candidate list, then
averaged across impressions (a "grouped" evaluation, not a pooled one).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    k = min(k, len(y_true_sorted))
    if k <= 0:
        return 0.0
    rel = y_true_sorted[:k]
    gains = (2.0 ** rel) - 1.0
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)
    actual = dcg_at_k(y_true[order], k)
    ideal = dcg_at_k(np.sort(y_true)[::-1], k)
    return actual / ideal if ideal > 0 else 0.0


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    ranks = np.arange(1, len(y_true_sorted) + 1)
    total_pos = y_true_sorted.sum()
    if total_pos == 0:
        return 0.0
    return float(np.sum(y_true_sorted / ranks) / total_pos)


def evaluate_impressions(labels_list: list[np.ndarray], scores_list: list[np.ndarray]) -> dict:
    """Per-impression metrics. Returns arrays (one entry per *valid* impression for that metric)."""
    auc_vals, mrr_vals, ndcg5_vals, ndcg10_vals = [], [], [], []
    for y_true, y_score in zip(labels_list, scores_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=np.float64)
        if y_true.sum() > 0:
            mrr_vals.append(mrr_score(y_true, y_score))
            ndcg5_vals.append(ndcg_at_k(y_true, y_score, 5))
            ndcg10_vals.append(ndcg_at_k(y_true, y_score, 10))
        if 0 < y_true.sum() < len(y_true):
            auc_vals.append(roc_auc_score(y_true, y_score))
    return {
        "auc": np.array(auc_vals), "mrr": np.array(mrr_vals),
        "ndcg5": np.array(ndcg5_vals), "ndcg10": np.array(ndcg10_vals),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def summarize(metric_arrays: dict, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    out = {}
    for name, arr in metric_arrays.items():
        mean, lo, hi = bootstrap_ci(arr, n_boot=n_boot, ci=ci, seed=seed)
        out[name] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(len(arr))}
    return out


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------

def intra_list_diversity(top_n_id_lists: list[list], embedding_index) -> np.ndarray:
    """1 - mean pairwise cosine similarity within each recommended list (content-space)."""
    out = np.zeros(len(top_n_id_lists))
    for i, ids in enumerate(top_n_id_lists):
        vecs = embedding_index.vectors_for(ids)
        if len(vecs) < 2:
            out[i] = np.nan
            continue
        sims = vecs @ vecs.T
        n = sims.shape[0]
        off_diag_sum = sims.sum() - np.trace(sims)
        mean_sim = off_diag_sum / (n * (n - 1))
        out[i] = 1.0 - mean_sim
    return out


def novelty(top_n_id_lists: list[list], popularity: dict, total_clicks: int, catalog_size: int) -> np.ndarray:
    """Mean self-information -log2(p(item)) of recommended items, Laplace-smoothed."""
    denom = total_clicks + catalog_size
    out = np.zeros(len(top_n_id_lists))
    for i, ids in enumerate(top_n_id_lists):
        if len(ids) == 0:
            out[i] = np.nan
            continue
        scores = [-np.log2((popularity.get(a, 0) + 1) / denom) for a in ids]
        out[i] = float(np.mean(scores))
    return out


def coverage(top_n_id_lists: list[list], catalog_size: int) -> float:
    seen = set()
    for ids in top_n_id_lists:
        seen.update(ids)
    return len(seen) / catalog_size if catalog_size else 0.0
