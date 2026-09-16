"""A2 Q1 -- Click-history & session behavioural features for the two-stage reranker.

Builds one row per (impression, candidate) pair with the engineered features Q1 asks
for, grouped by its four sub-requirements:

  1. Click-history:  hist_len, hist_category_match_frac (unweighted), hist_category_
                      match_recency (recency-weighted), hist_embed_sim (candidate's
                      Word2Vec vector vs. the recency-weighted mean of the user's
                      history vectors -- reuses A1's article_embeddings.npy)
  2. Session:         session_position (impressions-so-far in this user's current
                       session -- see boundary note below); user_avg_read_time and
                       user_avg_scroll_pct where available (EB-NeRD only)
  3. Article:         popularity (train-click count, log1p), freshness_hours,
                       category_match (candidate's category == user's single most-
                       clicked historical category)
  4. Position:        candidate_position and its list-length-normalized form -- the
                       position the candidate was actually shown at (position bias),
                       taken from the officially provided candidate-list order

Recency weighting uses real elapsed time where it's available and a position-rank
proxy where it isn't: EB-NeRD's history.parquet carries a per-item impression_time_
fixed, but MIND's news.tsv/behaviors.tsv carry no per-item history timestamp at all
-- only order. Same split for freshness: EB-NeRD's articles.parquet has a real
published_time; MIND has none, so freshness there is a proxy (see _mind_first_
appearance below). Both limitations are inherent to the raw MIND files, not a
shortcut taken here -- flagged for the design note.

Behavioural-window boundary (Q1.4 / Q9) -- the three places this module enforces it:
  - `popularity` is counted from behaviors_train.parquet ONLY (pipeline.common.
    popularity), reused unchanged for both the train split's own features and val's,
    the same pattern A1's Q4/Q5 already used for their popularity fallback.
  - `session_position` counts only impressions STRICTLY BEFORE this one in the same
    session -- never the session's eventual total size, which isn't knowable yet for
    an in-progress session at serving time.
  - MIND's freshness proxy (hours since an article's first appearance in any TRAIN
    candidate list) is, by construction, always <= the impression time it's computed
    for, and is only ever built from behaviors_train.parquet.
  - History-derived features only ever read user_history_<split>.parquet /
    history.parquet for that same split, which A1's build_pipeline already restricted
    to what preceded that split's own window -- inherited here, re-verified by
    pipeline/tests/test_no_leakage.py's existing split-boundary checks. The two new
    checks this module adds are in test_no_leakage.py::test_features_* below.

Usage:
    python -m pipeline.features --dataset mind --split all
    python -m pipeline.features --dataset ebnerd --split train --sample 750  # smoke test
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import config
from pipeline.adapters import ebnerd as ebnerd_adapter
from pipeline.common.popularity import train_popularity

CORE_COLUMNS = [
    "impression_id", "user_id", "candidate_id", "label",
    "hist_len", "hist_category_match_frac", "hist_category_match_recency", "hist_embed_sim",
    "session_position", "popularity_log", "freshness_hours", "category_match",
    "candidate_position", "candidate_position_norm",
]
# EB-NeRD-only: MIND's raw files carry neither per-item history timestamps nor any
# dwell-time/scroll signal, so these two columns simply don't exist for MIND -- the
# reranker (Q2) trains one model per dataset anyway, so the two feature sets don't
# need to match.
EBNERD_EXTRA_COLUMNS = ["user_avg_read_time", "user_avg_scroll_pct"]


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def _rank_decay_weights(n: int, decay: float) -> np.ndarray:
    """History list is oldest -> most-recent; the most recent item gets weight 1."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return decay ** np.arange(n - 1, -1, -1, dtype=np.float64)


def _time_decay_weights(item_times: np.ndarray, ref_time, half_life_hours: float) -> np.ndarray:
    if len(item_times) == 0:
        return np.zeros(0, dtype=np.float64)
    delta_hours = (ref_time - item_times) / np.timedelta64(1, "h")
    delta_hours = np.clip(delta_hours.astype(np.float64), 0.0, None)
    return 0.5 ** (delta_hours / half_life_hours)


def _weighted_mean_vec(rows: list[int], weights: np.ndarray, article_vecs: np.ndarray) -> np.ndarray:
    dim = article_vecs.shape[1]
    if not rows or weights.sum() <= 0:
        return np.zeros(dim, dtype=np.float32)
    w = weights[:len(rows)]
    return (article_vecs[rows] * w[:, None]).sum(axis=0) / w.sum()


def _cosine(a: np.ndarray | None, b: np.ndarray) -> float:
    if a is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Session bucketing (causal: position among STRICTLY PRIOR impressions only)
# ---------------------------------------------------------------------------

def _session_position_mind(behaviors: pd.DataFrame, gap_minutes: float = 30.0) -> pd.Series:
    """MIND has no session_id -- bucket by a 30-minute inactivity gap per user, a
    standard session-boundary heuristic (see design note for the choice of threshold)."""
    df = behaviors[["impression_id", "user_id", "timestamp"]].sort_values(["user_id", "timestamp"])
    gap = df.groupby("user_id")["timestamp"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    session_key = new_session.groupby(df["user_id"]).cumsum()
    position = df.groupby([df["user_id"], session_key]).cumcount()
    return pd.Series(position.values, index=df["impression_id"]).reindex(behaviors["impression_id"]).reset_index(drop=True)


def _session_position_ebnerd(behaviors: pd.DataFrame, session_ids: pd.DataFrame) -> pd.Series:
    df = behaviors[["impression_id", "user_id", "timestamp"]].merge(session_ids, on="impression_id", how="left")
    df = df.sort_values(["user_id", "session_id", "timestamp"])
    position = df.groupby(["user_id", "session_id"]).cumcount()
    return pd.Series(position.values, index=df["impression_id"]).reindex(behaviors["impression_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Freshness lookups (train-only where a proxy is needed)
# ---------------------------------------------------------------------------

def _mind_first_appearance(fs: Path) -> dict:
    """Article -> earliest TRAIN-split timestamp it appears in any candidate list --
    MIND's freshness proxy (see module docstring). Train-only, reused unchanged for
    both train's and val's own feature builds."""
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["timestamp", "candidates"])
    first_seen: dict = {}
    for ts, cands in zip(train["timestamp"], train["candidates"]):
        for c in cands:
            prev = first_seen.get(c)
            if prev is None or ts < prev:
                first_seen[c] = ts
    return first_seen


def _fallback_freshness_hours(fs: Path, lookup: dict) -> float:
    """Median freshness observed on a TRAIN sample -- used only for the rare candidate
    with no known publish/first-appearance time, so a missing lookup doesn't silently
    become "brand new" (freshness_hours=0)."""
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["timestamp", "candidates"])
    sample = train.sample(n=min(20_000, len(train)), random_state=config.RANDOM_SEED)
    vals = []
    for ts, cands in zip(sample["timestamp"], sample["candidates"]):
        for c in cands:
            t0 = lookup.get(c)
            if t0 is not None:
                vals.append((ts - t0) / np.timedelta64(1, "h"))
    return float(np.median(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_features(dataset: str, split: str, sample: int | None = None) -> Path:
    assert split in ("train", "val")
    fs = config.FEATURE_STORE / dataset
    is_ebnerd = dataset == "ebnerd"

    articles = pd.read_parquet(fs / "articles.parquet")
    behaviors = pd.read_parquet(fs / f"behaviors_{split}.parquet")
    if sample:
        behaviors = behaviors.sample(n=min(sample, len(behaviors)), random_state=config.RANDOM_SEED).reset_index(drop=True)

    article_vecs = np.load(fs / "article_embeddings.npy")
    id_to_row = {a: i for i, a in enumerate(articles["article_id"])}
    id_to_category = dict(zip(articles["article_id"], articles["category"]))
    popularity = train_popularity(fs)

    if is_ebnerd:
        publish_times = ebnerd_adapter.load_publish_times(
            config.EBNERD_SMALL_DIR / "articles.parquet", config.EBNERD_TESTSET_DIR / "articles.parquet")
        publish_time_by_article = dict(zip(publish_times["article_id"], publish_times["published_time"]))
        split_dir = config.EBNERD_SMALL_DIR / ("train" if split == "train" else "validation")
        session_ids = ebnerd_adapter.load_session_ids(split_dir / "behaviors.parquet")
        session_position = _session_position_ebnerd(behaviors, session_ids).to_numpy()
        full_history = ebnerd_adapter.load_user_engagement_history(split_dir / "history.parquet")
        history_by_user = {}
        for user_id, ids, times, reads, scrolls in zip(
            full_history["user_id"], full_history["history"], full_history["impression_time_fixed"],
            full_history["read_time_fixed"], full_history["scroll_percentage_fixed"],
        ):
            history_by_user[user_id] = (
                np.asarray(ids, dtype=np.int64),
                np.asarray(times, dtype="datetime64[us]"),
                np.asarray(reads, dtype=np.float64),
                np.asarray(scrolls, dtype=np.float64),
            )
        fallback_fresh = _fallback_freshness_hours(fs, publish_time_by_article)
    else:
        first_appearance = _mind_first_appearance(fs)
        session_position = _session_position_mind(behaviors).to_numpy()
        history_table = pd.read_parquet(fs / f"user_history_{split}.parquet")
        history_by_user = dict(zip(history_table["user_id"], history_table["history"]))
        fallback_fresh = _fallback_freshness_hours(fs, first_appearance)

    out_path = fs / f"behavioral_features_{split}.parquet"
    schema_cols = CORE_COLUMNS + (EBNERD_EXTRA_COLUMNS if is_ebnerd else [])

    writer = None
    n_rows = 0
    t0 = time.time()
    n = len(behaviors)
    chunk_size = config.FEATURE_BUILD_CHUNK

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        rows = {c: [] for c in schema_cols}

        for i in range(start, end):
            imp_id = behaviors["impression_id"].iat[i]
            user_id = behaviors["user_id"].iat[i]
            ref_time = behaviors["timestamp"].iat[i]
            cands = behaviors["candidates"].iat[i]
            labels = behaviors["labels"].iat[i]
            labels = labels if labels is not None else [None] * len(cands)
            sess_pos = int(session_position[i])

            if is_ebnerd:
                ids_full, times_full, reads_full, scrolls_full = history_by_user.get(
                    user_id, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype="datetime64[us]"),
                              np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)))
                window_ids = ids_full[-config.FEATURE_HISTORY_WINDOW:]
                window_times = times_full[-config.FEATURE_HISTORY_WINDOW:]
                weights_full = _time_decay_weights(window_times, ref_time, config.HIST_HALF_LIFE_HOURS)
                hist_len = int(len(ids_full))
                avg_read = float(np.nanmean(reads_full)) if len(reads_full) else 0.0
                avg_scroll = float(np.nanmean(scrolls_full)) if len(scrolls_full) else 0.0
            else:
                hist_ids = history_by_user.get(user_id)
                hist_ids = list(hist_ids) if hist_ids is not None and len(hist_ids) else []
                window_ids = np.asarray(hist_ids[-config.FEATURE_HISTORY_WINDOW:])
                weights_full = _rank_decay_weights(len(window_ids), config.HIST_DECAY_RANK)
                hist_len = len(hist_ids)
                avg_read = avg_scroll = None

            window_cats = [id_to_category.get(a) for a in window_ids]
            cat_counts: Counter = Counter(c for c in window_cats if c is not None)
            n_cat_known = sum(cat_counts.values())
            top_cat = cat_counts.most_common(1)[0][0] if cat_counts else None

            cat_weight: dict = {}
            total_w = float(weights_full.sum()) if len(weights_full) else 0.0
            for cat, w in zip(window_cats, weights_full):
                if cat is not None:
                    cat_weight[cat] = cat_weight.get(cat, 0.0) + float(w)

            emb_mask = np.array([a in id_to_row for a in window_ids], dtype=bool) if len(window_ids) else np.zeros(0, dtype=bool)
            hist_rows = [id_to_row[a] for a, m in zip(window_ids, emb_mask) if m]
            weights_emb = weights_full[emb_mask] if len(weights_full) else weights_full
            hist_vec = _weighted_mean_vec(hist_rows, weights_emb, article_vecs)

            for pos, (cand, label) in enumerate(zip(cands, labels)):
                cand_row = id_to_row.get(cand)
                cand_vec = article_vecs[cand_row] if cand_row is not None else None
                cand_cat = id_to_category.get(cand)

                rows["impression_id"].append(imp_id)
                rows["user_id"].append(user_id)
                rows["candidate_id"].append(cand)
                rows["label"].append(label)
                rows["hist_len"].append(hist_len)
                rows["hist_category_match_frac"].append(
                    (cat_counts.get(cand_cat, 0) / n_cat_known) if (cand_cat is not None and n_cat_known > 0) else 0.0)
                rows["hist_category_match_recency"].append(
                    (cat_weight.get(cand_cat, 0.0) / total_w) if (cand_cat is not None and total_w > 0) else 0.0)
                rows["hist_embed_sim"].append(_cosine(cand_vec, hist_vec))
                rows["session_position"].append(sess_pos)
                rows["popularity_log"].append(float(np.log1p(popularity.get(cand, 0))))
                if is_ebnerd:
                    pub = publish_time_by_article.get(cand)
                    fresh = max((ref_time - pub) / np.timedelta64(1, "h"), 0.0) if pub is not None else fallback_fresh
                else:
                    t0_first = first_appearance.get(cand)
                    fresh = max((ref_time - t0_first) / np.timedelta64(1, "h"), 0.0) if t0_first is not None else fallback_fresh
                rows["freshness_hours"].append(float(fresh))
                rows["category_match"].append(int(cand_cat is not None and cand_cat == top_cat))
                rows["candidate_position"].append(pos)
                rows["candidate_position_norm"].append(pos / max(len(cands) - 1, 1))
                if is_ebnerd:
                    rows["user_avg_read_time"].append(avg_read)
                    rows["user_avg_scroll_pct"].append(avg_scroll)

        chunk_df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_rows += len(chunk_df)
        print(f"[{dataset}/{split}] features: {n_rows:,}/-  rows written "
              f"({end:,}/{n:,} impressions, {time.time()-t0:.0f}s elapsed)")

    if writer is not None:
        writer.close()
    print(f"[{dataset}/{split}] DONE: {n_rows:,} (impression, candidate) rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="A2 Q1: build behavioural features.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None, help="subsample impressions (smoke test)")
    args = parser.parse_args()
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for ds in datasets:
        for sp in splits:
            build_features(ds, sp, sample=args.sample)


if __name__ == "__main__":
    main()
