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
                       candidate's rank in the retrieval-fused candidate pool built
                       for this row (see "Candidate generation" below), not a
                       platform display index

Recency weighting uses real elapsed time where it's available and a position-rank
proxy where it isn't: EB-NeRD's history.parquet carries a per-item impression_time_
fixed, but MIND's news.tsv/behaviors.tsv carry no per-item history timestamp at all
-- only order. Same split for freshness: EB-NeRD's articles.parquet has a real
published_time; MIND has none, so freshness there is a proxy (see
pipeline.common.behavioral_features.mind_first_appearance). Both limitations are
inherent to the raw MIND files, not a shortcut taken here -- flagged for the design
note.

Candidate generation (Q2.1: "Use Assignment 1's candidate generator to retrieve
top-K candidates (K ~ 100-200)"): each row's candidate pool is the union of A1's own
BM25 and Word2Vec+ANN top-config.RETRIEVAL_CANDIDATE_K retrieval over the FULL
article corpus, queried from the user's click-history window -- the same query
construction pipeline/retrieval_eval.py's `_build_queries` uses (plain mean-pooled
embedding + concatenated BM25 text), not the recency-weighted hist_vec used for the
hist_embed_sim feature below. Because a self-retrieved top-K can miss the article the
user actually clicked -- and per outputs/<dataset>/recall_at_k.json it usually does,
recall@100-200 over this ~125K-article corpus measuring only ~0.4-2.3% -- that
impression's true positive(s) (from the dataset's own officially-provided
candidates/labels) are force-included whenever retrieval didn't surface them, so no
impression silently loses its supervised label. They're spliced in at a uniformly
random position among the genuinely-retrieved candidates, NOT appended at a fixed
position: with retrieval missing the positive ~98%+ of the time, a fixed insertion
point would make `candidate_position` an almost perfect giveaway of the label -- a
leakage shortcut the GBDT could learn instead of the real features. This does NOT
touch pipeline/retrieval_eval.py's recall@K, which stays an honest, untouched measure
of open-corpus retrieval quality.

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
from pipeline.common import behavioral_features as bf
from pipeline.common.bm25 import BM25Index
from pipeline.common.embeddings import EmbeddingIndex
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


def _build_retrieval_indices(articles: pd.DataFrame, article_vecs: np.ndarray) -> tuple[BM25Index, EmbeddingIndex]:
    """A1's own candidate generator: a fresh BM25 index (cheap to rebuild -- see
    generate_submission.py) plus an ANN index over the already-persisted embeddings."""
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B).fit(
        articles["article_id"].tolist(), articles["text_lexical"].tolist())
    emb_index = EmbeddingIndex(articles["article_id"].tolist(), article_vecs)
    return bm25, emb_index


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
    id_to_text = dict(zip(articles["article_id"], articles["text_lexical"]))
    id_to_category = dict(zip(articles["article_id"], articles["category"]))
    popularity = train_popularity(fs)
    bm25, emb_index = _build_retrieval_indices(articles, article_vecs)
    emb_dim = article_vecs.shape[1]
    # Retrieval recall@100-200 over a 125K-article corpus is measured at ~0.4-2.3%
    # (outputs/<dataset>/recall_at_k.json) -- so a force-included true positive missed
    # by retrieval is the common case, not the rare one. Appending it at a fixed tail
    # position would make `candidate_position` a near-perfect giveaway of the label
    # (a leakage shortcut the GBDT could trivially learn instead of the real features),
    # so it's inserted at a uniformly random slot among the genuinely-retrieved
    # candidates instead -- seeded for reproducibility, not for realism.
    positive_insert_rng = np.random.default_rng(config.RANDOM_SEED)

    if is_ebnerd:
        publish_times = ebnerd_adapter.load_publish_times(
            config.EBNERD_SMALL_DIR / "articles.parquet", config.EBNERD_TESTSET_DIR / "articles.parquet")
        publish_time_by_article = dict(zip(publish_times["article_id"], publish_times["published_time"]))
        split_dir = config.EBNERD_SMALL_DIR / ("train" if split == "train" else "validation")
        session_ids = ebnerd_adapter.load_session_ids(split_dir / "behaviors.parquet")
        session_position = bf.session_position_ebnerd(behaviors, session_ids).to_numpy()
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
        fallback_fresh = bf.fallback_freshness_hours(fs, publish_time_by_article, config.RANDOM_SEED)
    else:
        first_appearance = bf.mind_first_appearance(fs)
        session_position = bf.session_position_mind(behaviors).to_numpy()
        history_table = pd.read_parquet(fs / f"user_history_{split}.parquet")
        history_by_user = dict(zip(history_table["user_id"], history_table["history"]))
        fallback_fresh = bf.fallback_freshness_hours(fs, first_appearance, config.RANDOM_SEED)

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

        # --- Pass A: per-row context + retrieval queries (batched below) ---
        row_ctx = []
        bm25_queries = []
        embed_queries = []

        for i in range(start, end):
            imp_id = behaviors["impression_id"].iat[i]
            user_id = behaviors["user_id"].iat[i]
            ref_time = behaviors["timestamp"].iat[i]
            official_cands = behaviors["candidates"].iat[i]
            official_labels = behaviors["labels"].iat[i]
            official_labels = official_labels if official_labels is not None else [None] * len(official_cands)
            sess_pos = int(session_position[i])

            if is_ebnerd:
                ids_full, times_full, reads_full, scrolls_full = history_by_user.get(
                    user_id, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype="datetime64[us]"),
                              np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)))
                window_ids = ids_full[-config.FEATURE_HISTORY_WINDOW:]
                window_times = times_full[-config.FEATURE_HISTORY_WINDOW:]
                weights_full = bf.time_decay_weights(window_times, ref_time, config.HIST_HALF_LIFE_HOURS)
                hist_len = int(len(ids_full))
                avg_read = bf.safe_nanmean(reads_full)
                avg_scroll = bf.safe_nanmean(scrolls_full)
            else:
                hist_ids = history_by_user.get(user_id)
                hist_ids = list(hist_ids) if hist_ids is not None and len(hist_ids) else []
                window_ids = np.asarray(hist_ids[-config.FEATURE_HISTORY_WINDOW:])
                weights_full = bf.rank_decay_weights(len(window_ids), config.HIST_DECAY_RANK)
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
            hist_vec = bf.weighted_mean_vec(hist_rows, weights_emb, article_vecs)

            # Retrieval query for candidate generation: A1's own plain (non-recency-
            # weighted) mean-pool + concatenated text, mirroring retrieval_eval.py's
            # _build_queries exactly -- independent of the recency-weighted hist_vec above.
            bm25_queries.append(" ".join(id_to_text.get(a, "") for a in window_ids))
            embed_queries.append(article_vecs[hist_rows].mean(axis=0) if hist_rows else np.zeros(emb_dim, dtype=np.float32))

            positives = {c for c, l in zip(official_cands, official_labels) if l == 1}

            row_ctx.append({
                "imp_id": imp_id, "user_id": user_id, "ref_time": ref_time, "sess_pos": sess_pos,
                "hist_len": hist_len, "avg_read": avg_read, "avg_scroll": avg_scroll,
                "cat_counts": cat_counts, "n_cat_known": n_cat_known, "top_cat": top_cat,
                "cat_weight": cat_weight, "total_w": total_w, "hist_vec": hist_vec,
                "positives": positives,
            })

        bm25_topk_chunk = bm25.top_k_batch(bm25_queries, k=config.RETRIEVAL_CANDIDATE_K)
        embed_topk_chunk = emb_index.top_k_batch(np.stack(embed_queries), k=config.RETRIEVAL_CANDIDATE_K)

        # --- Pass B: fuse retrieved candidates, guarantee positives, emit rows ---
        for ctx, bm25_top, embed_top in zip(row_ctx, bm25_topk_chunk, embed_topk_chunk):
            fused = list(dict.fromkeys(list(bm25_top) + list(embed_top)))
            missing_positives = [p for p in ctx["positives"] if p not in fused]
            final_cands = list(fused)
            for p in missing_positives:
                final_cands.insert(int(positive_insert_rng.integers(0, len(final_cands) + 1)), p)
            final_labels = [1 if c in ctx["positives"] else 0 for c in final_cands]

            cat_counts, n_cat_known, top_cat = ctx["cat_counts"], ctx["n_cat_known"], ctx["top_cat"]
            cat_weight, total_w, hist_vec = ctx["cat_weight"], ctx["total_w"], ctx["hist_vec"]

            for pos, (cand, label) in enumerate(zip(final_cands, final_labels)):
                cand_row = id_to_row.get(cand)
                cand_vec = article_vecs[cand_row] if cand_row is not None else None
                cand_cat = id_to_category.get(cand)

                rows["impression_id"].append(ctx["imp_id"])
                rows["user_id"].append(ctx["user_id"])
                rows["candidate_id"].append(cand)
                rows["label"].append(label)
                rows["hist_len"].append(ctx["hist_len"])
                rows["hist_category_match_frac"].append(
                    (cat_counts.get(cand_cat, 0) / n_cat_known) if (cand_cat is not None and n_cat_known > 0) else 0.0)
                rows["hist_category_match_recency"].append(
                    (cat_weight.get(cand_cat, 0.0) / total_w) if (cand_cat is not None and total_w > 0) else 0.0)
                rows["hist_embed_sim"].append(bf.cosine(cand_vec, hist_vec))
                rows["session_position"].append(ctx["sess_pos"])
                rows["popularity_log"].append(float(np.log1p(popularity.get(cand, 0))))
                if is_ebnerd:
                    pub = publish_time_by_article.get(cand)
                    fresh = max((ctx["ref_time"] - pub) / np.timedelta64(1, "h"), 0.0) if pub is not None else fallback_fresh
                else:
                    t0_first = first_appearance.get(cand)
                    fresh = max((ctx["ref_time"] - t0_first) / np.timedelta64(1, "h"), 0.0) if t0_first is not None else fallback_fresh
                rows["freshness_hours"].append(float(fresh))
                rows["category_match"].append(int(cand_cat is not None and cand_cat == top_cat))
                rows["candidate_position"].append(pos)
                rows["candidate_position_norm"].append(pos / max(len(final_cands) - 1, 1))
                if is_ebnerd:
                    rows["user_avg_read_time"].append(ctx["avg_read"])
                    rows["user_avg_scroll_pct"].append(ctx["avg_scroll"])

        chunk_df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            # use_dictionary=False: each chunk's pa.Table.from_pandas() otherwise
            # dictionary-encodes string columns (MIND's candidate_id/user_id) with
            # its own per-chunk dictionary, and reading the resulting multi-row-group
            # file back raises "Column cannot have more than one dictionary" on this
            # pyarrow version. Plain (non-dictionary) encoding sidesteps it entirely.
            writer = pq.ParquetWriter(out_path, table.schema, use_dictionary=False)
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
