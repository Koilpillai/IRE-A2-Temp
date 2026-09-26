"""Q5 -- Generate Codabench prediction files for both leaderboards.

Scores each impression's own OFFICIALLY-GIVEN candidate list with the trained Q2
GBDT reranker (feature_store/<dataset>/reranker_model.joblib) -- this is "the full
two-stage pipeline" Q5 asks the leaderboard predictions to come from, not Stage 1
alone. Unlike pipeline/features.py's retrieval-based candidate generation for
training/eval, the candidate SET here is never swapped for a self-retrieved pool:
Codabench's own server-side scoring is a permutation check against the exact given
list, so this module only changes how that fixed list gets scored.

Feature columns are computed with the exact same formulas pipeline/features.py uses
(shared via pipeline/common/behavioral_features.py), so training and serving compute
features identically rather than via two independently-maintained copies that could
drift (a real train/serving-skew risk otherwise). One feature needs a small
adjustment: `candidate_position` was defined at training time (post Q2.1 fix) as a
candidate's rank under A1's own BM25+embedding retrieval fused score -- so at serving
time it's computed the same way, by scoring the GIVEN candidates (not searching the
full corpus) with pipeline/common/submission.hybrid_score() and ranking them, not
from the candidate's raw position in Codabench's input file.

Falls back to popularity-only ranking only for the trivial len(candidates)<=1 case
(nothing to rank); cold-start (empty history) users are NOT special-cased beyond that
-- the GBDT sees the same zero-valued history features cold-start rows get at
train/val time, so serving does not add a training-time-unseen code path.

Usage:
    python pipeline/generate_submission.py --dataset mind
    python pipeline/generate_submission.py --dataset ebnerd
    python pipeline/generate_submission.py --dataset mind --sample 750   # smoke test
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.adapters import ebnerd as ebnerd_adapter
from pipeline.adapters import mind as mind_adapter
from pipeline.common import behavioral_features as bf
from pipeline.common.bm25 import BM25Index
from pipeline.common.embeddings import DEVICE, EmbeddingIndex
from pipeline.common.popularity import train_popularity as _train_popularity
from pipeline.common.submission import format_line, hybrid_score, scores_to_ranks, write_submission_zip

SUBBATCH_SIZE = 20_000  # bounds peak memory regardless of the caller's own chunk size


def _load_context(fs: Path) -> dict:
    articles = pd.read_parquet(fs / "articles.parquet")
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B).fit(
        articles["article_id"].tolist(), articles["text_lexical"].tolist())
    article_vecs = np.load(fs / "article_embeddings.npy")
    emb = EmbeddingIndex(articles["article_id"].tolist(), article_vecs)
    model_bundle = joblib.load(fs / "reranker_model.joblib")
    model = model_bundle["model"]
    if DEVICE != "cuda":
        # A GPU-trained booster still runs fine on CPU-only inference (tree traversal,
        # not matrix ops) -- but only once its device param is explicitly flipped, so a
        # parallel worker process with no CUDA visible (see _run_mind_parallel) doesn't
        # try to touch a GPU that isn't there.
        model.set_params(device="cpu")
    return {
        "bm25": bm25, "emb": emb, "article_vecs": article_vecs,
        "id_to_row": {a: i for i, a in enumerate(articles["article_id"])},
        "id_to_text": dict(zip(articles["article_id"], articles["text_lexical"])),
        "id_to_category": dict(zip(articles["article_id"], articles["category"])),
        "popularity": _train_popularity(fs),
        "model": model, "feature_cols": model_bundle["feature_cols"],
    }


def _session_positions_mind_test(path: Path) -> dict:
    """One lightweight upfront pass over the full test file's (user_id, timestamp)
    columns -- session_position needs a user's impressions sorted together, which
    isn't available from a single chunk when the file is streamed."""
    raw = pd.read_csv(path, sep="\t", header=None, names=mind_adapter.BEHAVIOR_COLS, quoting=3,
                       usecols=["impression_id", "user_id", "time"],
                       dtype={"impression_id": "int64", "user_id": "str"})
    raw["timestamp"] = pd.to_datetime(raw["time"], format="%m/%d/%Y %I:%M:%S %p")
    pos = bf.session_position_mind(raw[["impression_id", "user_id", "timestamp"]])
    return dict(zip(raw["impression_id"], pos))


def _session_positions_ebnerd_test(behaviors_path: Path) -> dict:
    """NOTE: the real large EB-NeRD test file has ~200K rows sharing impression_id=0
    (an apparent masking sentinel for withheld rows) -- joining/reindexing by
    impression_id there fans out combinatorially (200K x 200K matches, ~40GB), so
    this sorts+groups directly and zips the result into a dict positionally instead
    of merging on that non-unique key. The sentinel rows collapse to one arbitrary
    session_position in the returned dict (a pre-existing data quirk, not introduced
    here) -- harmless for the ~13.3M genuinely-unique impressions."""
    raw = pd.read_parquet(behaviors_path, columns=["impression_id", "user_id", "impression_time", "session_id"])
    raw = raw.rename(columns={"impression_time": "timestamp"})
    df = raw.sort_values(["user_id", "session_id", "timestamp"])
    position = df.groupby(["user_id", "session_id"]).cumcount().to_numpy()
    return dict(zip(df["impression_id"].to_numpy(), position))


def _score_chunk(
    impression_ids: list, user_ids: list, ref_times: list, histories: list, candidates: list[list],
    ctx: dict, is_ebnerd: bool, session_pos_by_impid: dict,
    freshness_lookup: dict, fallback_fresh: float,
    ebnerd_history_by_user: dict | None = None,
) -> list[list[int]]:
    """One rank list (1-indexed, aligned to `candidates` order) per impression."""
    bm25, emb = ctx["bm25"], ctx["emb"]
    article_vecs, id_to_row = ctx["article_vecs"], ctx["id_to_row"]
    id_to_text, id_to_category = ctx["id_to_text"], ctx["id_to_category"]
    popularity, model, feature_cols = ctx["popularity"], ctx["model"], ctx["feature_cols"]
    dim = article_vecs.shape[1]

    row_ctx = []
    bm25_queries, embed_queries = [], []
    for imp_id, user_id, ref_time, hist in zip(impression_ids, user_ids, ref_times, histories):
        if is_ebnerd:
            ids_full, times_full, reads_full, scrolls_full = ebnerd_history_by_user.get(
                user_id, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype="datetime64[us]"),
                          np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)))
            window_ids = ids_full[-config.FEATURE_HISTORY_WINDOW:]
            window_times = times_full[-config.FEATURE_HISTORY_WINDOW:]
            weights_full = bf.time_decay_weights(window_times, ref_time, config.HIST_HALF_LIFE_HOURS)
            hist_len = int(len(ids_full))
            avg_read = bf.safe_nanmean(reads_full)
            avg_scroll = bf.safe_nanmean(scrolls_full)
        else:
            hist_ids = [] if hist is None or len(hist) == 0 else list(hist)
            window_ids = np.asarray(hist_ids[-config.FEATURE_HISTORY_WINDOW:])
            weights_full = bf.rank_decay_weights(len(window_ids), config.HIST_DECAY_RANK)
            hist_len = len(hist_ids)
            avg_read = avg_scroll = None

        window_cats = [id_to_category.get(a) for a in window_ids]
        cat_counts = {}
        for c in window_cats:
            if c is not None:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        n_cat_known = sum(cat_counts.values())
        top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None

        cat_weight: dict = {}
        total_w = float(weights_full.sum()) if len(weights_full) else 0.0
        for cat, w in zip(window_cats, weights_full):
            if cat is not None:
                cat_weight[cat] = cat_weight.get(cat, 0.0) + float(w)

        emb_mask = np.array([a in id_to_row for a in window_ids], dtype=bool) if len(window_ids) else np.zeros(0, dtype=bool)
        hist_rows = [id_to_row[a] for a, m in zip(window_ids, emb_mask) if m]
        weights_emb = weights_full[emb_mask] if len(weights_full) else weights_full
        hist_vec = bf.weighted_mean_vec(hist_rows, weights_emb, article_vecs)

        bm25_queries.append(" ".join(id_to_text.get(a, "") for a in window_ids))
        embed_queries.append(article_vecs[hist_rows].mean(axis=0) if hist_rows else np.zeros(dim, dtype=np.float32))

        row_ctx.append({
            "imp_id": imp_id, "ref_time": ref_time, "sess_pos": session_pos_by_impid.get(imp_id, 0),
            "hist_len": hist_len, "avg_read": avg_read, "avg_scroll": avg_scroll,
            "cat_counts": cat_counts, "n_cat_known": n_cat_known, "top_cat": top_cat,
            "cat_weight": cat_weight, "total_w": total_w, "hist_vec": hist_vec,
        })

    embed_queries_mat = np.stack(embed_queries)
    bm25_cand_scores = bm25.batch_score_candidates(bm25_queries, candidates)
    embed_cand_scores = emb.batch_score_candidates(embed_queries_mat, candidates)

    # Feature rows for every multi-candidate impression in this sub-batch go into ONE
    # predict_proba call (spans[i] = that impression's slice), rather than one call per
    # impression -- on a GPU-trained model each call is a host->device round trip, and
    # there are millions of impressions in the Codabench test sets.
    all_feat_rows: list[list[float]] = []
    spans: list[tuple[int, int] | None] = []
    for ctx_row, cands, b_s, e_s in zip(row_ctx, candidates, bm25_cand_scores, embed_cand_scores):
        if len(cands) <= 1:
            spans.append(None)
            continue

        fused = hybrid_score(b_s, e_s)
        retrieval_rank = np.argsort(-fused)  # candidate_position: rank under A1's own retrieval score
        pos_by_cand = {cands[r]: p for p, r in enumerate(retrieval_rank)}

        cat_counts, n_cat_known, top_cat = ctx_row["cat_counts"], ctx_row["n_cat_known"], ctx_row["top_cat"]
        cat_weight, total_w, hist_vec = ctx_row["cat_weight"], ctx_row["total_w"], ctx_row["hist_vec"]
        ref_time = ctx_row["ref_time"]

        span_start = len(all_feat_rows)
        for cand in cands:
            cand_row = id_to_row.get(cand)
            cand_vec = article_vecs[cand_row] if cand_row is not None else None
            cand_cat = id_to_category.get(cand)

            if is_ebnerd:
                pub = freshness_lookup.get(cand)
                fresh = max((ref_time - pub) / np.timedelta64(1, "h"), 0.0) if pub is not None else fallback_fresh
            else:
                t0_first = freshness_lookup.get(cand)
                fresh = max((ref_time - t0_first) / np.timedelta64(1, "h"), 0.0) if t0_first is not None else fallback_fresh

            pos = pos_by_cand.get(cand, len(cands) - 1)
            feat = {
                "hist_len": ctx_row["hist_len"],
                "hist_category_match_frac": (cat_counts.get(cand_cat, 0) / n_cat_known) if (cand_cat is not None and n_cat_known > 0) else 0.0,
                "hist_category_match_recency": (cat_weight.get(cand_cat, 0.0) / total_w) if (cand_cat is not None and total_w > 0) else 0.0,
                "hist_embed_sim": bf.cosine(cand_vec, hist_vec),
                "session_position": ctx_row["sess_pos"],
                "popularity_log": float(np.log1p(popularity.get(cand, 0))),
                "freshness_hours": float(fresh),
                "category_match": int(cand_cat is not None and cand_cat == top_cat),
                "candidate_position": pos,
                "candidate_position_norm": pos / max(len(cands) - 1, 1),
            }
            if is_ebnerd:
                feat["user_avg_read_time"] = ctx_row["avg_read"]
                feat["user_avg_scroll_pct"] = ctx_row["avg_scroll"]
            all_feat_rows.append([0.0 if feat.get(c) is None else feat[c] for c in feature_cols])
        spans.append((span_start, len(all_feat_rows)))

    scores_all = (model.predict_proba(np.asarray(all_feat_rows, dtype=np.float32))[:, 1]
                  if all_feat_rows else np.zeros(0, dtype=np.float32))
    all_ranks: list[list[int]] = []
    for cands, span in zip(candidates, spans):
        if span is None:
            all_ranks.append([1] * len(cands))
        else:
            all_ranks.append(scores_to_ranks(scores_all[span[0]:span[1]]))
    return all_ranks


def _score_chunk_subbatched(impression_ids, user_ids, ref_times, histories, candidates, **kwargs) -> list[list[int]]:
    n = len(candidates)
    out: list[list[int]] = []
    for start in range(0, n, SUBBATCH_SIZE):
        end = min(start + SUBBATCH_SIZE, n)
        out.extend(_score_chunk(
            impression_ids[start:end], user_ids[start:end], ref_times[start:end],
            histories[start:end], candidates[start:end], **kwargs))
    return out


def run_mind(sample: int | None = None, skip_rows: int = 0, n_rows: int | None = None,
             out_txt: Path | None = None, zip_output: bool = True):
    """`skip_rows`/`n_rows`/`out_txt` let a caller process one contiguous row-range slice
    of the test file into its own partial output file (used to fan this out across
    parallel worker processes); defaults reproduce the original single-process behavior."""
    fs = config.FEATURE_STORE / "mind"
    ctx = _load_context(fs)
    first_appearance = bf.mind_first_appearance(fs)
    fallback_fresh = bf.fallback_freshness_hours(fs, first_appearance, config.RANDOM_SEED)
    test_path = config.MIND_TEST_DIR / "behaviors.tsv"
    session_pos = _session_positions_mind_test(test_path)

    if out_txt is None:
        out_txt = config.Q5_MIND_DIR / "prediction.txt"
    n_written = 0
    t0 = time.time()
    with open(out_txt, "w") as f:
        for chunk in mind_adapter.stream_behaviors_chunks(
                test_path, has_labels=False, chunk_rows=100_000, skip_rows=skip_rows, n_rows=n_rows):
            if sample is not None:
                chunk = chunk.head(sample)
            ranks = _score_chunk_subbatched(
                chunk["impression_id"].tolist(), chunk["user_id"].tolist(), chunk["timestamp"].tolist(),
                chunk["history"].tolist(), chunk["candidates"].tolist(),
                ctx=ctx, is_ebnerd=False, session_pos_by_impid=session_pos,
                freshness_lookup=first_appearance, fallback_fresh=fallback_fresh)
            for imp_id, r in zip(chunk["impression_id"], ranks):
                f.write(format_line(imp_id, r))
            n_written += len(chunk)
            print(f"[mind] {n_written:,} predictions written ({time.time()-t0:.0f}s elapsed)")
            del chunk, ranks
            gc.collect()
            if sample is not None:
                break

    if zip_output:
        write_submission_zip(out_txt, config.Q5_MIND_DIR / "prediction.zip", "prediction.txt")
    print(f"[mind] DONE: {n_written:,} predictions -> {out_txt}"
          f"{' and prediction.zip' if zip_output else ''} ({time.time()-t0:.0f}s total)")


def run_ebnerd(sample: int | None = None):
    fs = config.FEATURE_STORE / "ebnerd"
    ctx = _load_context(fs)
    publish_times = ebnerd_adapter.load_publish_times(
        config.EBNERD_SMALL_DIR / "articles.parquet", config.EBNERD_TESTSET_DIR / "articles.parquet")
    publish_time_by_article = dict(zip(publish_times["article_id"], publish_times["published_time"]))
    fallback_fresh = bf.fallback_freshness_hours(fs, publish_time_by_article, config.RANDOM_SEED)

    test_behaviors_path = config.EBNERD_TESTSET_DIR / "test" / "behaviors.parquet"
    test_history_path = config.EBNERD_TESTSET_DIR / "test" / "history.parquet"
    session_pos = _session_positions_ebnerd_test(test_behaviors_path)

    full_history = ebnerd_adapter.load_user_engagement_history(test_history_path)
    history_by_user = {}
    for user_id, ids, times, reads, scrolls in zip(
        full_history["user_id"], full_history["history"], full_history["impression_time_fixed"],
        full_history["read_time_fixed"], full_history["scroll_percentage_fixed"],
    ):
        history_by_user[user_id] = (
            np.asarray(ids, dtype=np.int64), np.asarray(times, dtype="datetime64[us]"),
            np.asarray(reads, dtype=np.float64), np.asarray(scrolls, dtype=np.float64),
        )
    del full_history
    gc.collect()

    out_txt = config.Q5_EBNERD_DIR / "predictions.txt"
    n_written = 0
    t0 = time.time()
    with open(out_txt, "w") as f:
        for chunk in ebnerd_adapter.stream_behaviors_chunks(test_behaviors_path, test_history_path, has_labels=False):
            if sample is not None:
                chunk = chunk.head(sample)
            ranks = _score_chunk_subbatched(
                chunk["impression_id"].tolist(), chunk["user_id"].tolist(), chunk["timestamp"].tolist(),
                [None] * len(chunk), chunk["candidates"].tolist(),
                ctx=ctx, is_ebnerd=True, session_pos_by_impid=session_pos,
                freshness_lookup=publish_time_by_article, fallback_fresh=fallback_fresh,
                ebnerd_history_by_user=history_by_user)
            for imp_id, r in zip(chunk["impression_id"], ranks):
                f.write(format_line(imp_id, r))
            n_written += len(chunk)
            print(f"[ebnerd] {n_written:,} predictions written ({time.time()-t0:.0f}s elapsed)")
            del chunk, ranks
            gc.collect()
            if sample is not None:
                break

    write_submission_zip(out_txt, config.Q5_EBNERD_DIR / "predictions.zip", "predictions.txt")
    print(f"[ebnerd] DONE: {n_written:,} predictions -> {out_txt} and predictions.zip "
          f"({time.time()-t0:.0f}s total)")


def main():
    parser = argparse.ArgumentParser(description="Q5: generate Codabench submission files.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None, help="only process first N rows of first chunk (smoke test)")
    parser.add_argument("--skip-rows", type=int, default=0,
                         help="mind only: skip this many data rows of the test file (parallel worker slicing)")
    parser.add_argument("--n-rows", type=int, default=None,
                         help="mind only: process only this many rows starting at --skip-rows")
    parser.add_argument("--out", type=str, default=None,
                         help="mind only: write predictions to this path instead of Q5_MIND/prediction.txt, "
                              "and skip building prediction.zip (used for parallel worker partial files)")
    args = parser.parse_args()
    config.Q5_MIND_DIR.mkdir(parents=True, exist_ok=True)
    config.Q5_EBNERD_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("mind", "all"):
        run_mind(sample=args.sample, skip_rows=args.skip_rows, n_rows=args.n_rows,
                  out_txt=Path(args.out) if args.out else None, zip_output=args.out is None)
    if args.dataset in ("ebnerd", "all"):
        run_ebnerd(sample=args.sample)


if __name__ == "__main__":
    main()
