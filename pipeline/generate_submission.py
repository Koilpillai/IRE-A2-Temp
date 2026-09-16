"""Q5 -- Generate Codabench prediction files for both leaderboards.

Reuses the BM25 + Word2Vec indices already built by Q2/Q3 (rebuilding BM25 is cheap;
embeddings are loaded from the feature store instead of re-training). Streams the raw,
unlabeled Codabench test files chunk-by-chunk (never fully materialized), scores each
impression's own candidate list with a simple min-max rank-fusion of BM25 + embedding
similarity, and falls back to train-set popularity for the ~2% of impressions with empty
click history (no query can be built).

Output formats (verified against each competition's own "Submission Guidelines" page,
screenshotted into Q5_MIND/ and "Q5_EB-NeRd/"):
  MIND:    zip containing exactly `prediction.txt`  -> Q5_MIND/
  EB-NeRD: zip containing exactly `predictions.txt` -> Q5_EB-NeRd/

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

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.adapters import ebnerd as ebnerd_adapter
from pipeline.adapters import mind as mind_adapter
from pipeline.common.bm25 import BM25Index
from pipeline.common.embeddings import EmbeddingIndex
from pipeline.common.popularity import train_popularity as _train_popularity
from pipeline.common.submission import format_line, hybrid_score, scores_to_ranks, write_submission_zip

HISTORY_WINDOW = 50
SUBBATCH_SIZE = 20_000  # see _score_chunk_subbatched: bounds the BM25 query-string list's memory


def _score_chunk(chunk_histories: list[list], candidates: list[list], bm25: BM25Index,
                  emb: EmbeddingIndex, id_to_text: dict, id_to_row: dict, article_vecs: np.ndarray,
                  popularity: dict) -> list[list[int]]:
    """Returns one rank list (1-indexed, aligned to `candidates` order) per impression."""
    dim = article_vecs.shape[1]
    bm25_queries, embed_queries, empty_history_mask = [], [], []
    for hist in chunk_histories:
        # NOTE: `hist` may be a numpy array (parquet list-columns round-trip that way via
        # pandas/pyarrow, even for files we didn't write ourselves) -- `hist or []` would
        # raise "truth value of an array is ambiguous" on any non-empty array, so check
        # length explicitly instead of relying on truthiness.
        window = [] if hist is None or len(hist) == 0 else list(hist)[-HISTORY_WINDOW:]
        empty_history_mask.append(len(window) == 0)
        bm25_queries.append(" ".join(id_to_text.get(a, "") for a in window))
        rows = [id_to_row[a] for a in window if a in id_to_row]
        embed_queries.append(article_vecs[rows].mean(axis=0) if rows else np.zeros(dim, dtype=np.float32))
    embed_queries = np.stack(embed_queries) if embed_queries else np.zeros((0, dim), dtype=np.float32)

    bm25_scores = bm25.batch_score_candidates(bm25_queries, candidates)
    embed_scores = emb.batch_score_candidates(embed_queries, candidates)

    all_ranks = []
    for cands, b_s, e_s, is_cold in zip(candidates, bm25_scores, embed_scores, empty_history_mask):
        if is_cold or len(cands) <= 1:
            scores = np.array([popularity.get(c, 0) for c in cands], dtype=np.float64)
        else:
            scores = hybrid_score(b_s, e_s)
        all_ranks.append(scores_to_ranks(scores))
    return all_ranks


def _score_chunk_subbatched(chunk_histories: list[list], candidates: list[list], **kwargs) -> list[list[int]]:
    """`_score_chunk`, but in SUBBATCH_SIZE-row slices.

    `_score_chunk` builds one BM25 query STRING per impression (up to 50 history
    articles' title+text each) for the whole input before any internal batching runs --
    fine at MIND's 100K-row outer chunk size, but EB-NeRD's parquet row-groups are
    ~265K rows, and that alone (~1.3GB+ of concatenated query strings resident at once,
    on top of everything else already live) was enough to OOM-kill a run that got
    through 3 full outer chunks without incident on the smaller dataset. Slicing here
    keeps peak memory bounded by SUBBATCH_SIZE regardless of what the caller's own
    chunk size happens to be.
    """
    n = len(candidates)
    out: list[list[int]] = []
    for start in range(0, n, SUBBATCH_SIZE):
        end = min(start + SUBBATCH_SIZE, n)
        out.extend(_score_chunk(chunk_histories[start:end], candidates[start:end], **kwargs))
    return out


def _build_indices(fs: Path):
    articles = pd.read_parquet(fs / "articles.parquet")
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B).fit(
        articles["article_id"].tolist(), articles["text_lexical"].tolist())
    article_vecs = np.load(fs / "article_embeddings.npy")
    emb = EmbeddingIndex(articles["article_id"].tolist(), article_vecs)
    id_to_text = dict(zip(articles["article_id"], articles["text_lexical"]))
    id_to_row = {a: i for i, a in enumerate(articles["article_id"])}
    return bm25, emb, article_vecs, id_to_text, id_to_row


def run_mind(sample: int | None = None):
    fs = config.FEATURE_STORE / "mind"
    bm25, emb, article_vecs, id_to_text, id_to_row = _build_indices(fs)
    popularity = _train_popularity(fs)

    out_txt = config.Q5_MIND_DIR / "prediction.txt"
    n_written = 0
    t0 = time.time()
    with open(out_txt, "w") as f:
        for chunk in mind_adapter.stream_behaviors_chunks(
                config.MIND_TEST_DIR / "behaviors.tsv", has_labels=False, chunk_rows=100_000):
            if sample is not None:
                chunk = chunk.head(sample)
            ranks = _score_chunk_subbatched(
                chunk["history"].tolist(), chunk["candidates"].tolist(), bm25=bm25, emb=emb,
                id_to_text=id_to_text, id_to_row=id_to_row, article_vecs=article_vecs, popularity=popularity)
            for imp_id, r in zip(chunk["impression_id"], ranks):
                f.write(format_line(imp_id, r))
            n_written += len(chunk)
            print(f"[mind] {n_written:,} predictions written ({time.time()-t0:.0f}s elapsed)")
            del chunk, ranks
            gc.collect()
            if sample is not None:
                break

    write_submission_zip(out_txt, config.Q5_MIND_DIR / "prediction.zip", "prediction.txt")
    print(f"[mind] DONE: {n_written:,} predictions -> {out_txt} and prediction.zip "
          f"({time.time()-t0:.0f}s total)")


def run_ebnerd(sample: int | None = None):
    fs = config.FEATURE_STORE / "ebnerd"
    bm25, emb, article_vecs, id_to_text, id_to_row = _build_indices(fs)
    popularity = _train_popularity(fs)

    test_history_path = config.EBNERD_TESTSET_DIR / "test" / "history.parquet"
    history_table = ebnerd_adapter.load_user_history(test_history_path)
    history_by_user = dict(zip(history_table["user_id"], history_table["history"]))
    del history_table
    gc.collect()

    out_txt = config.Q5_EBNERD_DIR / "predictions.txt"
    n_written = 0
    t0 = time.time()
    with open(out_txt, "w") as f:
        for chunk in ebnerd_adapter.stream_behaviors_chunks(
                config.EBNERD_TESTSET_DIR / "test" / "behaviors.parquet", test_history_path, has_labels=False):
            if sample is not None:
                chunk = chunk.head(sample)
            histories = [history_by_user.get(u) for u in chunk["user_id"]]
            ranks = _score_chunk_subbatched(
                histories, chunk["candidates"].tolist(), bm25=bm25, emb=emb, id_to_text=id_to_text,
                id_to_row=id_to_row, article_vecs=article_vecs, popularity=popularity)
            for imp_id, r in zip(chunk["impression_id"], ranks):
                f.write(format_line(imp_id, r))
            n_written += len(chunk)
            print(f"[ebnerd] {n_written:,} predictions written ({time.time()-t0:.0f}s elapsed)")
            del chunk, ranks, histories
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
    args = parser.parse_args()
    config.Q5_MIND_DIR.mkdir(parents=True, exist_ok=True)
    config.Q5_EBNERD_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("mind", "all"):
        run_mind(sample=args.sample)
    if args.dataset in ("ebnerd", "all"):
        run_ebnerd(sample=args.sample)


if __name__ == "__main__":
    main()
