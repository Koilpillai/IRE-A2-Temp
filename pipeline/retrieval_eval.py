"""Q2 + Q3 -- Lexical (BM25) and Semantic (Word2Vec) candidate generation.

For a dataset's validation/dev split:
  1. Build a BM25 inverted index and a Word2Vec-embedding ANN index over the full
     article corpus.
  2. For each impression, build a query from the user's click history (BM25: concatenated
     text; embeddings: mean-pooled article vectors).
  3. Retrieve top-K candidates from the FULL corpus and report recall@{50,100,200}
     (Q2.4 / Q3.4) -- does the true clicked article appear in the open-corpus retrieval?
  4. Also score each impression's OWN provided candidate list (needed by Q4's AUC/MRR/
     nDCG, which require the officially-labeled candidates) and save a top-10 full-corpus
     list per impression (needed by Q4's diversity/novelty/coverage). Both are written to
     one results file so Q4 does not have to rebuild either index.

Usage:
    python pipeline/retrieval_eval.py --dataset mind
    python pipeline/retrieval_eval.py --dataset ebnerd
    python pipeline/retrieval_eval.py --dataset mind --sample 750   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.common.bm25 import BM25Index
from pipeline.common.embeddings import EmbeddingIndex, mean_pool_texts, train_word2vec
from pipeline.common.text import tokenize

HISTORY_WINDOW = 50  # cap on how many recent history articles feed the query (see design note)


def _build_queries(behaviors: pd.DataFrame, history_by_user: dict, id_to_text: dict, id_to_row: dict,
                    article_vecs: np.ndarray):
    """Per impression: BM25 query text + mean-pooled embedding query vector."""
    bm25_queries, embed_queries = [], []
    dim = article_vecs.shape[1]
    for user_id in behaviors["user_id"]:
        hist = history_by_user.get(user_id)
        hist = hist if hist is not None and len(hist) else []
        window = hist[-HISTORY_WINDOW:]
        bm25_queries.append(" ".join(id_to_text.get(a, "") for a in window))
        rows = [id_to_row[a] for a in window if a in id_to_row]
        embed_queries.append(article_vecs[rows].mean(axis=0) if rows else np.zeros(dim, dtype=np.float32))
    return bm25_queries, np.stack(embed_queries)


def recall_at_k(candidate_topk: list[list], labels_true: list[list], ks: list[int]) -> dict:
    """Fraction of impressions (with >=1 click) whose clicked article appears in top-K."""
    out = {k: [0, 0] for k in ks}  # k -> [hits, total]
    for topk, clicked in zip(candidate_topk, labels_true):
        if not clicked:
            continue
        topk_set_full = topk  # already sorted, longest K
        for k in ks:
            out[k][1] += 1
            if any(c in topk_set_full[:k] for c in clicked):
                out[k][0] += 1
    return {k: (hits / total if total else float("nan"), hits, total) for k, (hits, total) in out.items()}


def run(dataset: str, sample: int | None = None):
    fs = config.FEATURE_STORE / dataset
    out_dir = config.OUTPUTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_parquet(fs / "articles.parquet")
    val = pd.read_parquet(fs / "behaviors_val.parquet")
    if sample:
        val = val.sample(n=min(sample, len(val)), random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Look up history per user via a plain dict rather than merging it onto `val` as a
    # column -- a merge would duplicate each user's (possibly long) history list across
    # every one of their impression rows, which is exactly the pattern that inflated
    # ebnerd's build_pipeline peak RSS from ~0.3GB to ~3GB (see adapters/ebnerd.py). A
    # dict lookup during query construction reuses the same underlying list by reference.
    history_table = pd.read_parquet(fs / "user_history_val.parquet")
    history_by_user = dict(zip(history_table["user_id"], history_table["history"]))

    print(f"[{dataset}] articles={len(articles):,} val_impressions={len(val):,}")
    id_to_text = dict(zip(articles["article_id"], articles["text_lexical"]))

    # --- Q2: BM25 index ---
    t0 = time.time()
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B).fit(
        articles["article_id"].tolist(), articles["text_lexical"].tolist())
    print(f"[{dataset}] BM25 index built in {time.time()-t0:.1f}s "
          f"(vocab={len(bm25.vectorizer.vocabulary_):,})")

    # --- Q3: Word2Vec embedding index ---
    t0 = time.time()
    tokenized = [tokenize(t) for t in articles["text_lexical"].tolist()]
    w2v = train_word2vec(tokenized, vector_size=config.W2V_DIM, window=config.W2V_WINDOW,
                          min_count=config.W2V_MIN_COUNT, epochs=config.W2V_EPOCHS,
                          workers=config.W2V_WORKERS, seed=config.RANDOM_SEED)
    article_vecs = mean_pool_texts(w2v, tokenized)
    emb = EmbeddingIndex(articles["article_id"].tolist(), article_vecs)
    print(f"[{dataset}] Word2Vec + embedding index built in {time.time()-t0:.1f}s "
          f"(dim={config.W2V_DIM}, faiss={emb.faiss_index is not None}, torch={emb.mat_t is not None})")

    # Persist to the feature store (article embeddings are an explicit Q1.4 requirement)
    # so Q4/Q5 reuse these vectors instead of re-training Word2Vec from scratch.
    np.save(fs / "article_embeddings.npy", article_vecs)
    w2v.save(str(fs / "word2vec.model"))

    id_to_row = {a: i for i, a in enumerate(articles["article_id"])}
    bm25_queries, embed_queries = _build_queries(val, history_by_user, id_to_text, id_to_row, article_vecs)

    # --- within-impression candidate scoring (feeds Q4 AUC/MRR/nDCG) ---
    t0 = time.time()
    candidates = val["candidates"].tolist()
    bm25_cand_scores = bm25.batch_score_candidates(bm25_queries, candidates)
    embed_cand_scores = emb.batch_score_candidates(embed_queries, candidates)
    print(f"[{dataset}] within-impression scoring done in {time.time()-t0:.1f}s")

    # --- full-corpus top-K retrieval (feeds Q2/Q3 recall@K and Q4 diversity/novelty/coverage) ---
    max_k = max(config.RECALL_KS)
    t0 = time.time()
    bm25_topk = bm25.top_k_batch(bm25_queries, k=max_k)
    embed_topk = emb.top_k_batch(embed_queries, k=max_k)
    print(f"[{dataset}] full-corpus top-{max_k} retrieval done in {time.time()-t0:.1f}s")

    clicked_lists = [[c for c, l in zip(cands, labels) if l == 1]
                      for cands, labels in zip(val["candidates"], val["labels"])]

    bm25_recall = recall_at_k(bm25_topk, clicked_lists, config.RECALL_KS)
    embed_recall = recall_at_k(embed_topk, clicked_lists, config.RECALL_KS)

    print(f"\n[{dataset}] Recall@K -- BM25 (lexical) vs Word2Vec (semantic), full corpus (n={len(articles):,})")
    print(f"{'K':>6} | {'BM25':>18} | {'Embedding':>18}")
    for k in config.RECALL_KS:
        b = bm25_recall[k]; e = embed_recall[k]
        print(f"{k:>6} | {b[0]:>7.4f} ({b[1]}/{b[2]:>5}) | {e[0]:>7.4f} ({e[1]}/{e[2]:>5})")

    results = {
        "dataset": dataset, "n_articles": len(articles), "n_val": len(val),
        "history_window": HISTORY_WINDOW, "w2v_dim": config.W2V_DIM,
        "bm25_recall_at_k": {k: {"recall": v[0], "hits": v[1], "total": v[2]} for k, v in bm25_recall.items()},
        "embedding_recall_at_k": {k: {"recall": v[0], "hits": v[1], "total": v[2]} for k, v in embed_recall.items()},
    }
    with open(out_dir / "recall_at_k.json", "w") as f:
        json.dump(results, f, indent=2)

    # save everything Q4 needs, so it never has to rebuild an index
    save_df = pd.DataFrame({
        "impression_id": val["impression_id"], "user_id": val["user_id"],
        "history_len": val["history_len"], "candidates": val["candidates"], "labels": val["labels"],
        "bm25_scores": [s.tolist() for s in bm25_cand_scores],
        "embed_scores": [s.tolist() for s in embed_cand_scores],
        "bm25_top10": [t[:10] for t in bm25_topk],
        "embed_top10": [t[:10] for t in embed_topk],
    })
    save_df.to_parquet(out_dir / "retrieval_results_val.parquet", index=False)
    print(f"[{dataset}] saved outputs/{dataset}/retrieval_results_val.parquet ({len(save_df):,} rows) "
          f"and recall_at_k.json")


def main():
    parser = argparse.ArgumentParser(description="Q2+Q3: BM25 and embedding candidate generation.")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None, help="subsample val impressions (smoke test)")
    args = parser.parse_args()
    targets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        run(name, sample=args.sample)


if __name__ == "__main__":
    main()
