"""BM25 lexical retrieval: inverted index + vectorized scoring.

Everything is expressed as sparse-matrix algebra (scipy.sparse, CPU/BLAS-backed) so it
scales to the 13.5M-row EB-NeRD test set and 2.37M-row MIND test set without a
Python-level loop per impression. See README / design note for why this stays on CPU
(sparse ops) while embedding similarity moves to GPU (dense ops) -- they suit different
hardware.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, min_df: int = 2):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.vectorizer: CountVectorizer | None = None
        self.W: sp.csr_matrix | None = None  # (n_docs, vocab) BM25-weighted matrix
        self.doc_ids: list = []
        self._id_to_row: dict = {}

    def fit(self, doc_ids: list, doc_texts: list[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self._id_to_row = {d: i for i, d in enumerate(self.doc_ids)}

        self.vectorizer = CountVectorizer(
            tokenizer=tokenize, preprocessor=lambda x: x, token_pattern=None,
            min_df=self.min_df, lowercase=False,
        )
        X = self.vectorizer.fit_transform(doc_texts).tocsr()  # raw term counts (n_docs, vocab)
        n_docs, _ = X.shape

        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs else 1.0
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        len_norm = self.k1 * (1 - self.b + self.b * doc_len / avgdl)  # (n_docs,)
        row_idx = np.repeat(np.arange(n_docs), np.diff(X.indptr))     # nnz -> owning row
        tf = X.data.astype(np.float32)
        idf = idf.astype(np.float32)
        len_norm = len_norm.astype(np.float32)

        w_data = idf[X.indices] * (tf * (self.k1 + 1)) / (tf + len_norm[row_idx])
        self.W = sp.csr_matrix((w_data, X.indices, X.indptr), shape=X.shape)
        return self

    def _query_matrix(self, texts: list[str]) -> sp.csr_matrix:
        Q = self.vectorizer.transform(texts).tocsr()
        Q.data = Q.data.astype(np.float32)
        return Q

    def top_k_batch(self, query_texts: list[str], k: int, batch_size: int = 100) -> list[list]:
        """Full-corpus top-K retrieval per query. Used for Q2 recall@K.

        Small batch_size is deliberate: (batch x n_docs) is densified per batch, and at
        MIND/EB-NeRD corpus scale (~125K docs) even a modest batch quickly reaches
        hundreds of MB -- see design note on this sandbox's tight (~4-7GB) RAM budget.
        """
        import gc
        results: list[list] = []
        n = len(query_texts)
        for start in range(0, n, batch_size):
            chunk = query_texts[start:start + batch_size]
            Q = self._query_matrix(chunk)                    # (b, vocab)
            scores = (Q @ self.W.T).toarray()                # (b, n_docs) dense chunk, float32
            for row in scores:
                if k < len(row):
                    idx = np.argpartition(-row, k)[:k]
                    idx = idx[np.argsort(-row[idx])]
                else:
                    idx = np.argsort(-row)
                results.append([self.doc_ids[i] for i in idx])
            del Q, scores
            if (start // batch_size) % 50 == 0:
                gc.collect()
        return results

    def batch_score_candidates(
        self, query_texts: list[str], candidate_id_lists: list[list], chunk_size: int = 2000,
    ) -> list[np.ndarray]:
        """Score only the given candidates per query (reranking, not full retrieval).

        Deliberately NOT the "gather both sides into one big sparse matrix, multiply,
        sum" vectorization: that requires replicating each query's sparse row once per
        candidate (a query can have hundreds of nonzeros, vs. ~20-40 for a doc), which at
        this corpus's scale (~125K docs, ~50K vocab) blew past this sandbox's ~4-7GB RAM
        budget under real Q4/Q5 traffic (73K+ impressions) even after batching the outer
        loop -- see design note. Per-impression sparse-matrix @ sparse-vector instead:
        candidates are typically only 5-250 rows, so each product is tiny and never
        replicates the query; only the (small) chunking below is for query-transform batching.
        """
        n = len(candidate_id_lists)
        out: list[np.ndarray] = [np.zeros(0) for _ in range(n)]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            Q = self._query_matrix(query_texts[start:end])  # (b, vocab), b small
            for i in range(end - start):
                cands = candidate_id_lists[start + i]
                cand_rows = [self._id_to_row.get(c, -1) for c in cands]
                valid = [r for r in cand_rows if r >= 0]
                scores = np.zeros(len(cands), dtype=np.float64)
                if valid:
                    sub_scores = np.asarray((self.W[valid] @ Q[i].T).todense()).ravel()
                    j = 0
                    for k, r in enumerate(cand_rows):
                        if r >= 0:
                            scores[k] = sub_scores[j]
                            j += 1
                out[start + i] = scores
        return out

    def score_candidates(self, query_text: str, candidate_ids: list) -> np.ndarray:
        return self.batch_score_candidates([query_text], [candidate_ids])[0]
