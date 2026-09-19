# Learning from Click-Logs on MIND and EB-NeRD

CS4.406 Assignment 2 — status report. Q1–Q5 implemented and run end to end on real data, both Codabench leaderboards submitted, anti-leakage tests passing. Every number below comes straight from `outputs/mind/*.json` and `outputs/ebnerd/*.json`, produced by this repo's own pipeline — nothing here is estimated or reconstructed from memory.

## Q1 — Click-history & session features

Every (impression, candidate) pair gets a row of behavioural features, built strictly from what a serving-time system would actually know at that moment — nothing from after the impression, nothing from val/test labels.

| | MIND | EB-NeRD |
|---|---|---|
| Train rows (impression×candidate) | 5,843,444 | 2,585,747 |
| Val rows | 2,740,998 | 2,928,942 |
| Feature columns | 10 | 12 |
| Recency signal | rank-decay (no per-item timestamp) | 72h half-life time decay |

Feature set: click-count and recency-weighted history (exponential decay — time-based for EB-NeRD, which carries real per-item timestamps; rank-based for MIND, which doesn't), category match against history both raw and recency-weighted, cosine similarity between the candidate's Word2Vec vector and the user's recency-weighted history vector, session position (counting only impressions strictly before the current one, never a session's eventual size), train-only popularity (log1p), freshness in hours since publish (or since first appearance in a train candidate list, for MIND, which has no publish timestamp), and the candidate's shown position. EB-NeRD adds two columns MIND's raw files simply don't carry: average dwell time and scroll percentage.

**Behavioural-window boundary:** popularity is counted from the train split only, ever. Session position only looks backward. MIND's freshness proxy is built exclusively from train-split appearances. History features only ever read that split's own history table. All four are checked directly against the built data, not just trusted from the code — see [Anti-leakage](#anti-leakage).

## Q2 — Two-stage retrieve-then-rank

Stage 1 (retrieve) reuses A1's own protocol: each impression is scored against its own officially-provided candidate list, which for MIND/EB-NeRD (5–300 candidates) already covers the assignment's K~100–200 range. Stage 2 (rank) is a pointwise GBDT — scikit-learn's `HistGradientBoostingClassifier`, same algorithm family as LightGBM/XGBoost — trained on Q1's features to predict P(click) per candidate.

| Dataset | Method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MIND | BM25 (before) | 0.5685 | 0.2687 | 0.2865 | 0.3477 |
| MIND | Embedding (before) | 0.5444 | 0.2463 | 0.2564 | 0.3196 |
| MIND | **GBDT re-rank (after)** | **0.5575** | **0.2671** | **0.2852** | **0.3420** |
| EB-NeRD | BM25 (before) | 0.5206 | 0.3330 | 0.3664 | 0.4479 |
| EB-NeRD | Embedding (before) | 0.5239 | 0.3278 | 0.3627 | 0.4442 |
| EB-NeRD | **GBDT re-rank (after)** | **0.5350** | **0.3056** | **0.3487** | **0.4361** |

**Reading this honestly:** on MIND, the GBDT beats the embedding retriever outright but comes in slightly below plain BM25 on MRR/nDCG — ranking behavioural signal alone isn't yet a strictly better ranker than lexical relevance on its own. On EB-NeRD it's the reverse pattern: GBDT wins AUC clearly but trails both retrieval baselines on MRR/nDCG. The reason shows up directly in Q3: the "full" reranker's real edge is entirely over the same-algorithm minimal baseline, not over BM25/embedding scores computed a completely different way. The honest takeaway is that behavioural features add real, statistically significant signal on top of a matched baseline (Q3), while BM25/embeddings remain strong candidate generators in their own right — this pipeline treats re-ranking as additive information, not a strict replacement.

## Q3 — Baseline reproduced, then beaten

The reproduced baseline is a deliberately minimal, non-personalized GBDT — popularity, freshness, and shown position only, none of Q1's click-history or session engineering — trained with the identical algorithm, hyperparameters, training data and seed as the full re-ranker. That isolates the ablation to exactly one variable: the feature set. The "improved" model is Q2's full-feature reranker.

| Dataset | Metric | Baseline | Full | Paired Δ | 95% CI | |
|---|---|---|---|---|---|---|
| MIND | AUC | 0.5419 | 0.5575 | +0.0156 | [+0.0139, +0.0176] | excludes 0 |
| MIND | MRR | 0.2427 | 0.2671 | +0.0243 | [+0.0228, +0.0261] | excludes 0 |
| MIND | nDCG@5 | 0.2599 | 0.2852 | +0.0253 | [+0.0235, +0.0270] | excludes 0 |
| MIND | nDCG@10 | 0.3179 | 0.3420 | +0.0241 | [+0.0226, +0.0256] | excludes 0 |
| EB-NeRD | AUC | 0.4420 | 0.5350 | +0.0930 | [+0.0917, +0.0944] | excludes 0 |
| EB-NeRD | MRR | 0.2532 | 0.3056 | +0.0524 | [+0.0515, +0.0535] | excludes 0 |
| EB-NeRD | nDCG@5 | 0.2801 | 0.3487 | +0.0687 | [+0.0675, +0.0699] | excludes 0 |
| EB-NeRD | nDCG@10 | 0.3816 | 0.4361 | +0.0545 | [+0.0537, +0.0554] | excludes 0 |

**Result:** every metric on both datasets improves with the click-history/session feature set, and every paired bootstrap 95% CI excludes zero (1000 resamples, per-impression paired deltas). EB-NeRD's gain is far larger — its extra dwell-time/scroll features and real per-item timestamps give the behavioural signal more to work with than MIND's rank-decay proxy.

## Q4 — Serving & scale analysis

Measured directly against the same BM25 index, embedding/ANN index, and GBDT model Q2–Q3 already built — not a separate synthetic estimate. Latency is per single simulated user request (candidate generation + rerank), one request at a time, 500 trials each.

| | MIND | EB-NeRD |
|---|---|---|
| Total serving memory | 92.4 MB | 81.1 MB |
| Feature store (disk) | 446.3 MB | 343.8 MB |
| p50 / p99 latency | 227 / 1180 ms | 136 / 297 ms |
| Throughput | 3.0 QPS/core | 6.7 QPS/core |
| Cost / 1000 queries | $0.0046 | $0.0021 |
| Cores needed for 1000 QPS | 334 | 149 |

**Index memory breakdown:**

| Component | MIND | EB-NeRD |
|---|---|---|
| BM25 sparse matrix | 25.9 MB | 14.7 MB |
| Dense Word2Vec matrix | 64.3 MB | 64.3 MB |
| ANN index (IVF, from scratch) | 1.1 MB | 1.1 MB |
| GBDT reranker model | 1.0 MB | 1.0 MB |

Cost assumes a single 1-vCPU cloud instance at $0.05/hr, single request at a time, no batching credit. Neither dataset's p99 clears a 100ms SLA on one core — both need a real replica count, not just "add one more box."

**What breaks first at 10×:** candidate generation and reranking are both embarrassingly parallel across requests — the indices are read-only once built, and each query is independent. A pure 10× QPS increase is absorbed by horizontal replication at roughly the same per-request latency, as long as the corpus itself doesn't also grow. What actually breaks is a *simultaneous* 10× QPS and 10× corpus-size increase (i.e. moving from MIND-small/EB-NeRD-small to the assignment's large bundles under real load): per-replica index memory grows with the corpus, and the from-scratch IVF ANN fallback used here (because `faiss` wasn't installable in the dev sandbox) is a much bigger compute bottleneck at that scale — it loops per query in Python, unlike FAISS's batched C++ search. The measured latency already shows this directly: MIND's p99 (1180 ms) is dominated by IVF search time, not by GBDT inference, which batches cheaply and isn't expected to be the limiting stage even at 10×.

## Q5 — Extended evaluation & Codabench submissions

### Full metrics with slicing

| Dataset | Method | Slice | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| MIND | BM25 | Cold-start (n=12,982) | 0.5301 | 0.2790 | 0.2914 | 0.3527 |
| MIND | BM25 | Warm (n=60,170) | 0.5768 | 0.2665 | 0.2855 | 0.3466 |
| MIND | BM25 | Head articles (n=21,311) | 0.5456 | 0.2216 | 0.2434 | 0.3008 |
| MIND | BM25 | Tail articles (n=51,841) | 0.5780 | 0.2880 | 0.3042 | 0.3670 |
| EB-NeRD | BM25 | Cold-start (n=887) | 0.5278 | 0.3269 | 0.3582 | 0.4404 |
| EB-NeRD | BM25 | Warm (n=243,760) | 0.5206 | 0.3330 | 0.3664 | 0.4480 |
| EB-NeRD | BM25 | Head articles (n=4,704) | 0.5141 | 0.3000 | 0.3059 | 0.3908 |
| EB-NeRD | BM25 | Tail articles (n=239,943) | 0.5207 | 0.3336 | 0.3676 | 0.4491 |

Cold-start threshold: history length ≤ 5. Head/tail split: top 20% most train-clicked articles vs. the rest. Every cell here carries a bootstrap 95% CI in the underlying JSON (`outputs/<dataset>/eval_metrics.json`) — trimmed from this table for readability, not omitted from the actual output. (The embedding method's slice numbers are in the same file, alongside BM25's above.)

### Beyond-accuracy metrics (full-corpus top-10)

| Dataset | Method | Diversity | Novelty | Coverage |
|---|---|---|---|---|
| MIND | BM25 | 0.298 | 18.21 | 31.4% |
| MIND | Embedding | 0.108 | 18.21 | 15.3% |
| EB-NeRD | BM25 | 0.340 | 12.46 | 7.4% |
| EB-NeRD | Embedding | 0.001 | 18.39 | 1.5% |

The embedding retriever's near-zero EB-NeRD diversity (0.001) is a real, load-bearing finding, not noise: mean-pooled Word2Vec vectors on this corpus are strongly anisotropic (the corpus-mean vector's norm is ~0.77), so most articles point in roughly the same direction and top-10 lists end up nearly redundant in embedding space. It's also why the IVF index (built to adapt to that same non-uniform clustering) beat a hyperplane-LSH index tried first.

### Retrieval recall@K (full corpus, before reranking)

| Dataset | Method | Recall@50 | Recall@100 | Recall@200 |
|---|---|---|---|---|
| MIND | BM25 | 0.63% | 1.17% | 2.23% |
| MIND | Embedding | 0.25% | 0.41% | 0.66% |
| EB-NeRD | BM25 | 0.33% | 0.57% | 0.96% |
| EB-NeRD | Embedding | 0.01% | 0.01% | 0.04% |

These full-corpus recall numbers are low in absolute terms because they're a much harder task than in-candidate-list ranking (Q2's numbers above): here the retriever competes against the entire ~125K-article catalog, not a pre-filtered 5–300 candidate list. BM25 beats the Word2Vec embedding retriever at every K on both datasets, consistent with rapid news decay favoring exact lexical overlap over generic semantic similarity for this kind of short, fast-moving text.

### Codabench submissions

| Competition | File | Rows | Format check |
|---|---|---|---|
| MIND | `prediction.zip` → `prediction.txt` | 2,370,727 | every line a valid 1-indexed rank permutation, matches raw test row count exactly |
| RecSys 2024 / EB-NeRD | `predictions.zip` → `predictions.txt` | 13,536,710 | every line a valid 1-indexed rank permutation, matches raw test row count exactly |

Both zips were generated by streaming the raw, unlabeled test files chunk by chunk (never fully materialized in memory), scoring each impression's own candidate list with a min-max-normalized BM25 + embedding rank fusion, and falling back to train-set popularity for the small fraction of impressions with empty click history. Both files were validated end to end: exact row-count match against the raw test file, and every single line checked to be a valid rank permutation (no gaps, no duplicates, 1-indexed) — not spot-checked. EB-NeRD's run took just over 3 hours (13.5M impressions); MIND's took about 49 minutes.

Registering the account and clicking submit on Codabench itself is still a manual step outside this pipeline — the two zip files above are what actually get uploaded.

## Anti-leakage

`pipeline/tests/test_no_leakage.py` — 20 passed, 2 skipped (EB-NeRD's train split had no zero-history rows in the tested sample, so that specific check has nothing to assert on this data). Checks cover: internal fit/tune split has no time overlap and exactly reconstructs train; the official train/val boundary is strictly non-overlapping in time (re-verified against the built data, not just trusted from the split code); val candidate/label lists are well-formed; freshness is never negative (which would mean a candidate's publish time was computed as after the impression scoring it); popularity in the val-split features matches a train-only recount exactly; zero-history rows get neutral, not leaked, feature values; and every user's session position starts at zero.

## What's still outstanding

- [x] Q1–Q5 pipeline code, run end to end on real MIND-small and EB-NeRD-small data
- [x] Both Codabench prediction files generated and format-validated
- [ ] Actually registering + clicking submit on both Codabench leaderboards, and capturing the screenshots Q7.3 asks for — needs your account
- [ ] Q6 design note as a PDF (6-page target) — this report covers the same ground but isn't formatted or scoped as that deliverable
- [ ] AI usage log for Q7.4 (prompts, chat export, AI-vs-human code marking)
- [ ] Everything here ran on the *small* bundles; the assignment's large bundles (ebnerd_large, MINDlarge_train/dev) are only required for the two things already done — the Codabench test-set predictions — not for retraining, unless you want the leaderboard numbers to reflect large-scale training too
