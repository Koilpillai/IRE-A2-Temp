# Concept & Parameter Reference — Assignment 2: Learning from Click-Logs on MIND and EB-NeRD

**CS4.406: Information Retrieval & Extraction**

This document is a self-contained explainer for every concept, algorithm, and parameter value used in this repository's two-stage news-recommendation pipeline. It complements `report.md` (which presents results and findings) by explaining *why* each technique was used, *how* it works mathematically, and *what* every tunable number is actually set to and why. All parameter values below were read directly from `pipeline/config.py`, the pipeline modules, and the JSON files under `outputs/` — nothing is approximated unless explicitly marked "measured" or "computed."

---

## Table of Contents

1. [Problem Setting](#1-problem-setting)
2. [Datasets & Corpus Statistics](#2-datasets--corpus-statistics)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Text Processing](#4-text-processing)
5. [Stage 1a — Lexical Retrieval: BM25](#5-stage-1a--lexical-retrieval-bm25)
6. [Stage 1b — Semantic Retrieval: Word2Vec + ANN Search](#6-stage-1b--semantic-retrieval-word2vec--ann-search)
7. [Behavioural Feature Engineering (Click-Logs)](#7-behavioural-feature-engineering-click-logs)
8. [Stage 2 — GBDT Re-Ranker](#8-stage-2--gbdt-re-ranker)
9. [Baseline, Ablation & Statistical Significance](#9-baseline-ablation--statistical-significance)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Beyond-Accuracy Metrics](#11-beyond-accuracy-metrics)
12. [Slicing Analysis](#12-slicing-analysis)
13. [Anti-Leakage / Temporal Splitting](#13-anti-leakage--temporal-splitting)
14. [Serving & Scale Analysis](#14-serving--scale-analysis)
15. [Codabench Submission Format](#15-codabench-submission-format)
16. [Complete Parameter Reference Table](#16-complete-parameter-reference-table)
17. [Complete Results Reference Table](#17-complete-results-reference-table)
18. [Glossary](#18-glossary)

---

## 1. Problem Setting

The assignment is **news recommendation**: given a user's past click history and a list of candidate articles shown in an "impression" (a single page-load event), predict which candidate(s) the user will click. This is framed as a **two-stage retrieve-then-rank** problem, the standard industrial pattern for large-catalog recommendation/search:

- **Stage 1 (Retrieval)**: cheaply narrow millions of articles down to a small candidate set (tens to hundreds) using lexical and/or semantic similarity.
- **Stage 2 (Ranking)**: expensively score only that small candidate set with a richer model that can use behavioural/contextual signals a retrieval index can't easily support.

Assignment 1 built Stage 1 (BM25 + Word2Vec). Assignment 2 (this repo) adds click-log-derived behavioural features and a GBDT re-ranker for Stage 2, then evaluates the whole pipeline for accuracy, statistical significance, and production serving cost.

Two datasets are used throughout so every technique is validated on two different logging regimes:

| | **MIND** (Microsoft News Dataset) | **EB-NeRD** (Ekstra Bladet) |
|---|---|---|
| Language | English | Danish |
| Per-click timestamps in history | ❌ No (order only) | ✅ Yes |
| Dwell time / scroll depth | ❌ No | ✅ Yes |
| Session IDs | ❌ No (heuristic reconstruction) | ✅ Yes (native) |
| Article publish timestamps | ❌ No (proxy used) | ✅ Yes |

These structural differences are why the two datasets need slightly different feature-engineering logic (Section 7) and why EB-NeRD benefits far more from personalization (Section 9).

---

## 2. Datasets & Corpus Statistics

All numbers below are **computed directly from the processed feature store** (`feature_store/{mind,ebnerd}/articles.parquet` and `behaviors_{train,val}.parquet`), not estimated. Token counts use the exact same `tokenize()` function BM25 and Word2Vec both consume (Section 4).

### 2.1 Article corpus statistics

| Metric | MIND | EB-NeRD |
|---|---:|---:|
| Number of articles (train+dev+test union) | 125,590 | 125,541 |
| **Average document length (tokens, post-stopword-removal)** | **30.78** | **15.88** |
| Median document length (tokens) | 25 | 16 |
| Std. dev. document length (tokens) | 17.21 | 5.76 |
| Min / Max document length (tokens) | 2 / 372 | 0 / 108 |
| **Average document length (characters, raw text)** | **279.66** | **148.30** |
| Distinct vocabulary (all tokens seen) | 69,237 | 118,795 |
| BM25 vocabulary (after `min_df=2` filter) | 49,986 | 58,946 |
| Number of categories | 18 | 33 |
| Text fields concatenated into `text_lexical` | `title + " " + abstract` | `title + " " + subtitle` |

Danish EB-NeRD articles are noticeably shorter (title + subtitle only) than MIND's title + abstract, but EB-NeRD's raw vocabulary is larger — a mix of Danish morphology (more inflected surface forms per lemma) and 33 fine-grained categories vs. MIND's 18.

### 2.2 Impression / behavioural log statistics

| Metric | MIND (train) | MIND (val) | EB-NeRD (train) | EB-NeRD (val) |
|---|---:|---:|---:|---:|
| Number of impressions | 156,965 | 73,152 | 232,887 | 244,647 |
| **Average candidates per impression** | 37.23 | 37.47 | 11.10 | 11.97 |
| Min / Max candidates per impression | 2 / 299 | 2 / 295 | 5 / 100 | 5 / 99 |
| **Average user history length** | 32.54 | 32.30 | 306.81 | 277.71 |
| Median user history length | 19 | 19 | 253 | 216 |
| **Average click-through rate per impression** | 10.85% | 10.01% | 12.09% | 11.74% |
| Train time range | 2019-11-09 → 2019-11-14 | — | 2023-05-18 → 2023-05-25 | — |
| Internal tune cutoff | 2019-11-14 07:28:44 | — | 2023-05-24 05:47:36 | — |
| Validation time range | — | 2019-11-15 (1 day) | — | 2023-05-25 → 2023-06-01 (7 days) |
| Internal fit / tune split | 133,422 / 23,543 | — | 197,955 / 34,932 | — |

EB-NeRD users have ~9x longer histories than MIND users on average (306.8 vs 32.5 articles), but each individual impression shows far fewer candidates (11 vs 37) — EB-NeRD's candidate lists are pre-filtered/curated per impression, while MIND exposes a wider slate per impression. This directly explains why EB-NeRD's behavioural (history-based) features carry more signal (Section 9): more history to average over, computed against real per-click timestamps rather than MIND's order-only proxy.

---

## 3. Pipeline Architecture

```
Raw files (TSV / Parquet)
        │  pipeline/adapters/{mind,ebnerd}.py
        ▼
Unified schema  →  feature_store/<dataset>/{articles, behaviors_*, user_history_*}.parquet
        │  pipeline/build_pipeline.py  (Q1: ingestion, temporal split)
        ▼
┌───────────────────────────────┬────────────────────────────────┐
│ Stage 1a: BM25 index          │ Stage 1b: Word2Vec + ANN index  │
│ pipeline/common/bm25.py       │ pipeline/common/embeddings.py   │
└───────────────────────────────┴────────────────────────────────┘
        │  pipeline/retrieval_eval.py  (candidate scores + recall@K)
        ▼
Behavioural feature engineering  →  behavioral_features_{train,val}.parquet
        │  pipeline/features.py  (Q1: click-history, session, article, position features)
        ▼
Stage 2: GBDT re-ranker (XGBoost, GPU)
        │  pipeline/rerank.py (Q2)  +  pipeline/ablation.py (Q3, baseline vs. full + paired bootstrap CI)
        ▼
Evaluation  →  outputs/<dataset>/{eval_metrics, rerank_metrics, ablation, recall_at_k}.json
        │  pipeline/evaluate.py (Q4/Q5: AUC/MRR/nDCG, slices, diversity/novelty/coverage)
        ▼
Serving & scale benchmark  →  outputs/<dataset>/serving_scale.json
        │  pipeline/serving_scale.py (Q4: index memory, p50/p95/p99 latency, cost/QPS, 10x argument)
        ▼
Codabench submission  →  Q5_MIND/prediction.zip, Q5_EB-NeRd/predictions.zip
        │  pipeline/generate_submission.py + pipeline/common/submission.py
        ▼
Anti-leakage verification  →  pipeline/tests/test_no_leakage.py (pytest, Q9)
```

Everything is orchestrated by the `Makefile` (`make data`, `make retrieve`, `make evaluate`, `make rerank`, `make serving_scale`, `make submit`, `make test`, or `make all`).

---

## 4. Text Processing

**File**: `pipeline/common/text.py`

Both BM25 and Word2Vec consume the exact same `tokenize()` function so that any performance difference between lexical and semantic retrieval reflects the *retrieval method*, not inconsistent preprocessing.

**Steps** (`tokenize(text, remove_stopwords=True)`):
1. **Unicode NFKC normalization** + lowercasing — collapses accented/composed characters (important for Danish æ/ø/å) to a canonical form.
2. **Regex tokenization**: `[^\W\d_]+` (Unicode-aware) — keeps runs of letters only, discards digits and punctuation, correctly handles non-ASCII letters.
3. **Stopword removal**: a small, hand-picked closed-class list — 60 English words + 45 Danish words (articles, pronouns, auxiliaries, conjunctions). Deliberately short: BM25's IDF term already down-weights frequent words statistically, so this only needs to strip the most extreme noise, not perform full linguistic stopword removal.
4. **Length filter**: tokens of length ≤ 1 are dropped (single letters carry no lexical signal).

This single shared tokenizer is why the "average document length" numbers in Section 2.1 differ from raw whitespace-token counts — they already reflect stopword removal and the letters-only regex.

---

## 5. Stage 1a — Lexical Retrieval: BM25

**File**: `pipeline/common/bm25.py`

BM25 (Okapi BM25) is a term-frequency/inverse-document-frequency ranking function — the strongest classical baseline for keyword search, and still hard to beat for short, entity-dense news text where users' click history vocabulary overlaps directly with target-article vocabulary.

### 5.1 Formula

For a query $Q$ (built from the user's history text) and document $D$:

$$\text{score}(Q, D) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where:
- $f(t, D)$ = raw term frequency of term $t$ in document $D$
- $|D|$ = document length in tokens; $\text{avgdl}$ = mean document length across the corpus
- $\text{IDF}(t) = \ln\left(1 + \dfrac{N - n_t + 0.5}{n_t + 0.5}\right)$, with $N$ = corpus size, $n_t$ = number of documents containing $t$ (the "+1" smoothed variant, always non-negative)
- $k_1$ controls term-frequency saturation (how quickly repeated occurrences stop adding score); $b$ controls document-length normalization strength (0 = no normalization, 1 = full normalization)

### 5.2 Implementation notes

- Implemented as **sparse-matrix algebra** (`scipy.sparse`, CSR format) — the entire corpus is pre-weighted into one `(n_docs × vocab)` BM25-weighted matrix at index-build time; a query becomes a sparse dot-product against it. This scales to the 13.5M-row EB-NeRD test set and 2.37M-row MIND test set without a Python-level loop per impression.
- **Two scoring paths**: `top_k_batch` (dense, full-corpus top-K — used for open-corpus Recall@K) vs. `batch_score_candidates` (sparse, scores only a given small candidate list — used for the officially-labeled AUC/MRR/nDCG evaluation). The two paths are separated because densifying `(batch × 125K docs)` for every batch would blow the sandbox's ~4–7GB RAM budget if applied to the small-candidate-list case.
- Minimum document frequency filter `min_df=2` (a term must appear in ≥ 2 documents to enter the vocabulary) — this is why the BM25 vocabulary (49,986 / 58,946 terms) is smaller than the raw distinct-token count (69,237 / 118,795 terms) from Section 2.1.

### 5.3 Parameters used

| Parameter | Value | Meaning |
|---|---:|---|
| `k1` | 1.5 | Standard Okapi BM25 default; moderate TF saturation |
| `b` | 0.75 | Standard Okapi BM25 default; full-strength length normalization |
| `min_df` | 2 | Drop terms occurring in fewer than 2 documents |

---

## 6. Stage 1b — Semantic Retrieval: Word2Vec + ANN Search

**File**: `pipeline/common/embeddings.py`

### 6.1 Word2Vec (skip-gram)

Word2Vec is trained **from scratch, per-corpus, fully offline** (via `gensim`) — one of the two semantic techniques the assignment names explicitly (Word2Vec / BERT / XLM-RoBERTa). Skip-gram (`sg=1`) is used, which predicts context words from a center word — generally stronger than CBOW for smaller/mid-size corpora and rarer terms, both relevant here (news vocabulary is long-tailed with many proper nouns/entities).

**Article vectors**: each article's word vectors are **mean-pooled** into a single dense vector (`mean_pool_texts`). Articles with zero in-vocabulary tokens fall back to the corpus mean vector rather than a zero vector (avoids spuriously high cosine similarity to unrelated all-zero vectors).

### 6.2 Parameters used

| Parameter | Value | Meaning |
|---|---:|---|
| `vector_size` (`W2V_DIM`) | 128 | Embedding dimensionality |
| `window` | 8 | Context window (words on each side) |
| `min_count` | 2 | Ignore words occurring < 2 times |
| `epochs` | 10 | Training passes over the corpus |
| `workers` | 20 | Parallel training threads |
| `sg` | 1 | Skip-gram (not CBOW) |
| `seed` | 42 | Reproducibility |

### 6.3 ANN (Approximate Nearest Neighbor) search

Exact brute-force cosine similarity over 125K articles is only used as a last-resort fallback and for scoring small (5–250 item) candidate lists, where an ANN index is unnecessary overhead. For **full-corpus top-K retrieval**, a genuine ANN index is used:

- **FAISS `IndexHNSWFlat`** (Hierarchical Navigable Small World graph) when the `faiss` package is importable — `M=32` neighbours/node, `efConstruction=40`, `efSearch=64` (standard FAISS defaults for a ~125K-vector corpus).
- **From-scratch `IVFIndex`** (Inverted File index, the same core idea as FAISS's `IndexIVFFlat`) when `faiss` is not installable (this was the case in the original dev sandbox — DNS resolution was down). Built with `sklearn.MiniBatchKMeans` to cluster the corpus, then at query time only the `n_probe` nearest clusters are scanned instead of the full corpus.

**Why IVF (clustering) instead of LSH (random hyperplanes)**: an LSH index was tried first and dropped. Mean-pooled Word2Vec article vectors are strongly **anisotropic** — the corpus mean vector's norm is ≈0.77, meaning most article vectors point in roughly the same general direction (a known property of averaged word embeddings). Random-hyperplane LSH assumes roughly uniform coverage of the unit sphere, so a handful of hyperplane-defined buckets ended up holding a large fraction of the corpus each; measured recall against exact brute force was only ~70% with barely any speedup. K-means-based IVF instead adapts directly to the corpus's actual clustered geometry.

**IVF parameter selection** — measured directly against exact brute-force search on this corpus:

| `n_probe` | Recall@200 | Candidate-pool reduction | Throughput |
|---:|---:|---:|---:|
| 8 | ~90% | ~24× smaller | ~690 queries/s |
| **16 (chosen)** | **~95%** | **~12× smaller** | **~380 queries/s** |
| 32 | ~99% | — | ~190 queries/s |

### 6.4 Parameters used

| Parameter | Value | Meaning |
|---|---:|---|
| `n_clusters` (IVF) | 256 | K-means clusters over the corpus |
| `n_probe` (IVF) | 16 | Clusters scanned per query (~95% recall@200) |
| HNSW `M` | 32 | Graph neighbours per node (FAISS path) |
| HNSW `efConstruction` | 40 | Build-time quality parameter |
| HNSW `efSearch` | 64 | Query-time recall/speed tradeoff |
| Similarity metric | Cosine (via L2-normalized inner product) | |

---

## 7. Behavioural Feature Engineering (Click-Logs)

**File**: `pipeline/features.py` — this is the core of Assignment 2's Q1, and produces one row per **(impression, candidate) pair** with the following engineered features.

### 7.0 Candidate generation (Q2.1)

Each row's candidate pool is **retrieved, not read off the dataset's own candidate list**: it's the union of Assignment 1's BM25 and Word2Vec+ANN top-`RETRIEVAL_CANDIDATE_K` (`=100` per channel, so the fused pool lands in Q2.1's target `K ~ 100–200`) full-corpus retrieval, queried from the user's click-history window with the exact same plain mean-pooled-embedding + concatenated-text query construction `pipeline/retrieval_eval.py`'s `_build_queries` uses — not the recency-weighted history vector used for `hist_embed_sim` below.

**A self-retrieved top-K usually misses the article the user actually clicked.** Per `outputs/<dataset>/recall_at_k.json`, BM25/embedding recall@100–200 over this ~125K-article corpus measures only **~0.4–2.3%** — i.e. for upwards of 97–99% of impressions, neither retrieval channel finds the true click at all. That impression's real positive (from the dataset's own officially-provided candidates/labels) is therefore force-included whenever retrieval didn't surface it, so no impression silently loses its supervised label — but it's spliced in at a **uniformly random position** among the genuinely-retrieved candidates, not appended at a fixed spot. With the miss rate this high, a fixed insertion point (e.g. always last) would make `candidate_position` an almost perfect giveaway of the label — a leakage shortcut a GBDT can trivially learn instead of the real behavioural features, which is exactly what an early version of this fix did (before this was caught: MIND's val AUC came out at an implausible 0.978, the tell that something was leaking). This candidate-pool swap does **not** touch `pipeline/retrieval_eval.py`'s recall@K, which remains an untouched, honest measure of open-corpus retrieval quality against the true click.

### 7.1 Feature categories

**1. Click-history features**
| Feature | Definition |
|---|---|
| `hist_len` | Total count of articles in the user's historical click log |
| `hist_category_match_frac` | Unweighted fraction of the user's recent history (last 50 clicks) whose category matches the candidate's category |
| `hist_category_match_recency` | Same, but **recency-weighted** (see decay formulas below) |
| `hist_embed_sim` | Cosine similarity between the candidate's 128-dim Word2Vec vector and the user's recency-weighted mean historical embedding |

**2. Session features**
| Feature | Definition |
|---|---|
| `session_position` | Count of impressions the user has already made **strictly earlier** in the current session (never the session's eventual total size — that isn't knowable at serving time) |
| `user_avg_read_time` | Historical average dwell time (EB-NeRD only — MIND's raw files carry no dwell signal) |
| `user_avg_scroll_pct` | Historical average scroll depth (EB-NeRD only) |

**3. Article & position features**
| Feature | Definition |
|---|---|
| `popularity_log` | $\log(1+\text{clicks})$, counted **strictly from the training split only** |
| `freshness_hours` | Elapsed hours since publication (EB-NeRD, real `published_time`) or since the article's first observed appearance in a training candidate list (MIND proxy, since MIND has no publish timestamp) |
| `category_match` | Binary: does the candidate's category equal the user's single most-frequently-clicked historical category? |
| `candidate_position` | The candidate's rank in the retrieval-fused candidate pool (see §7.0), not a platform display index — captures retrieval-confidence-order bias |
| `candidate_position_norm` | `candidate_position` normalized to $[0,1]$ by list length |

Feature count: **10 features for MIND**, **12 for EB-NeRD** (+2 dwell-time columns).

### 7.2 Recency-weighting: rank decay vs. time decay

This is the key methodological difference between the two datasets, since MIND's `news.tsv`/`behaviors.tsv` carry no per-click timestamp inside a user's history (order only), while EB-NeRD's `history.parquet` carries a real `impression_time_fixed` per history item.

**MIND — rank-based exponential decay** (`pipeline/common/behavioral_features.py::rank_decay_weights`, shared with `generate_submission.py`'s serving-time scoring): the most recent history item gets weight 1, and each position further back decays geometrically:

$$w_i = \gamma^{(n-1-i)}, \quad \gamma = \texttt{HIST\_DECAY\_RANK} = 0.9$$

for a history window of length $n$ (index 0 = oldest).

**EB-NeRD — real time-based exponential decay** (`pipeline/common/behavioral_features.py::time_decay_weights`), using an explicit half-life:

$$w_i = 0.5^{\Delta t_i / t_{1/2}}, \quad \Delta t_i = (\text{impression time} - \text{history item }i\text{'s time}) \text{ in hours}, \quad t_{1/2} = \texttt{HIST\_HALF\_LIFE\_HOURS} = 72.0$$

i.e. a click's influence on the recency-weighted category/embedding average halves every 72 hours (3 days) — appropriate for fast-decaying daily news cycles.

### 7.3 History window

Both datasets cap the click-history window used for feature computation at the most recent **`FEATURE_HISTORY_WINDOW = 50`** items (matching Assignment 1's retrieval query-construction window, for consistency between stages).

### 7.4 Session bucketing

- **MIND** has no native session concept — sessions are heuristically reconstructed per user via a **30-minute inactivity gap** (`session_position_mind`, `gap_minutes=30.0`), a standard session-boundary heuristic.
- **EB-NeRD** ships native `session_id` per impression — used directly (`session_position_ebnerd`).

### 7.5 Behavioural-window boundary enforcement (anti-leakage, Q1.4/Q9)

Four concrete guarantees, each independently verified by a pytest test (Section 13):
1. `popularity_log` is counted **exclusively** from `behaviors_train.parquet`; validation/test impressions reuse the training counts unchanged (never re-counted on future data).
2. `session_position` only counts impressions **strictly before** the current one in the same session.
3. MIND's freshness proxy table (article → first-seen timestamp) is built **exclusively** from training candidate lists.
4. History-derived features only ever read the user-history table for the **same temporal split** that impression belongs to.

### 7.6 Cold-start handling

Users with no click history get **neutral** feature defaults (0.0 similarity, 0.0 category overlap) rather than any leaked or arbitrary default — enforced by `test_features_empty_history_is_neutral`.

---

## 8. Stage 2 — GBDT Re-Ranker

**File**: `pipeline/common/reranker.py`, driven by `pipeline/rerank.py`

### 8.1 Why GBDT (not a neural ranker)

The assignment (Q2.2) offers two options: **Option A** — GBDT (LightGBM/XGBoost-style) over hand-crafted features, or **Option B** — a small neural ranker (NRMS-style or an MLP). This repo implements **Option A**, using XGBoost's histogram GBDT (`XGBClassifier`, `tree_method="hist"`), trained on the GPU (`device="cuda"`) whenever a CUDA device is available and on CPU otherwise. An earlier version used scikit-learn's `HistGradientBoostingClassifier`, which has no GPU path; hyperparameters (300 rounds, learning rate 0.08, depth 6, L2 1.0) and the early-stopping scheme (stratified 10% holdout, 10 rounds of patience) were carried over unchanged.

### 8.2 What it predicts

The reranker is a **pointwise** classifier, not a pairwise or listwise ranker. Concretely:

- **Training**: each (impression, candidate) row is treated as an independent binary-classification example, with no notion of "this candidate vs. the others in the same impression." The model is optimized with plain binary cross-entropy to predict $P(\text{click}=1 \mid \text{features})$ — it never sees the group structure (which rows belong to the same impression) during training, only isolated feature vectors and 0/1 labels.
- **Inference**: every candidate in an impression is scored independently through the model to get its click probability, and the final ranking is simply those candidates **sorted by that probability, descending**. The ranking itself is not learned directly — it falls out as a side effect of sorting scalar scores that were each computed in isolation.

This is in contrast to:
- **Pairwise** methods (e.g. RankNet), which train on pairs of candidates from the same impression and learn to predict which of the two should rank higher — the loss directly penalizes wrongly-ordered pairs.
- **Listwise** methods (e.g. LambdaMART, ListNet), which optimize a ranking metric (like NDCG) over the *entire* candidate list for an impression at once, so the loss is aware of full list order, not just single points or pairs.

Pointwise is the simplest of the three to implement (it's an off-the-shelf binary classifier, no custom loss or group-aware training loop needed) and is what `XGBClassifier`'s default `binary:logistic` objective does out of the box — the tradeoff is that it optimizes classification accuracy per row rather than ranking quality per list, so it can be a slightly worse proxy for ranking metrics like NDCG/MRR than a pairwise or listwise loss would be.

### 8.3 Hyperparameters

| Parameter | Value | Meaning |
|---|---:|---|
| `max_iter` | 300 | Number of boosting iterations (trees) |
| `learning_rate` | 0.08 | Shrinkage applied to each tree's contribution |
| `max_depth` | 6 | Maximum tree depth |
| `l2_regularization` | 1.0 | L2 penalty on leaf weights |
| `early_stopping` | True | Stops training if validation loss stops improving |
| `validation_fraction` | 0.1 | Internal held-out fraction used for early stopping |
| `random_state` | 42 | Reproducibility (shared `RANDOM_SEED` across the whole pipeline) |

> Note: `report.md`'s Q2/Q3 narrative describes "300 trees, learning rate 0.1" as the shared hyperparameters between baseline and full model — the actual code value for `learning_rate` is **0.08** (see `pipeline/common/reranker.py:48`); both models are trained with identical hyperparameters regardless, which is what makes the ablation in Section 9 valid.

### 8.4 Feature set used

All engineered columns from `pipeline/features.py` except identifier/label columns (`impression_id`, `user_id`, `candidate_id`, `label`) — i.e. 10 features for MIND, 12 for EB-NeRD (Section 7.1). One model is trained **per dataset**, since the two datasets' feature sets don't match exactly (EB-NeRD's two dwell-time columns don't exist for MIND).

### 8.5 "Before" vs "After" comparison

- **Before** = Assignment 1's own BM25 and Word2Vec scores on the exact same candidate lists (already computed in `retrieval_results_val.parquet`).
- **After** = this module's GBDT probability score on the same candidates.

All three (BM25, Embedding, GBDT) are evaluated with the identical AUC/MRR/nDCG@5/nDCG@10 pipeline (Section 10), making the comparison apples-to-apples.

---

## 9. Baseline, Ablation & Statistical Significance

**File**: `pipeline/ablation.py` — Assignment 2's Q3.

### 9.1 What "baseline" means here

Q3.1 asks to reproduce "the official/starter baseline (e.g., NRMS from the ebnerd-benchmark repo, or the MIND baseline)." NRMS is a neural ranker requiring substantial user/news encoder implementation, judged out of scope for this repo's Q2 (see `pipeline/rerank.py` docstring). Instead, the reproduced baseline is a **deliberately minimal, non-personalized GBDT**:

```
FEATURE_COLS_MINIMAL = ["popularity_log", "freshness_hours", "candidate_position_norm"]
```

i.e. popularity + freshness + shown position only — **no** click-history, category affinity, or session context. It is trained with the **exact same** algorithm, hyperparameters (Section 8.3), training data, and random seed as the full model. This isolates the ablation to exactly one variable: the feature set.

### 9.2 Paired bootstrap confidence intervals

For each metric, the **paired per-impression delta** is computed (full model's score minus baseline's score, on the *same* impression with the *same* labels), then that delta array itself is bootstrapped — this is a **paired** bootstrap, not two independent bootstraps, which is the statistically correct way to test whether one model beats another on the same evaluation set.

**Bootstrap procedure** (`pipeline/common/metrics.py::bootstrap_ci`):
1. Resample the delta array **with replacement**, `n_boot = 1000` times.
2. Compute the mean of each resample.
3. The 95% CI is the $[2.5\text{th}, 97.5\text{th}]$ percentile of the 1000 resampled means.
4. A gain is declared **statistically significant** iff this CI **excludes zero** (i.e. `ci_low > 0` or `ci_high < 0`).

### 9.3 Results

| Dataset | Metric | Baseline | Full Model | Paired Δ | 95% CI | Significant? |
|---|---|---:|---:|---:|---:|:---:|
| MIND | AUC | 0.5419 | 0.5575 | +0.0156 | [+0.0139, +0.0176] | ✅ |
| MIND | MRR | 0.2427 | 0.2671 | +0.0243 | [+0.0228, +0.0261] | ✅ |
| MIND | nDCG@5 | 0.2599 | 0.2852 | +0.0253 | [+0.0235, +0.0270] | ✅ |
| MIND | nDCG@10 | 0.3179 | 0.3420 | +0.0241 | [+0.0226, +0.0256] | ✅ |
| EB-NeRD | AUC | 0.4420 | 0.5350 | +0.0930 | [+0.0917, +0.0944] | ✅ |
| EB-NeRD | MRR | 0.2532 | 0.3056 | +0.0524 | [+0.0515, +0.0535] | ✅ |
| EB-NeRD | nDCG@5 | 0.2801 | 0.3487 | +0.0687 | [+0.0675, +0.0699] | ✅ |
| EB-NeRD | nDCG@10 | 0.3816 | 0.4361 | +0.0545 | [+0.0537, +0.0554] | ✅ |

**Interpretation**: every gain is positive and every CI excludes zero on both datasets — the behavioural features are a statistically significant improvement over the non-personalized baseline. The gain is **~6× larger on EB-NeRD** (+0.093 AUC) than on MIND (+0.016 AUC), which traces directly back to Section 7.2: EB-NeRD's exact per-click timestamps (true exponential time-decay) plus dwell-time/scroll signals give the reranker strictly more discriminative information than MIND's coarse rank-decay proxy and absent dwell signal.

---

## 10. Evaluation Metrics

**File**: `pipeline/common/metrics.py` — all four metrics follow the official MIND/RecSys-challenge convention: computed **per impression** against that impression's own candidate list, then averaged across impressions ("grouped" evaluation).

### 10.1 AUC (Area Under the ROC Curve)
Probability that a randomly chosen clicked candidate is scored higher than a randomly chosen non-clicked candidate, within the same impression. Computed via `sklearn.metrics.roc_auc_score`. Only defined (and only included in the average) for impressions with **at least one** click **and** at least one non-click.

### 10.2 MRR (Mean Reciprocal Rank)

$$\text{MRR} = \frac{1}{|\{\text{clicks}\}|}\sum_{i : y_i = 1} \frac{1}{\text{rank}(i)}$$

Averages $1/\text{rank}$ over all clicked items in the ranked list (supports multiple clicks per impression, not just top-1).

### 10.3 nDCG@k (normalized Discounted Cumulative Gain)

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where $\text{IDCG@}k$ is the DCG of the ideal (perfectly-sorted) ranking. Computed for $k=5$ and $k=10$.

### 10.4 Bootstrap 95% CI for every metric

Every reported metric mean is accompanied by a 95% CI from `n_boot=1000` resamples of the per-impression metric array (same procedure as Section 9.2, but unpaired — a single model's own confidence interval rather than a paired delta).

### 10.5 "Before vs. after re-ranking" results (Q2)

| Dataset | Stage | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| MIND | BM25 (before) | 0.5685 | **0.2687** | **0.2865** | **0.3477** |
| MIND | Embedding (before) | 0.5444 | 0.2463 | 0.2564 | 0.3196 |
| MIND | **GBDT re-rank (after)** | **0.5575** | 0.2671 | 0.2852 | 0.3420 |
| EB-NeRD | BM25 (before) | 0.5206 | **0.3330** | **0.3664** | **0.4479** |
| EB-NeRD | Embedding (before) | 0.5239 | 0.3278 | 0.3627 | 0.4442 |
| EB-NeRD | **GBDT re-rank (after)** | **0.5350** | 0.3056 | 0.3487 | 0.4361 |

Note that "before" and "after" use fundamentally different scoring distributions (lexical/semantic similarity scores vs. a calibrated click-probability), which is why the *ablation* in Section 9 (an apples-to-apples GBDT-vs-GBDT comparison) is the cleaner test of whether behavioural features actually help.

---

## 11. Beyond-Accuracy Metrics

**File**: `pipeline/common/metrics.py` — computed on each retrieval method's own **top-10 full-corpus** retrieval (what it would actually recommend), not a re-ranking of the officially-provided candidate list.

### 11.1 Intra-list diversity

$$\text{diversity}(\text{list}) = 1 - \frac{1}{n(n-1)}\sum_{i \ne j} \cos(\vec{v}_i, \vec{v}_j)$$

One minus the mean pairwise cosine similarity between all item-pairs in a recommended list, in the shared Word2Vec embedding space. Higher = more varied recommendations.

### 11.2 Novelty (self-information)

$$\text{novelty}(\text{list}) = \frac{1}{n}\sum_{i \in \text{list}} -\log_2\left(\frac{\text{clicks}(i) + 1}{\text{total\_clicks} + \text{catalog\_size}}\right)$$

Laplace-smoothed self-information of each recommended item's training-set popularity; unpopular items score higher (more "surprising").

### 11.3 Coverage

$$\text{coverage} = \frac{|\bigcup_{\text{all impressions}} \text{top-10 recommended items}|}{\text{catalog size}}$$

Fraction of the entire catalog that is *ever* recommended in the top-10 across the whole validation set — a system that always recommends the same handful of popular articles has near-zero coverage even with perfect accuracy.

### 11.4 Results

| Dataset | Method | Diversity | Novelty | Coverage |
|---|---|---:|---:|---:|
| MIND | BM25 | 0.298 | 18.21 | 31.4% |
| MIND | Word2Vec | 0.108 | 18.21 | 15.3% |
| EB-NeRD | BM25 | 0.340 | 12.46 | 7.4% |
| EB-NeRD | Word2Vec | **0.001** | 18.39 | **1.5%** |

**Why EB-NeRD's Word2Vec diversity collapses to ~0.001**: the mean-pooled embedding space is severely anisotropic on this corpus (mean vector norm ≈0.77, per Section 6.3), so dense retrieval repeatedly selects the same narrow cluster of articles regardless of query — nearly every top-10 list is filled with near-duplicate near-neighbors. BM25 delivers substantially healthier coverage and diversity on both corpora, since lexical overlap naturally spreads across distinct vocabulary/topics rather than collapsing to one geometric region.

---

## 12. Slicing Analysis

**File**: `pipeline/evaluate.py` — Q5's requirement of "at least two slices."

| Slice dimension | Definition | Threshold |
|---|---|---|
| **Cold-start vs. Warm** | User's `history_len` ≤ threshold = cold-start | `COLD_START_THRESHOLD = 5` |
| **Head vs. Tail articles** | Clicked article's train-popularity ≥ top-20th-percentile = head | `HEAD_FRACTION = 0.2` |

### Results (BM25 retriever)

| Dataset | Slice | Impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| MIND | Cold-start (≤5 hist.) | 12,982 | 0.5301 | 0.2790 | 0.2914 | 0.3527 |
| MIND | Warm (>5 hist.) | 60,170 | 0.5768 | 0.2665 | 0.2855 | 0.3466 |
| MIND | Head articles | 21,311 | 0.5456 | 0.2216 | 0.2434 | 0.3008 |
| MIND | Tail articles | 51,841 | 0.5780 | 0.2880 | 0.3042 | 0.3670 |
| EB-NeRD | Cold-start | 887 | 0.5278 | 0.3269 | 0.3582 | 0.4404 |
| EB-NeRD | Warm | 243,760 | 0.5206 | 0.3330 | 0.3664 | 0.4480 |
| EB-NeRD | Head articles | 4,704 | 0.5141 | 0.3000 | 0.3059 | 0.3908 |
| EB-NeRD | Tail articles | 239,943 | 0.5207 | 0.3336 | 0.3676 | 0.4491 |

**MIND**: warm users score much higher AUC (0.577 vs 0.530) — richer history gives BM25 a stronger query signal. **Both datasets**: tail articles outrank head articles on ranking metrics — head/popular candidates face higher competition within their impressions (many similar high-quality candidates), while position bias and query specificity favor niche articles when they do appear.

---

## 13. Anti-Leakage / Temporal Splitting

**Files**: `pipeline/common/split.py`, `pipeline/tests/test_no_leakage.py` — Assignment's **Q9 (Anti-Gaming)** requirement.

### 13.1 Splitting principle

Interaction data is **never split randomly** — every split is a cutoff on a timestamp column, so anything after the cutoff is strictly in the future relative to anything before it (`temporal_cutoff`, `split_by_cutoff`).

### 13.2 Two levels of split

1. **Official train/val split** — provided by each dataset (MIND: `MINDsmall_train` vs `MINDsmall_dev`; EB-NeRD: `train/` vs `validation/`).
2. **Internal fit/tune split** — carved out of the *training* set's own tail (`carve_internal_validation`, `INTERNAL_TUNE_HOLDOUT_FRACTION = 0.15`), used only for hyperparameter sanity checks so the official validation split is never touched until final reporting.

### 13.3 Verified invariants (pytest suite, `pipeline/tests/test_no_leakage.py`)

| Test | Guarantees |
|---|---|
| `test_train_val_no_overlap` | $\max(\text{Train}_{\text{ts}}) \le \min(\text{Val}_{\text{ts}})$ for both datasets |
| `test_internal_fit_tune_no_overlap` | Internal tuning splits don't leak future impressions into the fit split |
| `test_fit_tune_reconstruct_train` | `fit ∪ tune = train`, no duplicates or dropped rows |
| `test_features_freshness_non_negative` | `freshness_hours ≥ 0` everywhere (no future-publish leakage) |
| `test_features_popularity_is_train_only` | Validation `popularity_log` values match a recount from training logs only |
| `test_features_session_position_starts_at_zero` | `session_position` starts at 0 per user's first impression, increments monotonically |
| `test_features_empty_history_is_neutral` | Cold-start users get neutral (0.0) feature values, not leaked defaults |

**Test suite status**: 20 passed, 2 skipped (expected empty-sample guards).

Run with: `python -m pytest pipeline/tests/test_no_leakage.py -v`

---

## 14. Serving & Scale Analysis

**File**: `pipeline/serving_scale.py` — Assignment's **Q4**. All numbers are **measured**, not estimated, against the pipeline's own real trained artifacts (no synthetic stand-in).

### 14.1 Index memory (component breakdown)

| Component | MIND | EB-NeRD | Storage type |
|---|---:|---:|---|
| BM25 term-weight matrix | 25.93 MB | 14.68 MB | Sparse CSR (`.data`+`.indices`+`.indptr` bytes) |
| BM25 vocabulary size | 49,986 terms | 58,946 terms | |
| Word2Vec dense embeddings | 64.30 MB | 64.28 MB | 128-dim float32, dense |
| ANN index (IVF fallback) | 1.14 MB | 1.14 MB | K-means centroids + cluster ID lists |
| GBDT reranker model (pickled) | 1.02 MB | 1.04 MB | Serialized trees |
| **Total in-memory serving footprint** | **92.39 MB** | **81.13 MB** | Sum of the above |
| Feature store on disk | 446.27 MB | 343.81 MB | 12 parquet files |
| Process peak RSS (cross-check) | 620.02 MB | 1,248.65 MB | Includes Python/library overhead not captured by the object-level sum |

### 14.2 Latency (500 single-request trials, single CPU core, no batching)

| Metric | MIND | EB-NeRD |
|---|---:|---:|
| Mean | 333.02 ms | 148.82 ms |
| p50 (median) | 227.06 ms | 136.47 ms |
| p95 | 711.18 ms | 222.71 ms |
| **p99** | **1180.33 ms** | **297.46 ms** |
| Max | 1618.49 ms | 579.76 ms |
| Min | 48.09 ms | 74.25 ms |

Each simulated request = build a query from the user's history → BM25 top-K (K=100) union embedding top-K (K=100) candidate generation → GBDT rerank over the merged, de-duplicated ~200-candidate set. `RERANK_TOP_K = 200` total, split evenly across the two retrieval sources (`k_each = 100`).

### 14.3 Cost / QPS (back-of-envelope, target SLA p99 < 100 ms)

| Metric | MIND | EB-NeRD |
|---|---:|---:|
| Throughput per core | 3.00 QPS | 6.72 QPS |
| Cost per 1,000 queries (@ $0.05/hr, 1-vCPU) | $0.0046 | $0.0021 |
| p99 meets 100 ms SLA on 1 core? | ❌ No | ❌ No |
| Cores needed to sustain 1,000 QPS | 334 | 149 |

**Assumptions** (`cost_per_1000_queries`): `hourly_cpu_cost_usd = 0.05` (a modest 1-vCPU cloud instance, e.g. AWS t3-family on-demand), single request at a time per core (no batching credit — batching gains are already reported separately by the bulk Q2/Q3 scoring runs), perfect linear horizontal scaling for the cores-needed estimate.

### 14.4 10× scaling argument — what breaks first?

- **10× query volume alone (10,000 QPS, same corpus size)**: because the candidate index and GBDT model are read-only and stateless, this scales **horizontally** — add more replicas behind a load balancer, each replica holds its own full index copy, per-request latency stays roughly fixed. Not a bottleneck by itself.
- **Simultaneous 10× corpus scale (~1.25M articles)**:
  - *Memory*: the dense embedding matrix grows to ~643 MB per replica; BM25 grows to ~200 MB. Manageable per-replica, but multiplies across every replica in the fleet.
  - **Compute bottleneck — the ANN vector search breaks first**: this environment's IVF index executes centroid-distance checks and bucket traversals in a **Python loop per query** rather than a batched, native-vectorized library (real FAISS). As candidates per Voronoi cell grow 10×, single-threaded IVF search latency compounds and dominates the request pipeline. GBDT re-ranking, by contrast, only ever evaluates the top-K (~100–200) candidates regardless of total catalog size, so its cost stays constant — it is *not* expected to be the limiting stage.
- **Net argument**: horizontal replication cleanly absorbs a 10× QPS-only increase; a **simultaneous** 10× QPS + 10× corpus increase (i.e., moving to the assignment's "large" dataset bundles under load) first strains per-replica index memory, then makes the from-scratch IVF fallback's per-query Python loop the compute bottleneck — a real FAISS index (not installable in the original dev sandbox due to network restrictions) would remove this specific bottleneck without any other architecture change.

---

## 15. Codabench Submission Format

**Files**: `pipeline/common/submission.py`, `pipeline/generate_submission.py` — Assignment's **Q5** deliverable.

### 15.1 Prediction generation

Both leaderboards are ranked by **the trained Q2 GBDT reranker** (`feature_store/<dataset>/reranker_model.joblib`), predicting $P(\text{click}=1 \mid \text{features})$ for each candidate in that impression's own **officially-given** candidate list — this is "the full two-stage pipeline" Q5 asks the submission to reflect, not Stage 1 retrieval alone. Codabench's own server-side scoring is a permutation check against that exact given list, so unlike §7.0's retrieval-based candidate generation for training/eval, the candidate *set* here is never swapped for a self-retrieved pool.

Feature columns are computed by the same formulas `pipeline/features.py` uses (shared via `pipeline/common/behavioral_features.py`, so training and serving can't silently drift into two different implementations of "the same" feature). One feature needs care at serving time: `candidate_position` was defined at training time as a candidate's rank under A1's own fused BM25+embedding retrieval score over the *retrieved* pool, so at serving time it's computed the same way — by scoring the *given* candidates (not searching the full corpus) with `pipeline/common/submission.py`'s `hybrid_score()` and ranking them — rather than from the candidate's raw position in Codabench's input file. `hybrid_score()` is therefore no longer the final ranking function (as it was before this fix); it now only feeds this one feature.

A popularity-only shortcut is used only for the trivial `len(candidates) <= 1` case (nothing to rank); cold-start (empty-history) users are otherwise scored through the same GBDT path as everyone else, since training never special-cased them either. Test sets are processed via **chunked streaming** (never materialized fully in memory) since the large test sets are unlabeled and huge (see below); `session_position` — which needs a user's impressions sorted together — is computed in one lightweight upfront pass over just the test file's `(user_id, timestamp)` columns before the main streaming loop.

### 15.2 Output line format

```
impression_id [rank_of_candidate_1,rank_of_candidate_2,...,rank_of_candidate_N]
```

where `rank_of_candidate_i` is a 1-indexed rank (1 = most likely to be clicked) for the *i*-th candidate in that impression's originally-given candidate order — produced by `scores_to_ranks` (`scipy.stats.rankdata`, `method="ordinal"` for deterministic tie-breaking).

### 15.3 Submissions produced

| Competition | Archive | File | Rows | Validation |
|---|---|---|---:|---|
| MIND (Codabench) | `Q5_MIND/prediction.zip` | `prediction.txt` | 2,370,727 | Exact row count matched; valid 1-indexed permutations; no missing ranks |
| RecSys 2024 / EB-NeRD (Codabench) | `Q5_EB-NeRd/predictions.zip` | `predictions.txt` | 13,536,710 | Exact row count matched; valid 1-indexed permutations; no missing ranks |

---

## 16. Complete Parameter Reference Table

Every tunable constant in the pipeline, sourced directly from `pipeline/config.py` and the modules that define their own local constants.

| Parameter | Value | Defined in | Purpose |
|---|---:|---|---|
| `RANDOM_SEED` | 42 | `config.py` | Global reproducibility seed |
| `INTERNAL_TUNE_HOLDOUT_FRACTION` | 0.15 | `config.py` | Internal fit/tune carve fraction |
| `BM25_K1` | 1.5 | `config.py` | BM25 term-frequency saturation |
| `BM25_B` | 0.75 | `config.py` | BM25 length-normalization strength |
| BM25 `min_df` | 2 | `bm25.py` | Minimum document frequency for vocabulary |
| `W2V_DIM` | 128 | `config.py` | Word2Vec embedding dimension |
| `W2V_WINDOW` | 8 | `config.py` | Word2Vec context window |
| `W2V_MIN_COUNT` | 2 | `config.py` | Word2Vec minimum word frequency |
| `W2V_EPOCHS` | 10 | `config.py` | Word2Vec training epochs |
| `W2V_WORKERS` | 20 | `config.py` | Word2Vec training threads |
| Word2Vec `sg` | 1 (skip-gram) | `embeddings.py` | Skip-gram vs. CBOW |
| IVF `n_clusters` | 256 | `embeddings.py` | K-means clusters for ANN index |
| IVF `n_probe` | 16 | `embeddings.py` | Clusters scanned per ANN query |
| HNSW `M` | 32 | `embeddings.py` | Graph neighbours/node (FAISS path) |
| HNSW `efConstruction` | 40 | `embeddings.py` | HNSW build quality |
| HNSW `efSearch` | 64 | `embeddings.py` | HNSW query-time recall/speed tradeoff |
| `RECALL_KS` | [50, 100, 200] | `config.py` | Recall@K evaluation cutoffs |
| `FEATURE_HISTORY_WINDOW` | 50 | `config.py` | Max history items used per feature build |
| `HIST_DECAY_RANK` | 0.9 | `config.py` | MIND rank-decay base (per-position) |
| `HIST_HALF_LIFE_HOURS` | 72.0 | `config.py` | EB-NeRD time-decay half-life (hours) |
| `FEATURE_BUILD_CHUNK` | 20,000 | `config.py` | Impressions per feature-build chunk |
| MIND session gap | 30 min | `features.py` | Heuristic session-boundary threshold |
| GBDT `max_iter` | 300 | `reranker.py` | Boosting rounds |
| GBDT `learning_rate` | 0.08 | `reranker.py` | Shrinkage per tree |
| GBDT `max_depth` | 6 | `reranker.py` | Max tree depth |
| GBDT `l2_regularization` | 1.0 | `reranker.py` | L2 penalty on leaves |
| GBDT `validation_fraction` | 0.1 | `reranker.py` | Early-stopping holdout |
| `COLD_START_THRESHOLD` | 5 | `evaluate.py` | History length cutoff for cold-start slice |
| `HEAD_FRACTION` | 0.2 | `evaluate.py` | Top-percentile cutoff for "head" articles |
| Bootstrap `n_boot` | 1,000 | `metrics.py` | Bootstrap resamples |
| Bootstrap `ci` | 0.95 | `metrics.py` | Confidence level |
| `HISTORY_WINDOW` (serving) | 50 | `serving_scale.py` | Query-construction window at serving time |
| `RERANK_TOP_K` (serving) | 200 | `serving_scale.py` | Candidates sent to GBDT stage |
| `N_TRIALS_DEFAULT` | 500 | `serving_scale.py` | Latency-benchmark trial count |
| `TARGET_SLA_MS` | 100.0 | `serving_scale.py` | Target p99 latency SLA |
| `ASSUMED_HOURLY_CPU_COST_USD` | 0.05 | `serving_scale.py` | Cost-model assumption (1-vCPU/hr) |

---

## 17. Complete Results Reference Table

### 17.1 Corpus & log statistics

| Metric | MIND | EB-NeRD |
|---|---:|---:|
| Articles | 125,590 | 125,541 |
| Avg. doc length (tokens) | 30.78 | 15.88 |
| Avg. doc length (chars) | 279.66 | 148.30 |
| BM25 vocabulary size | 49,986 | 58,946 |
| Categories | 18 | 33 |
| Train / val impressions | 156,965 / 73,152 | 232,887 / 244,647 |
| Avg. candidates / impression | 37.23 / 37.47 | 11.10 / 11.97 |
| Avg. history length | 32.54 | 306.81 |
| Avg. CTR per impression | 10.85% / 10.01% | 12.09% / 11.74% |

### 17.2 Open-corpus Recall@K (before re-ranking, full ~125K-article catalog)

| Dataset | Method | Recall@50 | Recall@100 | Recall@200 |
|---|---|---:|---:|---:|
| MIND | BM25 | 0.63% | 1.17% | 2.23% |
| MIND | Word2Vec | 0.25% | 0.41% | 0.66% |
| EB-NeRD | BM25 | 0.33% | 0.57% | 0.96% |
| EB-NeRD | Word2Vec | 0.01% | 0.01% | 0.04% |

(BM25 dominates dense retrieval at every cutoff — exact keyword/entity matches on titles beat static averaged Word2Vec vectors for fast-decaying news content.)

### 17.3 Ranking metrics (see Sections 9, 10, 12 for full tables)

See Section 10.5 (before/after re-ranking) and Section 9.3 (baseline vs. full ablation with CIs) for the primary reported numbers.

### 17.4 Serving benchmark summary

See Section 14 in full; headline numbers: MIND p99 = 1180 ms / 334 cores needed for 1,000 QPS; EB-NeRD p99 = 297 ms / 149 cores needed for 1,000 QPS — neither meets a 100 ms p99 SLA on a single core.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **Impression** | A single instance of a candidate list being shown to a user (a page-load event) |
| **Candidate** | One article offered within an impression's slate |
| **Click-through rate (CTR)** | Fraction of shown candidates a user clicks, averaged per impression |
| **Cold-start** | A user (or item) with little/no historical interaction data |
| **Head / Tail** | Popular (head) vs. unpopular (tail) items, by training-set click count |
| **Retrieval** | Stage 1: cheaply narrowing a huge catalog to a small candidate set |
| **Re-ranking** | Stage 2: expensively re-scoring a small candidate set with richer signals |
| **Pointwise ranking** | Predicting each candidate's relevance independently (vs. pairwise/listwise) |
| **Behavioural-window boundary** | The temporal rule that no feature may use information from after the event it's predicting |
| **Anisotropy (embeddings)** | When most vectors in a space point in a similar direction, degrading similarity search's discriminative power |
| **ANN (Approximate Nearest Neighbor)** | A search index that trades a small amount of recall for large speedups vs. exact brute-force search |
| **IVF (Inverted File index)** | An ANN method that clusters the corpus and searches only the nearest clusters at query time |
| **HNSW** | A graph-based ANN method (used here via FAISS when available) |
| **Paired bootstrap** | A resampling technique for confidence intervals on a *difference* between two models evaluated on the same data |
| **QPS** | Queries per second — a throughput measure |
| **p50 / p95 / p99** | The 50th/95th/99th percentile of a latency distribution — p99 is the "tail latency" that dominates worst-case user experience |
| **SLA** | Service-Level Agreement — a target performance guarantee (here, p99 latency) |

---

*This document was generated from the actual repository code (`pipeline/`), configuration (`pipeline/config.py`), and persisted evaluation outputs (`outputs/*.json`) as of the current codebase state, plus directly-computed corpus statistics (Section 2) not otherwise persisted anywhere in the repository. Cross-reference `report.md` for the assignment's formal narrative write-up and `README.md` for reproduction commands.*
