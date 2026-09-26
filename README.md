# Learning from Click-Logs on MIND and EB-NeRD
**CS4.406: Information Retrieval & Extraction — Assignment 2**

This repository contains an end-to-end two-stage news recommendation pipeline implemented for Microsoft News Dataset (**MIND**) and Ekstra Bladet News Recommendation Dataset (**EB-NeRD**). 

The system unifies raw datasets into a shared schema, performs candidate retrieval using lexical (BM25) and semantic (Word2Vec) indices, extracts temporal behavioural features from click logs, trains a gradient boosted decision tree (GBDT) re-ranker, evaluates models with paired bootstrap significance testing, and benchmarks serving latency and memory under scale.

---

## Reproducibility & Commands

The entire pipeline is automated via `make`.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipeline

Execute the full pipeline sequentially:

```bash
make all
```

Or execute individual pipeline stages:

```bash
make data           # Q1: Parse raw datasets into standardized parquet feature stores
make retrieve       # Q2/Q3: Run BM25 and Word2Vec candidate retrieval
make evaluate       # Q4: Evaluate ranking metrics, slices, diversity, novelty, and coverage
make rerank         # A2 Q1-Q3: Extract behavioural features, train GBDT re-ranker, run ablation
make serving_scale  # A2 Q4: Benchmark index memory, p99 latency, and 10x scaling
make submit         # Q5: Stream test sets and generate Codabench submission archives
make test           # Anti-leakage: Run pytest suite on temporal boundaries and feature integrity
make clean          # Remove generated feature store and output artifacts
```

Each stage also supports standalone execution per dataset:
```bash
python -m pipeline.rerank --dataset mind
python -m pipeline.rerank --dataset ebnerd
python -m pipeline.serving_scale --dataset mind --n_trials 200
```

---

## Expected Data Layout

Raw datasets should be placed in the workspace root as follows (configured in `pipeline/config.py`):

```text
MINDsmall_train/MINDsmall_train/{news,behaviors}.tsv       # MIND Train
MINDsmall_dev/MINDsmall_dev/{news,behaviors}.tsv           # MIND Validation
MINDlarge_test/MINDlarge_test/{news,behaviors}.tsv         # MIND Test (Codabench)
ebnerd_small/articles.parquet                              # EB-NeRD Articles
ebnerd_small/{train,validation}/*.parquet                  # EB-NeRD Train / Validation
ebnerd_testset/ebnerd_testset/{articles,test/*}.parquet    # EB-NeRD Test (Codabench)
```

---

## Repository Structure

```text
IRE-A2-Temp/
├── Makefile                     # Top-level workflow automation
├── requirements.txt             # Project dependencies
├── report.md                    # Formal assignment status report
├── pipeline/
│   ├── adapters/                # Raw-to-unified schema parsers (mind.py, ebnerd.py)
│   ├── common/                  # Shared algorithmic modules
│   │   ├── bm25.py              # Sparse BM25 index and query scoring
│   │   ├── embeddings.py        # Word2Vec embeddings and IVF ANN index
│   │   ├── metrics.py           # Ranking, diversity, novelty, and bootstrap CI metrics
│   │   ├── popularity.py        # Train-only popularity computation
│   │   ├── reranker.py          # Pointwise GBDT trainer and inference wrappers
│   │   ├── split.py             # Temporal train/validation splitting logic
│   │   ├── submission.py        # Streaming rank-fusion and Codabench formatters
│   │   └── text.py              # Tokenization and text pre-processing
│   ├── build_pipeline.py        # Dataset ingestion and feature store creation
│   ├── retrieval_eval.py        # Candidate generation evaluation (Recall@K)
│   ├── evaluate.py              # Beyond-accuracy and slice evaluation
│   ├── features.py              # Behavioural, session, and recency feature engineering
│   ├── rerank.py                # Two-stage retrieve-then-rank pipeline
│   ├── ablation.py              # Baseline vs. full model ablation with paired bootstrap CI
│   ├── serving_scale.py         # Serving memory, latency, QPS, and scaling benchmark
│   ├── generate_submission.py   # Test-set prediction generation
│   └── tests/
│       └── test_no_leakage.py   # Unit test suite enforcing temporal boundary integrity
├── feature_store/               # Generated intermediate parquet data (excluded from git)
├── outputs/                     # Persisted metric summaries and evaluations (excluded from git)
├── Q5_MIND/                     # Validated MIND Codabench submission archive
└── Q5_EB-NeRd/                  # Validated EB-NeRD Codabench submission archive
```

---

## Shared Schema Architecture

To maintain dataset-agnostic ranking and evaluation logic, both datasets are parsed into a normalized relational schema:

- **`articles`**: `article_id`, `text_lexical` (concatenated title and body/subtitle), and `category`.
- **`behaviors`**: `impression_id`, `user_id`, `timestamp`, `history_len`, `candidates`, and `labels`.
- **`user_history`**: A separate table indexed by `user_id` storing the chronological sequence of clicked article IDs preceding that split. Separating user history from individual impressions prevents duplicating click lists across repeated user sessions, reducing intermediate memory overhead by approximately $90\%$.

---

## Pipeline Overview

### 1. Feature Engineering (`pipeline/features.py`)
Extracts impression-level behavioural features across four categories:
- **Click History**: Total clicks, category match proportion, recency-weighted category overlap (exponential decay via exact timestamps for EB-NeRD and reverse-rank for MIND), and cosine similarity against historical Word2Vec embeddings.
- **Session Dynamics**: Preceding impression count within the current session, plus historical average read time and scroll percentage (EB-NeRD).
- **Article & Context**: Train-set log popularity, elapsed freshness in hours, single-top-category match, and display position.

### 2. Two-Stage Retrieve-then-Rank (`pipeline/rerank.py`)
- **Stage 1 (Retrieval)**: Candidate lists ($K \approx 5\text{--}300$) are retrieved and pre-scored using BM25 lexical matching and dense semantic embedding search.
- **Stage 2 (Re-ranking)**: An XGBoost histogram GBDT (GPU when CUDA is available) is trained on pointwise click labels using the engineered behavioural features, re-ranking candidates by predicted click probability $P(\text{click})$.

### 3. Baseline & Ablation Study (`pipeline/ablation.py`)
A non-personalized baseline (popularity, freshness, and position only) is evaluated alongside the full model under identical hyperparameters. Statistical significance is computed using 1,000 paired bootstrap iterations at the impression level, confirming that all gains exclude zero at a 95% confidence interval.

### 4. Serving & Scale Profiling (`pipeline/serving_scale.py`)
Measures exact in-memory footprints (sparse BM25 matrix, dense Word2Vec array, IVF cluster centroids, and GBDT model binary) and assesses single-request $p_{50}$ and $p_{99}$ latency over 500 trials. Includes back-of-the-envelope calculations for cost-per-1,000-queries and analyzes architectural bottlenecks under $10\times$ request and corpus scale.

---

## Temporal Splitting & Anti-Leakage (Q9)

Temporal ordering is enforced across all processing stages to prevent lookahead bias:
1. **Official Splits**: Verified to ensure that $\max(\text{Train}_{\text{time}}) \le \min(\text{Val}_{\text{time}})$.
2. **Internal Tuning**: Train sets are partitioned into strictly ordered `fit` and `tune` subsets for hyperparameter tuning.
3. **Feature Boundaries**: Popularity counts are derived solely from training impressions; session positions count only strictly prior impressions; freshness is bounded $\ge 0$; and users without click histories receive neutral default representations.

All conditions are verified by running:
```bash
python -m pytest pipeline/tests/test_no_leakage.py -v
```

---

## Codabench Submissions (Q5)

Submission archives are generated using chunked streaming over unlabelled test splits, utilizing rank fusion (BM25 + embeddings) with a train-popularity fallback:

- **MIND**: `Q5_MIND/prediction.zip` containing `prediction.txt` (2,370,727 rows).
- **EB-NeRD**: `Q5_EB-NeRd/predictions.zip` containing `predictions.txt` (13,536,710 rows).

Both files have been verified to match exact test impression counts with valid 1-indexed rank permutations per row.
