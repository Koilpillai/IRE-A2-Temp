# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on MIND and EB-NeRD

One pipeline for news recommendation, run on both Microsoft's **MIND** dataset and
Ekstra Bladet's **EB-NeRD** dataset. Both datasets get parsed down into the same shape,
so BM25, the embeddings, and the evaluation code are all written once and shared —
only the raw-file parsing differs between the two.

## One-command reproduce

```bash
pip install -r requirements.txt

make data      # Q1: raw files -> feature_store/{mind,ebnerd}/  (~20s total)
make retrieve  # Q2+Q3: BM25 + Word2Vec, recall@{50,100,200}
make evaluate  # Q4: AUC/MRR/nDCG, diversity/novelty/coverage, slices, bootstrap CIs
make test      # anti-leakage checks on the temporal splits
make submit    # Q5: streams the Codabench test sets -> Q5_MIND/, Q5_EB-NeRd/

# or just run everything:
make all
```

Everything above also runs on one dataset at a time with `--dataset mind` or
`--dataset ebnerd`, e.g. `python3 -m pipeline.retrieval_eval --dataset mind --sample
750` for a quick sanity check before running the real thing. Numbers from a sample
that small are noisy, though — the actual results used in the design note are from
full runs.

## Where the raw data is expected to be

```
MINDsmall_train/MINDsmall_train/{news,behaviors}.tsv          -- Q1 train
MINDsmall_dev/MINDsmall_dev/{news,behaviors}.tsv               -- Q1 val
MINDlarge_test/MINDlarge_test/{news,behaviors}.tsv              -- Q5 only (no labels)
ebnerd_small/articles.parquet, {train,validation}/*.parquet     -- Q1 train/val
ebnerd_testset/ebnerd_testset/{articles,test/*}.parquet         -- Q5 only (no labels)
```

The pipeline doesn't download these itself — see the "environment" note below for why,
and `pipeline/config.py` for the exact paths it reads.

## Layout

```
pipeline/
  common/          shared between both datasets
    text.py          tokenizer
    bm25.py          BM25 index + scoring
    embeddings.py    Word2Vec + ANN similarity search (FAISS if available, else a
                     from-scratch IVF/k-means index; GPU if available, else numpy)
    metrics.py       AUC/MRR/nDCG, diversity/novelty/coverage, bootstrap CI
    split.py         temporal train/val splitting
    submission.py    Q5 rank-fusion + Codabench file writer
  adapters/        parses raw MIND/EB-NeRD files into the shared schema
    mind.py, ebnerd.py
  build_pipeline.py       Q1
  retrieval_eval.py       Q2 + Q3
  evaluate.py             Q4
  generate_submission.py  Q5
  tests/test_no_leakage.py

feature_store/<dataset>/     Q1 output — small parquet files, article + user features
outputs/<dataset>/           Q2-Q4 results (recall numbers, eval metrics, saved scores)
Q5_MIND/, Q5_EB-NeRd/        Q5 deliverables + leaderboard screenshots
```

## The shared schema

Both datasets end up as the same two tables, which is the whole reason the shared
code in `pipeline/common/` can stay dataset-agnostic:

- **articles**: `article_id, text_lexical (title+abstract/subtitle), category`
- **behaviors**: `impression_id, user_id, timestamp, history_len, candidates, labels`
- **user_history** (its own table, keyed by `user_id`): the list of articles a user
  clicked before this split started

One thing worth flagging: `history` is *not* a column on the behaviors table. A
user's click history doesn't change within a split, but a user shows up in ~15
impressions on average — so attaching their history to every one of those rows means
storing the same list ~15 times over. That's actually what crashed the first version
of the EB-NeRD build (pushed memory from ~0.3GB to ~3GB for `ebnerd_small/train`
alone). Now it lives in its own small table and only gets looked up by `user_id` when
it's actually needed for building a query.

## Splitting by time, not randomly (Q1.3 / Q9)

Two layers here:
1. MIND's train/dev/test and EB-NeRD's train/validation/test are already
   non-overlapping calendar weeks — checked this against the raw timestamps directly
   rather than just assuming it. Train vs. val is what all the Q4 numbers use.
2. On top of that, `carve_internal_validation` splits train itself into an earlier
   "fit" chunk and a later "tune" chunk, so things like BM25's k1/b or the history
   window size can be sanity-checked without ever touching the real val set.
   `pipeline/tests/test_no_leakage.py` checks both of these boundaries hold on the
   actual built feature store, not just by trusting the split code.

## Why some of this looks the way it does

This ran on a machine with 20 CPU cores and a 6GB GPU, but only about 7.6GB of RAM
(often less — something else on the box kept holding onto ~2GB of swap), and a PyPI
download speed that measured around 30-40KB/s when actually timed (and, in a later
session, the network was down outright at the DNS level). That ruled out installing
`torch`, `polars`, or `faiss` — a GPU-enabled torch wheel alone would've taken hours at
that speed, if it could connect at all — so this pipeline runs on plain `pandas` +
`pyarrow` with manual chunking for the huge test files instead. `embeddings.py` still
checks for `torch`/`faiss` and uses them automatically if importable (GPU brute-force
search, and a real FAISS `IndexHNSWFlat` ANN index); this run took the fallback path
for both. Since `faiss` wasn't installable, the fallback ANN index isn't a plain
brute-force scan either — it's an inverted-file (IVF) index built from scratch with
numpy + sklearn's k-means, the same core idea as FAISS's own `IndexIVFFlat`. Measured
against exact brute force on the real corpus before committing to it: ~95% recall@200
at a ~12x smaller candidate pool. (A random-hyperplane LSH index was tried first and
dropped — these Word2Vec vectors turned out to be strongly anisotropic, which tanked
LSH's recall to ~70% with barely any speedup; k-means-based IVF adapts to the actual
shape of the data instead of assuming roughly-uniform coverage of the unit sphere.
Full comparison numbers are in the design note.)

Getting the full-corpus retrieval and the huge Q5 test files to actually fit in that
RAM budget took a few rounds of real crashes and fixes along the way — the details are
in the design note, since they were genuinely useful things to learn (batching a
"vectorized" operation isn't the same as it actually staying small in memory).

## Anti-leakage (Q9)

`pipeline/tests/test_no_leakage.py` checks the split boundaries hold on the actual
built data rather than just trusting the code that made them. Q5 also keeps track of
how many impressions had to fall back to plain popularity because there was no click
history to build a query from, instead of quietly mixing those into everything else.

## Q5: Codabench submission

`Q5_MIND/prediction.zip` and `Q5_EB-NeRd/predictions.zip` are already built in the
exact format each competition's own submission page describes (checked against
screenshots of those pages, not from memory): a zip with just one file in it —
`prediction.txt` or `predictions.txt` — one line per impression, ranks lined up
against that impression's own candidate list. Registering on Codabench and actually
clicking submit needs an account, so that part is still a manual step.

## What's not in this repo

Raw datasets, `feature_store/`, and `outputs/` are all things `make all` regenerates,
so they're left out via `.gitignore`, per Q8's "no large files in git" rule.
