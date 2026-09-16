"""Parse raw MIND TSV files into the unified schema shared with EB-NeRD.

Unified article table:  article_id, text_lexical, category
Unified behaviors table: impression_id, user_id, timestamp, history, history_len,
                          candidates, labels (None for unlabeled test)

Pure pandas/pyarrow (no polars) -- see README for why: this sandbox's network egress
could not pull large wheels in reasonable time, so the pipeline was built on libraries
already installed. Large files are read with pandas' `chunksize` iterator to bound
peak memory instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLS = ["article_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_articles(*news_tsv_paths: Path) -> pd.DataFrame:
    """Union + dedup news.tsv across splits (train/dev/test each carry different subsets)."""
    frames = []
    for path in news_tsv_paths:
        df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, quoting=3, encoding="utf-8")
        frames.append(df[["article_id", "category", "title", "abstract"]])
    articles = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["article_id"], keep="first")
    articles["text_lexical"] = articles["title"].fillna("") + " " + articles["abstract"].fillna("")
    return articles[["article_id", "text_lexical", "category"]].reset_index(drop=True)


def _parse_history(s) -> list:
    if not isinstance(s, str) or not s:
        return []
    return s.split(" ")


def _parse_impressions_labeled(s: str):
    toks = s.split(" ")
    ids = [None] * len(toks)
    labels = [0] * len(toks)
    for i, t in enumerate(toks):
        cut = t.rfind("-")
        ids[i] = t[:cut]
        labels[i] = int(t[cut + 1:])
    return ids, labels


def _process_chunk(df: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].apply(_parse_history)
    df["history_len"] = df["history"].apply(len)
    if has_labels:
        parsed = df["impressions"].apply(_parse_impressions_labeled)
        df["candidates"] = parsed.apply(lambda x: x[0])
        df["labels"] = parsed.apply(lambda x: x[1])
    else:
        df["candidates"] = df["impressions"].str.split(" ")
        df["labels"] = None
    return df[["impression_id", "user_id", "timestamp", "history", "history_len",
               "candidates", "labels"]].reset_index(drop=True)


def extract_user_history(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup per-user history within a split (verified invariant per user per split in MIND;
    see design note). Mirrors ebnerd.load_user_history so both datasets end up with the same
    feature-store shape: behaviors_<split>.parquet carries only history_len, and the actual
    history lists live in their own small user_history_<split>.parquet, joined by user_id
    only where actually needed -- avoids re-duplicating each user's history across every one
    of their impression rows (see ebnerd.py for the concrete memory blowup this caused)."""
    return df[["user_id", "history"]].drop_duplicates(subset="user_id", keep="first").reset_index(drop=True)


def load_behaviors(path: Path, has_labels: bool) -> pd.DataFrame:
    """Direct load for modest-size files (MIND train/dev: <=160K rows)."""
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, quoting=3,
                      dtype={"impression_id": "int64", "user_id": "str"})
    return _process_chunk(df, has_labels)


def stream_behaviors_chunks(path: Path, has_labels: bool, chunk_rows: int = 200_000):
    """Chunked generator for the large (2.37M-row) test file -- bounds peak memory."""
    reader = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, quoting=3,
                          dtype={"impression_id": "int64", "user_id": "str"}, chunksize=chunk_rows)
    for chunk in reader:
        yield _process_chunk(chunk, has_labels)
