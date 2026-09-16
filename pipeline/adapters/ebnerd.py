"""Parse raw EB-NeRD parquet files into the unified schema shared with MIND.

EB-NeRD ships history separately from behaviors (one row per user); labels are derived
by matching article_ids_inview against article_ids_clicked per impression.

Pure pandas/pyarrow (no polars) -- see README for why. The 13.5M-row test file is
streamed row-group by row-group via pyarrow.parquet (same pattern the course-provided
ebnerd_analysis.ipynb used), which bounds peak memory without needing polars' lazy scan.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def load_articles(*articles_paths: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=["article_id", "title", "subtitle", "category_str"])
              for p in articles_paths]
    articles = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["article_id"], keep="first")
    articles["text_lexical"] = articles["title"].fillna("") + " " + articles["subtitle"].fillna("")
    articles = articles.rename(columns={"category_str": "category"})
    articles["article_id"] = articles["article_id"].astype(int)
    return articles[["article_id", "text_lexical", "category"]].reset_index(drop=True)


def load_user_history(history_path: Path) -> pd.DataFrame:
    """Only user_id + article_id_fixed -- dropping the other 3 list columns keeps this small."""
    df = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed"])
    return df.rename(columns={"article_id_fixed": "history"})


def load_user_engagement_history(history_path: Path) -> pd.DataFrame:
    """Full per-user history incl. per-item timestamp/read_time/scroll_percentage (A2 Q1).

    EB-NeRD's own per-history-item timestamps let recency-weighting use real elapsed
    time instead of the position-rank proxy MIND is stuck with (MIND's news.tsv has no
    per-item timestamps at all -- see pipeline/features.py). `*_fixed` lists are already
    chronological (oldest -> most recent), same convention as `article_id_fixed`.
    """
    df = pd.read_parquet(history_path, columns=[
        "user_id", "article_id_fixed", "impression_time_fixed",
        "read_time_fixed", "scroll_percentage_fixed",
    ])
    return df.rename(columns={"article_id_fixed": "history"})


def load_publish_times(*articles_paths: Path) -> pd.DataFrame:
    """article_id -> published_time, unioned across splits (mirrors load_articles)."""
    frames = [pd.read_parquet(p, columns=["article_id", "published_time"]) for p in articles_paths]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["article_id"], keep="first")
    out["article_id"] = out["article_id"].astype(int)
    return out.reset_index(drop=True)


def load_session_ids(behaviors_path: Path) -> pd.DataFrame:
    """impression_id -> session_id only. MIND has no session concept (see features.py,
    which buckets MIND sessions from timestamp gaps instead)."""
    return pd.read_parquet(behaviors_path, columns=["impression_id", "session_id"])


def _attach_labels_and_history(behaviors: pd.DataFrame, history: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
    # IMPORTANT: do NOT merge the full history list onto every behavior row -- a user's
    # history is constant within a split but repeats across ~15 impressions on average,
    # so merging the list column duplicates it ~15x and (via the Python-object overhead of
    # exploding numpy arrays into per-row lists) inflated peak RSS from ~0.3GB to ~3GB for
    # ebnerd_small/train alone. Only the *length* is joined per row (a scalar, cheap); the
    # actual history stays in its own small per-user table (see extract_user_history /
    # build_pipeline.py), joined by user_id only where it's transiently needed.
    len_by_user = pd.DataFrame({
        "user_id": history["user_id"], "history_len": history["history"].apply(len),
    })
    behaviors = behaviors.merge(len_by_user, on="user_id", how="left")
    behaviors["history_len"] = behaviors["history_len"].fillna(0).astype(int)

    def _as_list(x):
        return [int(v) for v in x] if x is not None and hasattr(x, "__len__") else []

    behaviors["candidates"] = behaviors["article_ids_inview"].apply(_as_list)

    if has_labels:
        inview_col = behaviors["article_ids_inview"].tolist()
        clicked_col = behaviors["article_ids_clicked"].tolist()
        labels_col = []
        for inview, clicked in zip(inview_col, clicked_col):
            clicked_set = {int(v) for v in clicked} if clicked is not None and len(clicked) else set()
            labels_col.append([1 if int(a) in clicked_set else 0 for a in inview])
        behaviors["labels"] = labels_col
    else:
        behaviors["labels"] = None

    behaviors = behaviors.rename(columns={"impression_time": "timestamp"})
    return behaviors[["impression_id", "user_id", "timestamp", "history_len",
                       "candidates", "labels"]].reset_index(drop=True)


def load_behaviors(behaviors_path: Path, history_path: Path, has_labels: bool) -> pd.DataFrame:
    """Direct load for modest-size files (ebnerd_small train/validation: ~230-245K rows).

    Does not include the `history` list itself (see `_attach_labels_and_history`) -- use
    `load_user_history(history_path)` for that, joined by user_id where actually needed.
    """
    cols = ["impression_id", "user_id", "impression_time", "article_ids_inview"]
    if has_labels:
        cols.append("article_ids_clicked")
    behaviors = pd.read_parquet(behaviors_path, columns=cols)
    history = load_user_history(history_path)
    return _attach_labels_and_history(behaviors, history, has_labels)


def stream_behaviors_chunks(behaviors_path: Path, history_path: Path, has_labels: bool):
    """Row-group generator for the 13.5M-row test file -- bounds peak memory."""
    history = load_user_history(history_path)
    cols = ["impression_id", "user_id", "impression_time", "article_ids_inview"]
    if has_labels:
        cols.append("article_ids_clicked")
    pf = pq.ParquetFile(behaviors_path)
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=cols).to_pandas()
        yield _attach_labels_and_history(chunk, history, has_labels)
