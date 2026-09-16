"""Train-only click popularity -- shared by A1's Q4/Q5 and A2's Q1 behavioural features.

Moved out of evaluate.py/generate_submission.py (which each defined their own identical
copy) so A2's feature engineering reuses the exact same boundary: popularity is always
counted from `behaviors_train.parquet` only, never from val/test, since it stands in for
"what's popular based on everything known before serving time."
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def train_popularity(fs: Path) -> dict:
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["candidates", "labels"])
    counter = Counter()
    for cands, labels in zip(train["candidates"], train["labels"]):
        for c, l in zip(cands, labels):
            if l == 1:
                counter[c] += 1
    return dict(counter)
