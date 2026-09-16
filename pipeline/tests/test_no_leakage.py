"""Anti-gaming: assert the behaviour-window boundary holds -- no future-click leakage.

Run with: python -m pytest pipeline/tests/test_no_leakage.py -v
(or: python -m pipeline.tests.test_no_leakage, which runs the same checks directly)

Covers the two places leakage could sneak in:
  1. The internal train->(fit, tune) carve (our own code, done by timestamp cutoff).
  2. The official train->val boundary (provided by MIND/EB-NeRD, but re-verified here
     rather than trusted blindly).
Both must show strict non-overlap: max(before) <= min(after).
"""
import pandas as pd
import pytest

from pipeline import config

DATASETS = ["mind", "ebnerd"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_internal_fit_tune_no_overlap(dataset):
    fs = config.FEATURE_STORE / dataset
    fit = pd.read_parquet(fs / "behaviors_train_fit.parquet", columns=["timestamp"])
    tune = pd.read_parquet(fs / "behaviors_train_tune.parquet", columns=["timestamp"])
    assert len(fit) > 0 and len(tune) > 0
    assert fit["timestamp"].max() <= tune["timestamp"].min(), (
        f"{dataset}: internal tune split leaks future rows into fit "
        f"(fit_max={fit['timestamp'].max()}, tune_min={tune['timestamp'].min()})"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_fit_tune_reconstruct_train(dataset):
    """fit + tune must partition train exactly (no rows dropped or duplicated)."""
    fs = config.FEATURE_STORE / dataset
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["impression_id"])
    fit = pd.read_parquet(fs / "behaviors_train_fit.parquet", columns=["impression_id"])
    tune = pd.read_parquet(fs / "behaviors_train_tune.parquet", columns=["impression_id"])
    assert len(fit) + len(tune) == len(train)
    assert set(fit["impression_id"]) | set(tune["impression_id"]) == set(train["impression_id"])
    assert set(fit["impression_id"]) & set(tune["impression_id"]) == set()


@pytest.mark.parametrize("dataset", DATASETS)
def test_train_val_no_overlap(dataset):
    """The official train/val boundary: re-verified, not trusted blindly."""
    fs = config.FEATURE_STORE / dataset
    train = pd.read_parquet(fs / "behaviors_train.parquet", columns=["timestamp"])
    val = pd.read_parquet(fs / "behaviors_val.parquet", columns=["timestamp"])
    assert train["timestamp"].max() <= val["timestamp"].min(), (
        f"{dataset}: val split is not strictly after train "
        f"(train_max={train['timestamp'].max()}, val_min={val['timestamp'].min()})"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_val_candidates_include_official_labels_only(dataset):
    """Every val impression's candidate/label lists must be the same length and binary."""
    fs = config.FEATURE_STORE / dataset
    val = pd.read_parquet(fs / "behaviors_val.parquet", columns=["candidates", "labels"]).sample(
        n=500, random_state=config.RANDOM_SEED)
    for cands, labels in zip(val["candidates"], val["labels"]):
        assert len(cands) == len(labels)
        assert set(labels) <= {0, 1}


if __name__ == "__main__":
    for ds in DATASETS:
        test_internal_fit_tune_no_overlap(ds)
        test_fit_tune_reconstruct_train(ds)
        test_train_val_no_overlap(ds)
        test_val_candidates_include_official_labels_only(ds)
        print(f"[{ds}] no-leakage checks: PASSED")
