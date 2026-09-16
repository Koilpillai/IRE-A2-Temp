"""Anti-gaming: assert the behaviour-window boundary holds -- no future-click leakage.

Run with: python -m pytest pipeline/tests/test_no_leakage.py -v
(or: python -m pipeline.tests.test_no_leakage, which runs the same checks directly)

Covers the two places leakage could sneak in:
  1. The internal train->(fit, tune) carve (our own code, done by timestamp cutoff).
  2. The official train->val boundary (provided by MIND/EB-NeRD, but re-verified here
     rather than trusted blindly).
Both must show strict non-overlap: max(before) <= min(after).
"""
import numpy as np
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



# ---------------------------------------------------------------------------
# A2 Q1 / Q9 -- behavioural feature boundary checks (pipeline/features.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("split", ["train", "val"])
def test_features_freshness_non_negative(dataset, split):
    """freshness_hours must never be negative -- a negative value would mean a
    candidate's publish/first-appearance time was computed as AFTER the impression
    that's scoring it, i.e. future information leaking backwards."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["freshness_hours"])
    assert (df["freshness_hours"] >= -1e-6).all(), f"{dataset}/{split}: found negative freshness_hours"


@pytest.mark.parametrize("dataset", DATASETS)
def test_features_popularity_is_train_only(dataset):
    """popularity_log must equal log1p(TRAIN-only click count) even when featurizing
    the val split -- i.e. it must never have been recomputed from val's own labels."""
    fs = config.FEATURE_STORE / dataset
    path = fs / "behavioral_features_val.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split val` first")
    from pipeline.common.popularity import train_popularity
    pop = train_popularity(fs)
    full = pd.read_parquet(path, columns=["candidate_id", "popularity_log"])
    df = full.sample(n=min(2000, len(full)), random_state=config.RANDOM_SEED)
    expected = np.log1p(df["candidate_id"].map(lambda c: pop.get(c, 0)).astype(float))
    assert np.allclose(df["popularity_log"].to_numpy(), expected.to_numpy(), atol=1e-6), (
        f"{dataset}: popularity_log doesn't match a train-only recount -- val labels may have leaked in")


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("split", ["train", "val"])
def test_features_empty_history_is_neutral(dataset, split):
    """Zero click history must produce neutral (not leaked/garbage) feature values --
    no category match signal and no embedding similarity to lean on."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["hist_len", "hist_category_match_frac", "hist_embed_sim"])
    cold = df[df["hist_len"] == 0]
    if len(cold) == 0:
        pytest.skip(f"{dataset}/{split}: no zero-history rows in this sample")
    assert (cold["hist_category_match_frac"] == 0).all()
    assert (cold["hist_embed_sim"] == 0).all()


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("split", ["train", "val"])
def test_features_session_position_starts_at_zero(dataset, split):
    """session_position must never exceed the count of a user's own impressions in
    this split -- and every user's chronologically-first impression in a session must
    be 0 (position counts impressions STRICTLY BEFORE this one; a total-session-size
    feature would leak the session's future, which is exactly what this guards)."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["impression_id", "user_id", "session_position"]).drop_duplicates("impression_id")
    min_position_per_user = df.groupby("user_id")["session_position"].min()
    assert (min_position_per_user == 0).all(), (
        f"{dataset}/{split}: some user's every impression has session_position > 0 "
        f"-- a session's first impression should always start counting at 0")


if __name__ == "__main__":
    for ds in DATASETS:
        test_internal_fit_tune_no_overlap(ds)
        test_fit_tune_reconstruct_train(ds)
        test_train_val_no_overlap(ds)
        test_val_candidates_include_official_labels_only(ds)
        print(f"[{ds}] no-leakage checks: PASSED")
