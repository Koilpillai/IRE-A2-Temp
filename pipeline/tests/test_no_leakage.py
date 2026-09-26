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
    that's scoring it, i.e. future information leaking backwards.

    Checked on freshness_hours_raw (the UNCLAMPED value) rather than freshness_hours
    itself: pipeline/features.py clamps the model-facing feature to >=0 by
    construction, which would make this assertion vacuously true regardless of any
    real underlying bug -- see freshness_hours_raw's own docstring there. A small
    negative tolerance (a few hours) is allowed for real-world clock skew/batching in
    the raw timestamps; anything beyond that is treated as real leakage."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["freshness_hours_raw"])
    NOISE_TOLERANCE_HOURS = 6.0  # generous clock-skew allowance; real leakage would be systematic, not a few hours
    violations = df[df["freshness_hours_raw"] < -NOISE_TOLERANCE_HOURS]
    assert len(violations) == 0, (
        f"{dataset}/{split}: {len(violations)} rows have freshness_hours_raw below "
        f"-{NOISE_TOLERANCE_HOURS}h (worst={df['freshness_hours_raw'].min():.2f}h) -- "
        f"a candidate's publish/first-appearance time was computed as meaningfully "
        f"AFTER the impression scoring it")


@pytest.mark.parametrize("dataset", DATASETS)
def test_features_popularity_is_train_only(dataset):
    """popularity_log on the VAL split must equal log1p(TRAIN-only click count) exactly
    -- i.e. it must never have been recomputed from val's own labels. (Train-split rows
    are checked separately by test_features_popularity_excludes_own_click below, since
    train positives get a leave-one-out adjustment that val rows don't need.)"""
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
def test_features_popularity_excludes_own_click(dataset):
    """A TRAIN row's own click must never be counted in its own popularity_log feature.

    train_popularity() counts clicks over the ENTIRE train split (by design -- it's
    meant to approximate "popularity as of serving time" for every split). Featurizing
    a train impression whose candidate has label==1 with that raw count would fold the
    row's own click into its own feature -- a direct label leak that was measured
    pre-fix: 100% of MIND/EB-NeRD train positives had popularity_log > 0 (vs ~98% of
    negatives), and EB-NeRD's baseline-feature ablation AUC came out below 0.5. This
    regression-tests the leave-one-out fix in pipeline/features.py (search for
    "Leave-one-impression-out on TRAIN").
    """
    fs = config.FEATURE_STORE / dataset
    path = fs / "behavioral_features_train.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split train` first")
    from pipeline.common.popularity import train_popularity
    pop = train_popularity(fs)
    full = pd.read_parquet(path, columns=["candidate_id", "label", "popularity_log"])
    positives = full[full["label"] == 1]
    df = positives.sample(n=min(2000, len(positives)), random_state=config.RANDOM_SEED)
    raw_count = df["candidate_id"].map(lambda c: pop.get(c, 0)).astype(float)
    leave_one_out_expected = np.log1p(np.clip(raw_count - 1, 0, None))
    raw_unadjusted = np.log1p(raw_count)
    actual = df["popularity_log"].to_numpy()
    assert np.allclose(actual, leave_one_out_expected.to_numpy(), atol=1e-6), (
        f"{dataset}: train-split popularity_log on positive rows doesn't match a "
        f"leave-one-out recount -- looks like it still includes each row's own click")
    # Extra guard against a no-op "fix" that just relabels the same leaking value: for
    # any candidate clicked more than once in train, leave-one-out must actually differ
    # from the raw (unadjusted) count -- if it never does, the subtraction isn't wired in.
    differs = ~np.isclose(actual, raw_unadjusted.to_numpy(), atol=1e-6)
    if (raw_count > 1).any():
        assert differs[raw_count.to_numpy() > 1].any(), (
            f"{dataset}: leave-one-out adjustment never changes popularity_log for "
            f"repeat-clicked candidates -- the leak fix may not actually be wired in")


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
    """Every user's chronologically-first impression in a session must be 0 (position
    counts impressions STRICTLY BEFORE this one; a total-session-size feature would
    leak the session's future, which is exactly what this guards)."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["impression_id", "user_id", "session_position"]).drop_duplicates("impression_id")
    min_position_per_user = df.groupby("user_id")["session_position"].min()
    assert (min_position_per_user == 0).all(), (
        f"{dataset}/{split}: some user's every impression has session_position > 0 "
        f"-- a session's first impression should always start counting at 0")


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("split", ["train", "val"])
def test_features_session_position_is_bounded_and_causal(dataset, split):
    """session_position must never exceed (that user's total impressions in this split
    - 1) -- a value equal to or above the user's own impression count would mean the
    feature was computed from information not yet available (the session's eventual
    future size), not strictly-prior impressions only. This is the upper-bound half of
    the causality guarantee that test_features_session_position_starts_at_zero (the
    lower-bound half) doesn't check on its own -- min()==0 alone can't catch a position
    counter that runs past the end of what's actually observed so far."""
    fs = config.FEATURE_STORE / dataset
    path = fs / f"behavioral_features_{split}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built -- run `python -m pipeline.features --dataset {dataset} --split {split}` first")
    df = pd.read_parquet(path, columns=["impression_id", "user_id", "session_position"]).drop_duplicates("impression_id")
    counts = df.groupby("user_id")["session_position"].transform("count")
    max_allowed = counts - 1
    violations = df[df["session_position"] > max_allowed]
    assert len(violations) == 0, (
        f"{dataset}/{split}: {len(violations)} rows have session_position exceeding "
        f"that user's own total impression count in this split -- position is "
        f"counting impressions that haven't happened yet")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_ebnerd_history_precedes_split_window(split):
    """The actual behaviour-window boundary Q9 asks to enforce: every item in a user's
    click history must have a timestamp strictly before that split's own window starts
    -- i.e. history genuinely comes from BEFORE the impressions it's used to predict,
    not just "some other file" that happens to be called history.parquet. Checked
    directly against EB-NeRD's real per-item impression_time_fixed timestamps (the one
    dataset here that carries them; MIND's raw files have no per-item history
    timestamp at all -- order only, see pipeline/features.py's module docstring)."""
    import numpy as np
    split_dir = config.EBNERD_SMALL_DIR / split
    behaviors = pd.read_parquet(split_dir / "behaviors.parquet", columns=["impression_time"])
    history = pd.read_parquet(split_dir / "history.parquet", columns=["user_id", "impression_time_fixed"])

    split_start = behaviors["impression_time"].min()
    max_history_time_per_user = history["impression_time_fixed"].apply(
        lambda times: np.max(times) if times is not None and len(times) else None)
    violations = max_history_time_per_user[max_history_time_per_user.notna() & (max_history_time_per_user >= split_start)]
    assert len(violations) == 0, (
        f"ebnerd/{split}: {len(violations)} users have a history item at or after "
        f"this split's own window start ({split_start}) -- history is not strictly "
        f"prior to the impressions it's used to featurize")


if __name__ == "__main__":
    for ds in DATASETS:
        test_internal_fit_tune_no_overlap(ds)
        test_fit_tune_reconstruct_train(ds)
        test_train_val_no_overlap(ds)
        test_val_candidates_include_official_labels_only(ds)
        for split in ["train", "val"]:
            test_features_freshness_non_negative(ds, split)
            test_features_empty_history_is_neutral(ds, split)
            test_features_session_position_starts_at_zero(ds, split)
            test_features_session_position_is_bounded_and_causal(ds, split)
        test_features_popularity_is_train_only(ds)
        test_features_popularity_excludes_own_click(ds)
        print(f"[{ds}] no-leakage checks: PASSED")
    for split in ["train", "validation"]:
        test_ebnerd_history_precedes_split_window(split)
        print(f"[ebnerd] history-precedes-window check ({split}): PASSED")
