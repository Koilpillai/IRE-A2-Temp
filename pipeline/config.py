"""Raw data paths and shared constants. Single source of truth for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = ROOT / "datasets"

MIND_TRAIN_DIR = DATASETS / "mind" / "train" / "MINDsmall_train"
MIND_DEV_DIR = DATASETS / "mind" / "dev" / "MINDsmall_dev"
MIND_TEST_DIR = DATASETS / "mind" / "test" / "MINDlarge_test"

EBNERD_SMALL_DIR = DATASETS / "ebnerd" / "small"
EBNERD_TESTSET_DIR = DATASETS / "ebnerd" / "testset" / "ebnerd_testset"

FEATURE_STORE = ROOT / "feature_store"
OUTPUTS = ROOT / "outputs"

Q5_MIND_DIR = ROOT / "Q5_MIND"
Q5_EBNERD_DIR = ROOT / "Q5_EB-NeRd"

# Internal temporal carve of each dataset's official "train" split, used only for
# hyperparameter/index sanity checks -- the official val/dev split is never touched
# until final reporting (Q4).
INTERNAL_TUNE_HOLDOUT_FRACTION = 0.15

RANDOM_SEED = 42

# BM25 hyperparameters (Okapi BM25, standard defaults).
BM25_K1 = 1.5
BM25_B = 0.75

# Word2Vec hyperparameters (shared by both datasets for a fair lexical-vs-semantic comparison).
W2V_DIM = 128
W2V_WINDOW = 8
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_WORKERS = 20

RECALL_KS = [50, 100, 200]
DEV_SAMPLE_FOR_TESTING = 750  # per user instruction: 500-1000 rows for smoke tests

# --- A2 Q1: behavioural feature engineering ---
FEATURE_HISTORY_WINDOW = 50  # matches A1's retrieval HISTORY_WINDOW, for a consistent query definition
HIST_DECAY_RANK = 0.9        # per-position decay for recency weighting when no per-item timestamp exists (MIND)
HIST_HALF_LIFE_HOURS = 72.0  # 3-day half-life for time-based recency decay (EB-NeRD, which has per-item timestamps)
FEATURE_BUILD_CHUNK = 20_000  # impressions per chunk when exploding to (impression, candidate) rows
