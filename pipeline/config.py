"""Raw data paths and shared constants. Single source of truth for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MIND_TRAIN_DIR = ROOT / "MINDsmall_train" / "MINDsmall_train"
MIND_DEV_DIR = ROOT / "MINDsmall_dev" / "MINDsmall_dev"
MIND_TEST_DIR = ROOT / "MINDlarge_test" / "MINDlarge_test"

EBNERD_SMALL_DIR = ROOT / "ebnerd_small"
EBNERD_TESTSET_DIR = ROOT / "ebnerd_testset" / "ebnerd_testset"

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
