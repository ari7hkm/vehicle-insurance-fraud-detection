from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
MODELS_PATH = ARTIFACTS_PATH / "models/random_forest.pkl"

DATA_PATH = "data/fraud_oracle.csv"
TARGET_COLUMN = "FraudFound_P"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 700
MAX_DEPTH = 12
MIN_SAMPLES_SPLIT = 6
MIN_SAMPLES_LEAF = 5
VERBOSE = 1
MAX_FEATURES = "log2"
PARAM_DIST = {
    "n_estimators": [300, 400, 500, 600, 600, 700, 800],
    "max_depth": [8, 9, 10, 11, 12, 13],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}