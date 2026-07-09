from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
MODELS_PATH = ARTIFACTS_PATH / "models/random_forest.pkl"
METRICS_PATH = ARTIFACTS_PATH / "metrics/random_forest_metrics.json"
PLOTS_PATH = ARTIFACTS_PATH / "plots"

DATA_PATH = PROJECT_ROOT / "data/fraud_oracle.csv"
TARGET_COLUMN = "FraudFound_P"
TEST_SIZE = 0.2
RANDOM_STATE = 42
PARAM_DIST = {
    "n_estimators": [300, 400, 500, 600, 700, 800],
    "max_depth": [8, 9, 10, 11, 12, 13],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}
