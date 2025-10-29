from pathlib import Path

# ==== PATHS ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "mutual-fund-data.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


def safe_mkdir(path: Path):
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Expected directory but found file: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)

safe_mkdir(REPORTS_DIR)
safe_mkdir(FIGURES_DIR)
safe_mkdir(METRICS_DIR)

MODEL_COMPARISON_PLOT = FIGURES_DIR / "model_comparison.png"
BASELINE_METRICS_JSON = METRICS_DIR / "baseline_metrics.json"
OPTIMIZED_METRICS_JSON = METRICS_DIR / "optimized_metrics.json"
COEFFICIENTS_CSV = METRICS_DIR / "logreg_coefficients_by_class.csv"

# ==== MODEL TARGET ====
TARGET_COL = "Scheme_Type"

# Numeric columns (same for both models)
NUMERIC_FEATURES = [
    "NAV",
    "Average_AUM_Cr",
]

# Baseline sees ALL categorical features
CATEGORICAL_FEATURES_BASELINE = [
    "Scheme_Category",
    "Scheme_Min_Amt",
    "AAUM_Quarter",
]

# Optimised model will NOT see Scheme_Category.
# This prevents perfect linear separation of rare classes.
CATEGORICAL_FEATURES_OPT = [
    "Scheme_Min_Amt",
    "AAUM_Quarter",
]

TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5
