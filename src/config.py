from pathlib import Path

# =========================
#         PATHS
# =========================

# Project root (…/MUTUAL-FUND-ML/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CSV used by EDA / data_prep
# (same as original project)
DATA_PATH = PROJECT_ROOT / "data" / "mutual-fund-data.csv"

# Reports and subfolders
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Models directory (for saved .joblib files)
MODELS_DIR = PROJECT_ROOT / "models"


def safe_mkdir(path: Path) -> None:
    """Create directory if it doesn't exist; fail if a file is in the way."""
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Expected directory but found file: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


# Ensure all important dirs exist
for _p in [REPORTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    safe_mkdir(_p)

# Old paths used by the original project (keep them for compatibility)
MODEL_COMPARISON_PLOT = FIGURES_DIR / "model_comparison.png"
BASELINE_METRICS_JSON = METRICS_DIR / "baseline_metrics.json"
OPTIMIZED_METRICS_JSON = METRICS_DIR / "optimized_metrics.json"
COEFFICIENTS_CSV = METRICS_DIR / "logreg_coefficients_by_class.csv"

# New paths for the unified models + API
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_METRICS_PATH = METRICS_DIR / "best_model_api_summary.json"


# =========================
#       MODEL TARGET
# =========================

TARGET_COL = "Scheme_Type"  # unchanged from original


# =========================
#      FEATURE LISTS
# =========================

# Numeric columns (same for both models)
NUMERIC_FEATURES = [
    "NAV",
    "Average_AUM_Cr",
]

# Baseline sees ALL categorical features (original behaviour)
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


# =========================
#   TRAIN/TEST SETTINGS
# =========================

TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5
