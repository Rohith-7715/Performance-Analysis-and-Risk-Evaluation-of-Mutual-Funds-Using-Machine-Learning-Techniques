# train_risk_model.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve,
    StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from src.utils import divider
from src.eda import run_eda
from src.mutual_funds_models_all_in_one import run_models as run_all_models
from src.config import DATA_PATH  # points to data/mutual-fund-data.csv


# -------------------- risk mapping --------------------

low_cats = [
    "Debt Scheme - Liquid Fund",
    "Debt Scheme - Overnight Fund",
    "Debt Scheme - Money Market Fund",
    "Debt Scheme - Low Duration Fund",
    "Debt Scheme - Ultra Short Duration Fund",
    "Debt Scheme - Gilt Fund",
    "Debt Scheme - Gilt Fund with 10 year constant duration",
    "Debt Scheme - Banking and PSU Fund",
    "Debt Scheme - Corporate Bond Fund",
    "Debt Scheme - Floater Fund",
    "Liquid",
    "Money Market",
]

medium_cats = [
    "Hybrid Scheme - Conservative Hybrid Fund",
    "Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage",
    "Hybrid Scheme - Equity Savings",
    "Hybrid Scheme - Arbitrage Fund",
    "Hybrid Scheme - Multi Asset Allocation",
    "Hybrid Scheme - Balanced Hybrid Fund",
    "Debt Scheme - Medium Duration Fund",
    "Debt Scheme - Medium to Long Duration Fund",
    "Debt Scheme - Long Duration Fund",
    "Debt Scheme - Dynamic Bond",
    "Other Scheme - Index Funds",
    "Other Scheme - FoF Domestic",
    "Solution Oriented Scheme - Children s Fund",
    "Solution Oriented Scheme - Retirement Fund",
    "Income",
    "Assured Return",
    "Balanced",
    "Growth",
]

high_cats = [
    "Equity Scheme - Large Cap Fund",
    "Equity Scheme - Large & Mid Cap Fund",
    "Equity Scheme - Mid Cap Fund",
    "Equity Scheme - Small Cap Fund",
    "Equity Scheme - Multi Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - ELSS",
    "Equity Scheme - Sectoral/ Thematic",
    "Equity Scheme - Focused Fund",
    "Equity Scheme - Value Fund",
    "Equity Scheme - Contra Fund",
    "Hybrid Scheme - Aggressive Hybrid Fund",
    "Other Scheme - FoF Overseas",
    "Debt Scheme - Credit Risk Fund",
    "Other Scheme - Gold ETF",
    "Other Scheme - Other  ETFs",
]


def map_risk(cat: str):
    if pd.isna(cat):
        return np.nan
    if cat in low_cats:
        return "Low"
    if cat in medium_cats:
        return "Medium"
    if cat in high_cats:
        return "High"
    return np.nan


# -------------------- plotting helpers --------------------

def ensure_figures_dir() -> Path:
    figures_dir = Path("reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_safe_cv(y: pd.Series, max_splits: int = 3) -> StratifiedKFold:
    """
    learning_curve/validation_curve require each class to have at least n_splits samples.
    Use a safe n_splits based on the smallest class count (and cap at max_splits).
    """
    counts = y.value_counts()
    min_class_count = int(counts.min())

    if min_class_count < 2:
        raise ValueError(
            f"Not enough samples in the smallest class for CV.\n"
            f"Class counts:\n{counts}\n"
            f"Smallest class count = {min_class_count} (need >= 2)."
        )

    n_splits = min(max_splits, min_class_count)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def plot_learning_curve(model, X, y, out_path: Path) -> None:
    """
    Learning curve: training vs validation score as training size increases.
    Use macro-F1 for imbalanced multiclass.
    FAST SETTINGS: 3 train sizes, <=3 CV folds, n_jobs=1 (Windows-safe).
    """
    cv = get_safe_cv(y, max_splits=3)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 3),
        n_jobs=1,  # Windows-safe
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training (F1-macro)")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation (F1-macro)")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1-macro")
    plt.title("Learning Curve – Risk Model (Macro-F1)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_validation_curve(model, X, y, out_path: Path) -> None:
    """
    RandomForest has no true loss curve. Validation curve is a good loss-like diagnostic.
    Use macro-F1 for imbalanced multiclass.
    FAST SETTINGS: 3 parameter points, <=3 CV folds, n_jobs=1 (Windows-safe).
    """
    cv = get_safe_cv(y, max_splits=3)
    param_range = [50, 100, 200]

    train_scores, val_scores = validation_curve(
        model,
        X,
        y,
        param_name="model__n_estimators",
        param_range=param_range,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,  # Windows-safe
    )

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_scores.mean(axis=1), marker="o", label="Training (F1-macro)")
    plt.plot(param_range, val_scores.mean(axis=1), marker="o", label="Validation (F1-macro)")
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("F1-macro")
    plt.title("Validation Curve – Risk Model (Macro-F1)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# -------------------- risk model training --------------------

def train_risk_model() -> Path:
    """Train the RandomForest risk model and save to models/risk_model.joblib."""
    print("\n[Risk] Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Map Scheme_Category -> Risk_Level
    df["Risk_Level"] = df["Scheme_Category"].apply(map_risk)

    # Drop rows where risk is unknown
    df_model = df.dropna(subset=["Risk_Level"]).copy()
    print(f"[Risk] Rows after risk mapping: {df_model.shape[0]}")

    # Scheme_Min_Amt -> numeric
    df_model["Scheme_Min_Amt_num"] = pd.to_numeric(
        df_model["Scheme_Min_Amt"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    # Dates
    for col in ["Latest_NAV_Date", "Launch_Date"]:
        df_model[col] = pd.to_datetime(df_model[col], errors="coerce")

    # Fund age in years
    df_model["Fund_Age_Years"] = (
        (df_model["Latest_NAV_Date"] - df_model["Launch_Date"]).dt.days
    ) / 365.25

    # Features and target
    target = "Risk_Level"

    numeric_features = ["NAV", "Average_AUM_Cr", "Scheme_Min_Amt_num", "Fund_Age_Years"]

    # IMPORTANT: drop Scheme_NAV_Name to reduce leakage/high-cardinality memorization
    categorical_features = ["AMC", "Scheme_Type", "AAUM_Quarter"]

    X = df_model[numeric_features + categorical_features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[Risk] y_train class counts:\n", y_train.value_counts())

    # Pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Final model (regularized to reduce overfitting)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,  # ok here (normal fit), curves still use n_jobs=1
                ),
            ),
        ]
    )

    print("\n[Risk] Fitting final RandomForest risk model (regularized)...")
    model.fit(X_train, y_train)

    # Evaluation (console only)
    y_pred = model.predict(X_test)
    print("\n[Risk] Classification report (Risk_Level):")
    print(classification_report(y_test, y_pred))

    # -------------------- curves (FAST) --------------------
    MAX_CURVE_SAMPLES = 3000  # adjust (1000–5000) based on your laptop

    if len(X_train) > MAX_CURVE_SAMPLES:
        X_curve, _, y_curve, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_CURVE_SAMPLES,
            random_state=42,
            stratify=y_train,
        )
        print(f"\n[Risk] Using stratified subset for curves: {len(X_curve)} rows")
    else:
        X_curve, y_curve = X_train, y_train
        print(f"\n[Risk] Using full training data for curves: {len(X_curve)} rows")

    # Lighter clone ONLY for curves (faster)
    model_curve = clone(model)
    model_curve.set_params(model__n_estimators=100)

    figs = ensure_figures_dir()

    print("[Risk] Generating learning curve plot (fast, macro-F1)...")
    plot_learning_curve(model_curve, X_curve, y_curve, figs / "learning_curve_risk_model.png")
    print(f"[Risk] Saved: {figs / 'learning_curve_risk_model.png'}")

    print("[Risk] Generating validation curve plot (fast, macro-F1)...")
    plot_validation_curve(model_curve, X_curve, y_curve, figs / "validation_curve_risk_model.png")
    print(f"[Risk] Saved: {figs / 'validation_curve_risk_model.png'}")

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "risk_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[Risk] Saved risk model to {model_path}")

    return model_path


# -------------------- master pipeline entrypoint --------------------

def main() -> None:
    divider("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda()

    divider("STEP 2: MULTI-MODEL TRAINING & COMPARISON")
    run_all_models()

    divider("STEP 3: TRAIN FINAL RISK MODEL (Low / Medium / High)")
    train_risk_model()

    divider("TRAINING PIPELINE COMPLETED")
    print("✓ EDA finished (plots in reports/figures/)")
    print("✓ Scheme-Type models trained (reports/metrics, best_model.joblib)")
    print("✓ Risk model trained and saved as models/risk_model.joblib")
    print("✓ Learning + validation curves saved in reports/figures/\n")


if __name__ == "__main__":
    main()

