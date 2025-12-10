from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    categorical_features = ["AMC", "Scheme_Type", "Scheme_NAV_Name", "AAUM_Quarter"]

    X = df_model[numeric_features + categorical_features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\n[Risk] Fitting RandomForest risk model...")
    clf.fit(X_train, y_train)

    # Evaluation (console only)
    y_pred = clf.predict(X_test)
    print("\n[Risk] Classification report (Risk_Level):")
    print(classification_report(y_test, y_pred))

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "risk_model.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[Risk] Saved best risk model to {model_path}")

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
    print("✓ Risk model trained and saved as models/risk_model.joblib\n")


if __name__ == "__main__":
    main()

