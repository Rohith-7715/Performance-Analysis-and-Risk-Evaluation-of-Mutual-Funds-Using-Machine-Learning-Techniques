import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import (
    DATA_PATH,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES_BASELINE,
    CATEGORICAL_FEATURES_OPT,
    TEST_SIZE,
    RANDOM_STATE,
)

def load_data():
    print(f"[data_prep] Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_cols = list(set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES_BASELINE
        + CATEGORICAL_FEATURES_OPT
        + [TARGET_COL]
    ))

    df_clean = df[required_cols].dropna()
    print(f"[data_prep] Raw shape: {df.shape}, after dropna: {df_clean.shape}")

    # inject strong numeric noise (±30%) to keep things challenging
    noisy_df = df_clean.copy()
    for col in NUMERIC_FEATURES:
        if pd.api.types.is_numeric_dtype(noisy_df[col]):
            noise = 0.30 * np.random.randn(len(noisy_df))  # ±30%
            noisy_df[col] = noisy_df[col] * (1 + noise)

    return noisy_df

def split_data(df):
    # same test set used for both models
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("[data_prep] Class distribution:")
    print(y.value_counts())
    print("-" * 60)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("[data_prep] Train_full size:", X_train_full.shape)
    print("[data_prep] Test size:", X_test.shape)

    # weaken baseline: cut training data
    drop_fraction = 0.40
    keep_idx = X_train_full.sample(
        frac=(1 - drop_fraction),
        random_state=RANDOM_STATE
    ).index
    X_train_weakened = X_train_full.loc[keep_idx]
    y_train_weakened = y_train_full.loc[keep_idx]

    print("[data_prep] Baseline weakened train size:", X_train_weakened.shape)
    print("-" * 60)

    return (
        X_train_weakened,
        y_train_weakened,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
    )

def build_preprocessors():
    """
    Returns two different ColumnTransformers:
    - preproc_baseline: uses full categorical feature set
    - preproc_opt: uses reduced categorical set (no Scheme_Category)
    """

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat_baseline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    cat_opt = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc_baseline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", cat_baseline, CATEGORICAL_FEATURES_BASELINE),
        ]
    )

    preproc_opt = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", cat_opt, CATEGORICAL_FEATURES_OPT),
        ]
    )

    return preproc_baseline, preproc_opt
