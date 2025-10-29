import json
import pandas as pd
import numpy as np
from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES_OPT,
    BASELINE_METRICS_JSON,
    OPTIMIZED_METRICS_JSON,
    COEFFICIENTS_CSV,
)

def save_metrics(metrics_dict, path):
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[evaluate] Saved metrics -> {path}")

def extract_and_save_coefficients(best_model):
    """
    Pull coefficients from the trained logistic regression model
    and write them to CSV for interpretation.

    We use the optimised model pipeline. That pipeline has:
    - step 'preprocessor'   (ColumnTransformer)
    - step 'classifier'     (LogisticRegression)

    We also build feature names using:
    - numeric features (NUMERIC_FEATURES)
    - one-hot encoded categorical features (CATEGORICAL_FEATURES_OPT)
    """
    try:
        preprocessor = best_model.named_steps["preprocessor"]
        logreg = best_model.named_steps["classifier"]

        # grab the one-hot encoder from the 'cat' transformer
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES_OPT)

        final_feature_names = np.concatenate([
            NUMERIC_FEATURES,
            encoded_cat_names
        ])

        coefs = pd.DataFrame(
            logreg.coef_,
            columns=final_feature_names,
            index=logreg.classes_
        )

        coefs.to_csv(COEFFICIENTS_CSV, index=True)
        print(f"[evaluate] Saved coefficients -> {COEFFICIENTS_CSV}")

    except Exception as e:
        # Not fatal if we can't extract (e.g. shapes changed, categories removed)
        print("[evaluate] Could not extract/save coefficients:", e)
