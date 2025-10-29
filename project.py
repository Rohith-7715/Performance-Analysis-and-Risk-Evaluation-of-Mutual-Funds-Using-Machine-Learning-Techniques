import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)

###############################################################################
# CONFIG SECTION - CHANGE THIS IF YOUR COLUMN NAMES ARE DIFFERENT
###############################################################################

# CSV from Kaggle
DATA_PATH = "mutual-fund-data.csv"

# Target column for classification
TARGET_COL = "risk_level"

# Numeric feature columns in your dataset
NUMERIC_FEATURES = [
    "expense_ratio",        # annual fee %
    "aum",                  # assets under management
    "beta",                 # market sensitivity
    "stdev_return",         # volatility
    "one_year_return",      # trailing 1Y return %
    "three_year_return",    # trailing 3Y return %
]

# Categorical feature columns
CATEGORICAL_FEATURES = [
    "fund_category",        # e.g. "Large Cap", "Mid Cap", etc.
]

# Output plot filename
PLOT_FILENAME = "model_comparison.png"


###############################################################################
# 1. LOAD DATA
###############################################################################

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
df_model = df[required_cols].dropna()
print("After dropping NA:", df_model.shape)

X = df_model[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df_model[TARGET_COL]

print("\nClass distribution:")
print(y.value_counts())
print("-" * 60)


###############################################################################
# 2. TRAIN / TEST SPLIT
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("-" * 60)


###############################################################################
# 3. PREPROCESSING PIPELINE
###############################################################################

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)


###############################################################################
# 4. BASELINE MODEL (UNOPTIMIZED LOGISTIC REGRESSION)
###############################################################################

baseline_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, multi_class="auto"))
])

print("Fitting baseline Logistic Regression...")
baseline_clf.fit(X_train, y_train)

y_pred_baseline = baseline_clf.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average="macro")

print("\n=== BASELINE MODEL PERFORMANCE ===")
print("Accuracy:", round(baseline_accuracy, 4))
print("Macro F1:", round(baseline_f1, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_baseline))

cm_base = confusion_matrix(y_test, y_pred_baseline)
print("Confusion Matrix (baseline):\n", cm_base)
print("-" * 60)


###############################################################################
# 5. OPTIMISED MODEL (GRIDSEARCH ON LOGISTIC REGRESSION)
###############################################################################

print("Running hyperparameter tuning (GridSearchCV)...")

pipeline_for_search = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=2000, solver="liblinear", multi_class="auto"))
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"],
}

grid_search = GridSearchCV(
    estimator=pipeline_for_search,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best params found:", grid_search.best_params_)
print("Best CV macro F1:", round(grid_search.best_score_, 4))

best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

optimized_accuracy = accuracy_score(y_test, y_pred_best)
optimized_f1 = f1_score(y_test, y_pred_best, average="macro")

print("\n=== OPTIMISED MODEL PERFORMANCE ===")
print("Accuracy:", round(optimized_accuracy, 4))
print("Macro F1:", round(optimized_f1, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

cm_opt = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix (optimised):\n", cm_opt)
print("-" * 60)


###############################################################################
# 6. PLOT COMPARISON (BEFORE VS AFTER OPTIMISATION)
###############################################################################

print("Plotting comparison bar chart...")

metrics_names = ["Accuracy", "Macro F1"]
baseline_scores = [baseline_accuracy, baseline_f1]
optimized_scores = [optimized_accuracy, optimized_f1]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(8,5))
bars1 = plt.bar(x - width/2, baseline_scores, width, label="Before optimisation")
bars2 = plt.bar(x + width/2, optimized_scores, width, label="After optimisation")

plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Baseline vs Optimised Logistic Regression (Risk Classification)")
plt.xticks(x, metrics_names)
plt.legend()

for bar in bars1:
    h = bar.get_height()
    plt.annotate(f"{h:.3f}",
                 xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha="center", va="bottom")
for bar in bars2:
    h = bar.get_height()
    plt.annotate(f"{h:.3f}",
                 xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha="center", va="bottom")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.close()

print("Saved comparison chart to model_comparison.png")
print("-" * 60)


###############################################################################
# 7. OPTIONAL: FEATURE EXPLANATION
###############################################################################

print("Extracting feature importances from optimised Logistic Regression...")

best_preprocessor = best_clf.named_steps["preprocess"]
best_model = best_clf.named_steps["model"]

ohe = best_preprocessor.named_transformers_["cat"].named_steps["onehot"]
encoded_cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)

final_feature_names = np.concatenate([
    NUMERIC_FEATURES,
    encoded_cat_names
])

coefs = pd.DataFrame(
    best_model.coef_,
    columns=final_feature_names,
    index=best_model.classes_
)

print("\nLogistic Regression coefficients by class (rows = classes):")
print(coefs)

coefs.to_csv("logreg_coefficients_by_class.csv")
print("Saved coefficient table to logreg_coefficients_by_class.csv")

print("\nDONE.")
