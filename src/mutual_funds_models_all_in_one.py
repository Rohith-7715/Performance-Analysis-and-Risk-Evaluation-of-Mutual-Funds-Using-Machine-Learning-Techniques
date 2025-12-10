from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from .config import (
    DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES_BASELINE,
    CATEGORICAL_FEATURES_OPT,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    BEST_MODEL_PATH,
    BEST_MODEL_METRICS_PATH,
)
from .utils import divider


# -------------------------------------------------------------------
# Data loading + preprocessing (inlined from data_prep.py)
# -------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    required_cols = list(
        set(
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES_BASELINE
            + CATEGORICAL_FEATURES_OPT
            + [TARGET_COL]
        )
    )

    df_clean = df[required_cols].dropna()

    noisy_df = df_clean.copy()
    for col in NUMERIC_FEATURES:
        if pd.api.types.is_numeric_dtype(noisy_df[col]):
            noise = 0.30 * np.random.randn(len(noisy_df))
            noisy_df[col] = noisy_df[col] * (1 + noise)

    return noisy_df


def build_preprocessors():
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    cat_baseline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    cat_opt = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[save_json] Saved -> {path}")


def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved confusion matrix -> {filename}")


def plot_model_comparison(models_metrics: dict, filename: Path, title: str) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(models_metrics.keys())
    acc_scores = [m["accuracy"] * 100 for m in models_metrics.values()]
    f1_scores = [m["macro_f1"] * 100 for m in models_metrics.values()]
    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, acc_scores, width, label="Accuracy")
    bars2 = plt.bar(x + width / 2, f1_scores, width, label="Macro F1")

    plt.ylabel("Score (%)")
    plt.ylim(0, 110)
    plt.title(title)
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        plt.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved model comparison -> {filename}")


def plot_before_after_single(
    models_before: dict,
    models_after: dict,
    filename: Path,
    metric_key: str = "macro_f1",
    metric_label: str = "Macro F1 (%)",
    title: str = "Macro F1: Before vs After Tuning (All Models)",
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(models_before.keys())

    print("\n[DEBUG] Before/after combined plot")
    print("Models (before):", list(models_before.keys()))
    print("Models (after): ", list(models_after.keys()))
    print("Saving to      :", filename)

    valid_models = []
    before_vals = []
    after_vals = []

    for m in model_names:
        if (
            m in models_after
            and metric_key in models_before[m]
            and metric_key in models_after[m]
        ):
            valid_models.append(m)
            before_vals.append(models_before[m][metric_key] * 100)
            after_vals.append(models_after[m][metric_key] * 100)

    if len(valid_models) == 0:
        print("[plot] No valid models found for before/after comparison – skipping plot.")
        return

    x = np.arange(len(valid_models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars_before = plt.bar(x - width / 2, before_vals, width, label="Before tuning")
    bars_after = plt.bar(x + width / 2, after_vals, width, label="After tuning")

    plt.ylabel(metric_label)
    plt.ylim(0, 110)
    plt.title(title)
    plt.xticks(x, valid_models, rotation=15, ha="right")
    plt.legend()

    for bar in list(bars_before) + list(bars_after):
        h = bar.get_height()
        plt.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved before/after comparison -> {filename}\n")


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    if label_encoder:
        target_names = label_encoder.classes_
        cls_report = classification_report(y_test, y_pred, target_names=target_names)
        labels_for_cm = np.arange(len(target_names))
        tick_labels = list(target_names)
    else:
        cls_report = classification_report(y_test, y_pred)
        labels_for_cm = None
        tick_labels = sorted(set(y_test))

    cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm)

    print(f"=== {name} PERFORMANCE ===")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print(cls_report)
    print(cm)

    report_file = METRICS_DIR / f"{name.lower()}_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(cls_report)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classes": [str(l) for l in tick_labels],
    }

    metrics_file = METRICS_DIR / f"{name.lower()}_metrics.json"
    save_json(metrics, metrics_file)

    return metrics


# -------------------------------------------------------------------
# Model training functions
# -------------------------------------------------------------------
def train_logreg_baseline(preprocessor, X_train, y_train) -> Pipeline:
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs", multi_class="auto")
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)
    return model


def train_rf_baseline(preprocessor, X_train, y_train) -> Pipeline:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)
    return model


def train_xgb_baseline(preprocessor, X_train, y_train) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)
    return model


def train_logreg_tuned(preprocessor, X_train, y_train) -> Pipeline:
    base_clf = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", base_clf)])
    param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__penalty": ["l1", "l2"]}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)
    results_file = METRICS_DIR / "logreg_tuned_cv_results.json"
    save_json({"best_params": grid.best_params_, "best_score_macro_f1": float(grid.best_score_)}, results_file)
    return grid.best_estimator_


def train_rf_tuned(preprocessor, X_train, y_train) -> Pipeline:
    from sklearn.ensemble import RandomForestClassifier

    base_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", base_clf)])
    param_grid = {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)
    results_file = METRICS_DIR / "rf_tuned_cv_results.json"
    save_json({"best_params": grid.best_params_, "best_score_macro_f1": float(grid.best_score_)}, results_file)
    return grid.best_estimator_


def train_xgb_tuned(preprocessor, X_train, y_train) -> Pipeline:
    base_clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", base_clf)])
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)
    results_file = METRICS_DIR / "xgb_tuned_cv_results.json"
    save_json({"best_params": grid.best_params_, "best_score_macro_f1": float(grid.best_score_)}, results_file)
    return grid.best_estimator_


# -------------------------------------------------------------------
# Best-model analysis
# -------------------------------------------------------------------
def analyse_best_model(best_name: str, best_model: Pipeline) -> None:
    preprocessor = best_model.named_steps["preprocessor"]
    clf = best_model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(clf, "coef_"):
        coef = clf.coef_
        importance = np.mean(np.abs(coef), axis=0)
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        raise ValueError("Best model has no coefficients or feature_importances_ attribute")

    def prettify_feature(name: str) -> str:
        if "__" in name:
            _, name = name.split("__", 1)

        if name.startswith("Scheme_Min_Amt_"):
            label = name.replace("Scheme_Min_Amt_", "")
            label = label.replace("_", " ")
            label = f"Min Investment: {label}"
        elif name.startswith("AMC_"):
            label = "AMC: " + name.replace("AMC_", "").replace("_", " ")
        elif name.startswith("Scheme_Type_"):
            label = "Scheme Type: " + name.replace("Scheme_Type_", "").replace("_", " ")
        elif name.startswith("Scheme_NAV_Name_"):
            label = "NAV Option: " + name.replace("Scheme_NAV_Name_", "").replace("_", " ")
        elif name.startswith("AAUM_Quarter_"):
            label = "AAUM Quarter: " + name.replace("AAUM_Quarter_", "").replace("_", " ")
        else:
            label = name.replace("_", " ")

        max_len = 60
        if len(label) > max_len:
            label = label[: max_len - 3] + "..."
        return label

    pretty_names = [prettify_feature(n) for n in feature_names]

    fi_df = (
        pd.DataFrame(
            {
                "feature_raw": feature_names,
                "feature": pretty_names,
                "importance": importance,
            }
        )
        .sort_values("importance", ascending=False)
    )

    csv_path = METRICS_DIR / f"{best_name.lower()}_feature_importances.csv"
    fi_df.to_csv(csv_path, index=False)
    print(f"[best_model] Saved feature importances -> {csv_path}")

    K = min(15, len(fi_df))
    top_df = fi_df.head(K).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, max(5, 0.45 * K)))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{best_name} - Top {K} Features (Bar Plot)")
    plt.tight_layout()
    bar_path = FIGURES_DIR / f"{best_name.lower()}_top_{K}_features.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[best_model] Saved top-feature bar plot -> {bar_path}")

    top_df = top_df.copy()
    if top_df["importance"].max() > 0:
        top_df["importance_norm"] = top_df["importance"] / top_df["importance"].max()
    else:
        top_df["importance_norm"] = top_df["importance"]

    heat_df = top_df.set_index("feature")[["importance_norm"]]

    plt.figure(figsize=(6, max(5, 0.45 * K)))
    ax = sns.heatmap(
        heat_df,
        cmap="Reds",
        cbar_kws={"label": "Normalised Importance"},
        xticklabels=["Importance"],
        yticklabels=True,
    )
    ax.tick_params(axis="y", labelsize=9)
    plt.title(f"{best_name} - Top {K} Features (Importance Heatmap)")
    plt.xlabel("")
    plt.ylabel("Feature")
    plt.tight_layout()
    heat_path = FIGURES_DIR / f"{best_name.lower()}_feature_heatmap_top{K}.png"
    plt.savefig(heat_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[best_model] Saved feature-importance heatmap -> {heat_path}")


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
def run_models() -> None:
    divider("LOAD DATA")
    df = load_data()

    divider("TRAIN / TEST SPLIT")
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    divider("BUILD PREPROCESSOR")
    _, preproc_opt = build_preprocessors()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    divider("BASELINE MODELS")
    baseline_metrics = {}
    baseline_models = {}

    lr_base = train_logreg_baseline(preproc_opt, X_train, y_train)
    baseline_models["LogReg"] = lr_base
    baseline_metrics["LogReg"] = evaluate_model("LogReg_Baseline", lr_base, X_test, y_test, label_encoder)

    rf_base = train_rf_baseline(preproc_opt, X_train, y_train)
    baseline_models["RandomForest"] = rf_base
    baseline_metrics["RandomForest"] = evaluate_model(
        "RandomForest_Baseline", rf_base, X_test, y_test, label_encoder
    )

    xgb_base = train_xgb_baseline(preproc_opt, X_train, y_train)
    baseline_models["XGBoost"] = xgb_base
    baseline_metrics["XGBoost"] = evaluate_model("XGBoost_Baseline", xgb_base, X_test, y_test, label_encoder)

    divider("TUNED MODELS (GRIDSEARCHCV)")
    tuned_metrics = {}
    tuned_models = {}

    lr_tuned = train_logreg_tuned(preproc_opt, X_train, y_train)
    tuned_models["LogReg"] = lr_tuned
    tuned_metrics["LogReg"] = evaluate_model("LogReg_Tuned", lr_tuned, X_test, y_test, label_encoder)

    rf_tuned = train_rf_tuned(preproc_opt, X_train, y_train)
    tuned_models["RandomForest"] = rf_tuned
    tuned_metrics["RandomForest"] = evaluate_model("RandomForest_Tuned", rf_tuned, X_test, y_test, label_encoder)

    xgb_tuned = train_xgb_tuned(preproc_opt, X_train, y_train)
    tuned_models["XGBoost"] = xgb_tuned
    tuned_metrics["XGBoost"] = evaluate_model("XGBoost_Tuned", xgb_tuned, X_test, y_test, label_encoder)

    divider("MODEL COMPARISONS")
    print("[DEBUG] FIGURES_DIR:", FIGURES_DIR.resolve())
    print("[DEBUG] Baseline metric keys:", baseline_metrics.keys())
    print("[DEBUG] Tuned metric keys   :", tuned_metrics.keys())

    plot_model_comparison(
        baseline_metrics,
        FIGURES_DIR / "model_comparison_baseline.png",
        "Baseline Models: Accuracy vs Macro F1",
    )

    plot_model_comparison(
        tuned_metrics,
        FIGURES_DIR / "model_comparison_tuned.png",
        "Tuned Models: Accuracy vs Macro F1",
    )

    plot_before_after_single(
        baseline_metrics,
        tuned_metrics,
        FIGURES_DIR / "before_after_macro_f1_all_models.png",
        metric_key="macro_f1",
        metric_label="Macro F1 (%)",
        title="Macro F1: Before vs After Tuning (All Models)",
    )

    save_json(baseline_metrics, METRICS_DIR / "all_models_baseline_summary.json")
    save_json(tuned_metrics, METRICS_DIR / "all_models_tuned_summary.json")

    divider("SELECT & SAVE BEST TUNED MODEL")
    best_name = None
    best_score = -1.0
    for name, m in tuned_metrics.items():
        if m["macro_f1"] > best_score:
            best_score = m["macro_f1"]
            best_name = name

    best_model = tuned_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)

    best_summary = {
        "best_model": best_name,
        "accuracy": tuned_metrics[best_name]["accuracy"],
        "macro_f1": tuned_metrics[best_name]["macro_f1"],
    }
    save_json(best_summary, BEST_MODEL_METRICS_PATH)

    divider("BEST MODEL FEATURE ANALYSIS")
    analyse_best_model(best_name + "_Tuned", best_model)

    divider("SUMMARY")
    print("Baseline models:")
    for name, m in baseline_metrics.items():
        print(f"{name:12s} | Acc: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f}")

    print("\nTuned models:")
    for name, m in tuned_metrics.items():
        print(f"{name:12s} | Acc: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f}")

    print(f"\nBest tuned model: {best_name} (Macro F1={best_score:.4f})")


if __name__ == "__main__":
    run_models()
