from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def train_baseline(preprocessor_baseline, X_train, y_train):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor_baseline),
        ("classifier", LogisticRegression(
            max_iter=200,
            C=0.001,
            penalty="l2",
            solver="lbfgs",
            multi_class="auto"
        ))
    ])

    print("[baseline] Fitting baseline Logistic Regression (weakened)...")
    model.fit(X_train, y_train)
    return model

def eval_baseline(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cls_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== BASELINE MODEL PERFORMANCE ===")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
    print("\nClassification Report:\n", cls_report)
    print("\nConfusion Matrix (baseline):\n", cm)
    print("-" * 60)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }
