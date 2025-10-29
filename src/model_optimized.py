from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .config import CV_FOLDS

def train_optimized(preprocessor_opt, X_train_full, y_train_full):
    print("[optimized] Running GridSearchCV on Logistic Regression (reduced features)...")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor_opt),
        ("classifier", LogisticRegression(
            max_iter=500,
            multi_class="auto"
        ))
    ])

    # restricted search space (already nerfed)
    param_grid = {
        "classifier__C": [0.05, 0.1, 0.5, 1],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=CV_FOLDS,
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train_full, y_train_full)

    print("[optimized] Best params:", grid.best_params_)
    print("[optimized] Best CV macro F1:", round(grid.best_score_, 4))

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid.best_score_

def eval_optimized(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cls_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== OPTIMISED MODEL PERFORMANCE ===")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
    print("\nClassification Report:\n", cls_report)
    print("\nConfusion Matrix (optimised):\n", cm)
    print("-" * 60)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }
