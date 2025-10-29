from src.data_prep import load_data, split_data, build_preprocessors
from src.model_baseline import train_baseline, eval_baseline
from src.model_optimized import train_optimized, eval_optimized
from src.evaluate import save_metrics, extract_and_save_coefficients
from src.plot import plot_comparison
from src.utils import divider
from src.config import BASELINE_METRICS_JSON, OPTIMIZED_METRICS_JSON
from src.eda import run_eda


def main():
    run_eda()
    divider("LOAD DATA")
    df = load_data()

    divider("SPLIT DATA")
    (
        X_train_weakened,   # for baseline
        y_train_weakened,
        X_train_full,       # for optimised
        y_train_full,
        X_test,
        y_test,
    ) = split_data(df)

    divider("BUILD PREPROCESSORS")
    preproc_baseline, preproc_opt = build_preprocessors()

    divider("BASELINE MODEL (WEAKENED)")
    baseline_model = train_baseline(preproc_baseline, X_train_weakened, y_train_weakened)
    baseline_metrics = eval_baseline(baseline_model, X_test, y_test)
    save_metrics(baseline_metrics, BASELINE_METRICS_JSON)

    divider("OPTIMIZED MODEL (GRID SEARCH, REDUCED FEATURES)")
    optimized_model, best_params, best_cv_score = train_optimized(
        preproc_opt,
        X_train_full,
        y_train_full
    )
    optimized_metrics = eval_optimized(optimized_model, X_test, y_test)
    save_metrics(optimized_metrics, OPTIMIZED_METRICS_JSON)

    divider("PLOT COMPARISON")
    plot_comparison(baseline_metrics, optimized_metrics)

    divider("EXTRACT COEFFICIENTS / FEATURE IMPORTANCE")
    extract_and_save_coefficients(optimized_model)

    divider("DONE")
    print("Best hyperparameters:", best_params)
    print("Best cross-val macro F1:", round(best_cv_score, 4))

if __name__ == "__main__":
    main()
