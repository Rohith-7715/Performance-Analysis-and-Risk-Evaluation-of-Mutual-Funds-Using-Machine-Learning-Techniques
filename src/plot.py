import numpy as np
import matplotlib.pyplot as plt
from .config import MODEL_COMPARISON_PLOT

def plot_comparison(baseline_metrics, optimized_metrics):
    metrics_names = ["Accuracy", "Macro F1"]
    baseline_scores = [
        baseline_metrics["accuracy"] * 100,
        baseline_metrics["macro_f1"] * 100
    ]
    optimized_scores = [
        optimized_metrics["accuracy"] * 100,
        optimized_metrics["macro_f1"] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, baseline_scores, width, label="Before optimisation")
    bars2 = plt.bar(x + width/2, optimized_scores, width, label="After optimisation")

    plt.ylabel("Score (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.title("Baseline vs Optimised Logistic Regression (Risk Classification)",
              pad=20, fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_names, fontsize=11)
    plt.legend(frameon=True, facecolor="white", edgecolor="black")

    # Annotate bars with percentage values
    for bar in bars1 + bars2:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[plot] Saved comparison plot -> {MODEL_COMPARISON_PLOT}")
