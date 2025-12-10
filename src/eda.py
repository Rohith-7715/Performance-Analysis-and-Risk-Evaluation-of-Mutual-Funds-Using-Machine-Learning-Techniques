import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DATA_PATH, FIGURES_DIR, TARGET_COL

plt.style.use("ggplot")


def run_eda():
    print("================================ EDA =================================")
    print(f"[EDA] DATA_PATH   : {DATA_PATH}")
    print(f"[EDA] FIGURES_DIR : {FIGURES_DIR}")

    # make sure the folder exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"[EDA] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("-" * 70)

    print("[EDA] Sample rows:")
    print(df.head())
    print("-" * 70)

    missing = df.isnull().sum()
    has_missing = missing[missing > 0].sort_values(ascending=False)
    if not has_missing.empty:
        print("[EDA] Columns with missing values:")
        print(has_missing)
    else:
        print("[EDA] No missing values.")
    print("-" * 70)

    print("[EDA] Data types:")
    print(df.dtypes)
    print("-" * 70)

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print("[EDA] Numeric summary stats:")
        print(numeric_df.describe().T)
    else:
        print("[EDA] No numeric columns detected for describe()")
    print("-" * 70)

    # 1) Target / class distribution
    if TARGET_COL in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=TARGET_COL, palette="Set2")
        plt.title("Distribution of Mutual Fund Types")
        plt.xlabel("Scheme Type")
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        out_path = FIGURES_DIR / "scheme_type_distribution.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[EDA] Saved: {out_path}")
        print("[EDA] Class counts:")
        print(df[TARGET_COL].value_counts())
        print("-" * 70)
    else:
        print(f"[EDA] Target column '{TARGET_COL}' not found, skipping class distribution plot.")
        print("-" * 70)

    # 2) Correlation heatmap of numeric features
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        out_path = FIGURES_DIR / "correlation_heatmap.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[EDA] Saved: {out_path}")
    else:
        print("[EDA] Skipping correlation heatmap (need at least 2 numeric columns).")
    print("-" * 70)

    # Just print top AUM schemes (no figure)
    if "Average_AUM_Cr" in df.columns and "Scheme_Name" in df.columns:
        df_copy = df.copy()
        df_copy["Average_AUM_Cr"] = pd.to_numeric(df_copy["Average_AUM_Cr"], errors="coerce")
        top_aum = (
            df_copy[["Scheme_Name", "AMC", "Scheme_Type", "Average_AUM_Cr"]]
            .dropna()
            .sort_values("Average_AUM_Cr", ascending=False)
            .head(10)
        )
        print("[EDA] Top 10 schemes by AUM (Cr):")
        print(top_aum.to_string(index=False))
    else:
        print("[EDA] Skipping top AUM table (required columns not found).")

    print("================================ EDA DONE =============================\n")


if __name__ == "__main__":
    run_eda()
