import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import DATA_PATH, FIGURES_DIR, TARGET_COL

plt.style.use("ggplot")


def run_eda():
    print("================================ EDA =================================")

    # 1️⃣ Load data
    df = pd.read_csv(DATA_PATH)
    print(f"[EDA] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("-" * 70)

    # 2️⃣ Preview first few rows
    print("[EDA] Sample rows:")
    print(df.head())
    print("-" * 70)

    # 3️⃣ Missing values
    missing = df.isnull().sum()
    has_missing = missing[missing > 0].sort_values(ascending=False)
    if not has_missing.empty:
        print("[EDA] Columns with missing values:")
        print(has_missing)
    else:
        print("[EDA] No missing values.")
    print("-" * 70)

    # 4️⃣ Data types overview
    print("[EDA] Data types:")
    print(df.dtypes)
    print("-" * 70)

    # 5️⃣ Summary stats for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print("[EDA] Numeric summary stats:")
        print(numeric_df.describe().T)
    else:
        print("[EDA] No numeric columns detected for describe()")
    print("-" * 70)

    # 6️⃣ Distribution of the target / class imbalance
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

    # 7️⃣ Correlation heatmap of numeric features
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

    # 8️⃣ NAV vs AUM scatter, coloured by fund type
    if "NAV" in df.columns and "Average_AUM_Cr" in df.columns:
        # ensure numeric (dataset sometimes stores numbers as strings)
        nav_num = pd.to_numeric(df["NAV"], errors="coerce")
        aum_num = pd.to_numeric(df["Average_AUM_Cr"], errors="coerce")

        scatter_df = pd.DataFrame({
            "NAV": nav_num,
            "Average_AUM_Cr": aum_num,
            TARGET_COL: df[TARGET_COL] if TARGET_COL in df.columns else None
        }).dropna()

        if not scatter_df.empty:
            plt.figure(figsize=(8, 6))
            if TARGET_COL in scatter_df.columns:
                sns.scatterplot(
                    data=scatter_df,
                    x="NAV",
                    y="Average_AUM_Cr",
                    hue=TARGET_COL,
                    alpha=0.7
                )
            else:
                sns.scatterplot(
                    data=scatter_df,
                    x="NAV",
                    y="Average_AUM_Cr",
                    alpha=0.7
                )
            plt.title("NAV vs Average AUM")
            plt.tight_layout()
            out_path = FIGURES_DIR / "nav_vs_aum.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[EDA] Saved: {out_path}")
        else:
            print("[EDA] Skipping NAV vs AUM scatter (not enough clean numeric rows).")
    else:
        print("[EDA] Skipping NAV vs AUM scatter (columns not found).")
    print("-" * 70)

    # 9️⃣ Launch year distribution (if Launch_Date exists)
    if "Launch_Date" in df.columns:
        years = pd.to_datetime(df["Launch_Date"], errors="coerce").dt.year
        year_counts = years.value_counts().sort_index()

        if year_counts.shape[0] > 0:
            plt.figure(figsize=(10, 4))
            sns.histplot(years.dropna(), bins=20, kde=True, color="skyblue")
            plt.title("Distribution of Mutual Fund Launch Years")
            plt.xlabel("Launch Year")
            plt.ylabel("Number of Funds")
            plt.tight_layout()
            out_path = FIGURES_DIR / "launch_year_distribution.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[EDA] Saved: {out_path}")

            # also print summary of launch years
            print("[EDA] Launch year summary:")
            print(year_counts.head(10))
        else:
            print("[EDA] Cannot plot launch years (no valid dates).")
    else:
        print("[EDA] Skipping launch year distribution (Launch_Date not found).")
    print("-" * 70)

    # 🔟 Pairplot across important numeric features (robust version)
    # We'll try known interesting columns and keep only those that are numeric after conversion.
    candidate_cols = [c for c in ["NAV", "Average_AUM_Cr", "Scheme_Min_Amt"] if c in df.columns]

    # convert all candidate cols to numeric if possible
    pair_df = df.copy()
    for col in candidate_cols:
        pair_df[col] = pd.to_numeric(pair_df[col], errors="coerce")

    # keep only actually numeric ones
    numeric_for_pairplot = [
        col for col in candidate_cols
        if pd.api.types.is_numeric_dtype(pair_df[col])
    ]

    if len(numeric_for_pairplot) >= 2:
        # build a clean dataframe with only numeric cols + target
        cols_to_plot = numeric_for_pairplot.copy()
        if TARGET_COL in pair_df.columns:
            cols_to_plot.append(TARGET_COL)

        pairplot_df = pair_df[cols_to_plot].dropna()

        if pairplot_df.shape[0] > 0:
            # Seaborn pairplot
            sns.pairplot(
                pairplot_df,
                vars=numeric_for_pairplot,
                hue=TARGET_COL if TARGET_COL in pairplot_df.columns else None,
                corner=True,
                diag_kind="kde"
            )
            plt.suptitle("Pairplot of Key Numeric Features", y=1.02)
            out_path = FIGURES_DIR / "pairplot_numeric.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[EDA] Saved: {out_path}")
        else:
            print("[EDA] Skipping pairplot (not enough valid rows after cleaning).")
    else:
        print("[EDA] Skipping pairplot (fewer than 2 usable numeric columns).")
    print("-" * 70)

    # 1️⃣1️⃣ Top AUM funds: let's get the largest funds by AUM
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


# Allow running this file directly with: python -m src.eda
if __name__ == "__main__":
    run_eda()
