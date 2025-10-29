# Mutual Fund ML - Risk Classification with Logistic Regression

## What this project does
- Loads mutual fund data.
- Trains a baseline Logistic Regression model.
- Tunes it with GridSearchCV.
- Compares BEFORE vs AFTER optimisation.
- Saves:
  - metrics to `reports/metrics/`
  - comparison plot to `reports/figures/model_comparison.png`
  - model coefficients to `reports/metrics/logreg_coefficients_by_class.csv`

## How to run
1. Create virtual env (optional but recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
