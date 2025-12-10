💹 Performance Analysis & Risk Evaluation of Mutual Funds Using Machine Learning
📘 Overview

This project explores how machine learning can help investors understand and compare mutual funds more effectively.
It focuses on two things:

1️⃣ Analysing Fund Performance Using Multiple ML Models
I built a full pipeline that trains and evaluates several models (Logistic Regression, Random Forest, and XGBoost), compares how they perform, and identifies which features influence the predictions the most.

2️⃣ Predicting Mutual Fund Risk Using a Deployed API
A separate model classifies mutual funds into Low, Medium, or High risk.
This model is deployed using FastAPI, and comes with a clean modern UI where users can input fund details and instantly get a risk score.
Together, these two parts demonstrate how machine learning can support real-world financial decision-making, from analysis to deployment.

🎯 What This Project Aims to Do

✔️ Understand mutual fund behaviour through data
✔️ Find which features (AUM, NAV, AMC, Scheme Type, etc.) matter most
✔️ Train & compare multiple machine learning models
✔️ Improve model performance through hyperparameter tuning
✔️ Build an end-to-end prediction system that anyone can use
✔️ Present results with clean visuals and meaningful insights

This makes it suitable not only for academic submissions but also for interviews and real-world applications.

🧠 How the Project Works
1️⃣ Data Preparation
The dataset is taken from Kaggle and includes:
Scheme details
AMC details
NAV history
AUM values
Minimum investment
Scheme category (used for risk mapping)
The pipeline:
Cleans and formats the data
Converts categorical features to machine-readable form
Repairs numeric columns (e.g., minimum investment with commas)
Calculates extra features such as Fund Age
Removes missing or inconsistent rows
This creates a structured dataset ready for both the performance models and the risk prediction model.
2️⃣ Exploratory Data Analysis (EDA)
Before modeling, the project generates clear visualizations that help explain the structure of mutual funds:
📊 Fund type distribution
🔥 Correlation heatmaps
📈 NAV vs AUM scatterplots
📉 AUM trends
🧮 Class balance
The visuals give an intuitive feel for the dataset and make the final model insights much easier to understand.

3️⃣ Machine Learning Models (Performance Pipeline)
Three models are trained:
Logistic Regression
Random Forest
XGBoost
Each is trained twice:
Baseline model (default settings)
Tuned model (optimized using GridSearchCV + Stratified K-Fold CV)
Each model is evaluated on:
Accuracy
Macro F1-score
Confusion matrix
Classification report
All results are saved into a structured reports/ directory.
The pipeline also produces:
📌 Baseline vs Tuned performance comparison
📌 Before/After visualization across all models
📌 Best model selection based on Macro F1

4️⃣ Best Model Feature Importance
Once the best model is chosen:
Its full feature importance table is saved
A bar chart of the top features is generated
A clean heatmap visualises the importance distribution
Feature names are automatically converted into more human-friendly versions like:
AMC_ICICI → AMC: ICICI
Scheme_Type_Open_Ended → Scheme Type: Open Ended
Scheme_Min_Amt_num → Minimum Investment
This makes the visuals extremely readable and presentation-ready.
🔐 Risk Prediction Model (Deployed API)
This part of the project is designed to feel real and practical.
It uses a Random Forest Classifier trained on engineered features such as:
NAV
Average AUM
Minimum investment
Fund age
Scheme type
AMC
NAV option
AAUM quarter
The model predicts whether a fund is:
🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

The entire system is deployed using FastAPI, with:
A modern HTML homepage
A user-friendly prediction form
A JSON API endpoint for programmatic use
A dedicated model-info page showing the model summary
This makes the project industry-ready and easy to demonstrate.

🧩 Project Structure (Human-Friendly Explanation)
MUTUAL-FUND-ML/
│
├── data/                 → Raw dataset
├── models/               → Saved ML models
├── reports/              → All figures & metrics
│   ├── figures/          → PNG charts
│   └── metrics/          → Accuracy, F1, confusion matrix, etc.
│
├── src/                  → Core logic
│   ├── mutual_funds.py   → Full training + evaluation pipeline
│   ├── config.py         → Central settings
│   ├── eda.py            → Visual analysis
│   ├── utils.py          → Helper utilities
│   └── pipeline.py       → Runs the whole pipeline
│
├── templates/            → Frontend UI for the API
├── train.py              → Run entire ML pipeline
├── train_risk_model.py   → Train the deployed risk model
├── serve.py              → FastAPI app
├── runner.py             → train/serve combined runner
└── README.md


Everything is modular, cleanly separated, and easy to maintain.
🧮 Technologies Used
Python
scikit-learn
XGBoost
matplotlib & seaborn
FastAPI + Jinja2
joblib

📊 Key Insights From the Project
Certain AMCs and Scheme Types strongly influence the risk level.
Features like AUM, minimum investment, and fund age play a major role.
Tuned models consistently outperform baseline versions.
XGBoost or Random Forest often becomes the best-performing model depending on dataset characteristics.
The deployed risk model performs well (~88% accuracy) and generalizes cleanly.

🚀 How to Run the Project
Install dependencies
pip install -r requirements.txt
Run EDA
python -m src.eda
Run the full model training pipeline
python train.py
Train the risk prediction model
python train_risk_model.py
Launch the FastAPI app
python runner.py serve


Then open:

👉 http://localhost:8000/
to access the web interface.

🧠 Future Improvements

Add time-series forecasting (predict future NAV)
Add risk metrics such as Sharpe Ratio
Deploy the API on cloud (AWS/GCP/Azure)
Build a full dashboard with Streamlit
