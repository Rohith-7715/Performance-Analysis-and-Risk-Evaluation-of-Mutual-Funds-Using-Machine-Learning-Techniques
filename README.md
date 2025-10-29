💹 Performance Analysis and Risk Evaluation of Mutual Funds Using Machine Learning Techniques
📘 Project Overview
This project aims to analyze and evaluate the performance and risk levels of mutual funds using machine learning techniques. By leveraging historical financial data, the project identifies the key factors influencing mutual fund performance and classifies funds based on their risk characteristics.
The analysis combines Exploratory Data Analysis (EDA), predictive modeling, and model optimization to create a robust pipeline that can classify mutual funds into risk categories such as Open Ended, Close Ended, and Interval Funds.
This work demonstrates how data-driven insights and machine learning algorithms can enhance financial decision-making, risk assessment, and portfolio management.

🎯 Objectives
To explore and visualize key financial indicators of mutual funds.
To identify the most influential factors impacting fund performance.
To develop and evaluate machine learning models for fund classification and risk analysis.
To optimize model parameters for improved prediction accuracy and generalization.
To present comparative insights between baseline and optimized models.

🧠 Methodology
1️⃣ Data Collection & Preparation
Dataset: Mutual Fund Data (Kaggle)
Loaded and cleaned raw financial data.
Handled missing values and inconsistent entries.
Converted numerical and categorical features into a machine-readable format.

2️⃣ Exploratory Data Analysis (EDA)
Performed detailed EDA to understand fund structure, distribution, and correlations.
Generated insightful visualizations including:
Fund type distribution (Open/Close Ended)
Correlation heatmap of NAV, AUM, and risk indicators
NAV vs AUM scatter plots
Launch year trends

3️⃣ Machine Learning Modeling
Implemented Logistic Regression as the base classification model to predict mutual fund types based on financial attributes.

4️⃣ Model Optimization
Used GridSearchCV for hyperparameter tuning to improve accuracy and generalization.
Compared baseline vs optimized model performance using accuracy and macro F1-score metrics.

5️⃣ Evaluation & Visualization
Evaluated model using Accuracy, Precision, Recall, and F1-Score.
Visualized comparison between baseline and optimized models using bar charts.
Saved all results and metrics in the reports/ directory for reference.

📊 Key Results
Baseline model achieved high accuracy but moderate class balance (F1-score).
After optimization, model performance improved significantly.
Detailed visualization highlights the improvement between pre- and post-optimization stages.

🧩 Project Structure
MUTUAL-FUND-ML/
│
├── data/                   # Dataset folder
│   └── mutual-fund-data.csv
│
├── reports/                # Visualizations and metrics
│   ├── figures/
│   └── metrics/
│
├── src/                    # Core logic and modules
│   ├── config.py           # Global configurations
│   ├── data_prep.py        # Data loading and preprocessing
│   ├── eda.py              # Exploratory Data Analysis
│   ├── model_baseline.py   # Baseline model
│   ├── model_optimized.py  # Optimized model (GridSearch)
│   ├── evaluate.py         # Evaluation metrics
│   ├── plot.py             # Visualization scripts
│   ├── utils.py            # Helper utilities
│   └── test.py             # Debugging script
│
├── run.py                  # Main execution script
├── project.py              # Experimentation file
├── requirements.txt         # Python dependencies
└── README.md

🧮 Tech Stack

Programming Language: Python 🐍
Libraries:
pandas, numpy – Data processing
matplotlib, seaborn – Visualization
scikit-learn – Machine Learning
reportlab – Report generation
Environment: Visual Studio Code / Jupyter

🧾 Insights from EDA

Most funds are Open Ended, indicating investor flexibility.
Strong correlation found between NAV and AUM, as expected in fund performance.
Steady rise in fund launches post-2010, suggesting growing investor participation.
Some categorical columns (like fund type and AMC) significantly impact fund classification.

🧾 Conclusion
This project demonstrates the power of machine learning in financial analytics.
By automating the process of analyzing mutual fund data, it enables:
More accurate performance evaluation
Better understanding of risk factors
Data-driven investment decisions
The framework is modular and extendable — it can be adapted to other financial domains like portfolio optimization, ETF analysis, or stock risk prediction.

🚀 How to Run
1️⃣ Clone the repository:
git clone https://github.com/Rohith-7715/Performance-Analysis-and-Risk-Evaluation-of-Mutual-Funds-Using-Machine-Learning-Techniques
cd mutual-fund-ml
2️⃣ Install dependencies:
pip install -r requirements.txt
3️⃣ Run the complete project pipeline:
python run.py
4️⃣ View results:
Figures: reports/figures/
Metrics: reports/metrics/

🧠 Future Enhancements
Integrate additional ML models (Random Forest, XGBoost).
Implement a web dashboard for interactive performance analysis.
Incorporate time-series forecasting of NAV trends.
Add risk scoring using volatility or Sharpe ratio.
