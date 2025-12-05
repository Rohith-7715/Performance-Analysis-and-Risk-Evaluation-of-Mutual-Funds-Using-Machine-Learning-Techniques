from pathlib import Path
from typing import Any, Dict
import json

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import (
    BEST_MODEL_PATH,
    BEST_MODEL_METRICS_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES_OPT,
)


# =========================
#  FastAPI model schema
# =========================

class FundFeatures(BaseModel):
    """Input schema for prediction requests."""
    NAV: float
    Average_AUM_Cr: float
    Scheme_Min_Amt: float
    AAUM_Quarter: str


# =========================
#  FastAPI app definition
# =========================

app = FastAPI(title="Mutual Fund Risk Model API")

MODEL = None           # will be loaded on startup
BEST_MODEL_INFO: Dict[str, Any] = {}   # summary for homepage


@app.on_event("startup")
def load_model() -> None:
    """
    Load the best tuned model and summary from disk when the API starts.
    """
    global MODEL, BEST_MODEL_INFO

    model_path = Path(BEST_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            "Run `python train.py` first to train and save the model."
        )
    MODEL = joblib.load(model_path)
    print(f"[deploy] Loaded best model from: {model_path}")

    metrics_path = Path(BEST_MODEL_METRICS_PATH)
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            BEST_MODEL_INFO = json.load(f)
    else:
        BEST_MODEL_INFO = {
            "best_model": "Unknown",
            "accuracy": "N/A",
            "macro_f1": "N/A",
        }
    print(f"[deploy] Best-model summary: {BEST_MODEL_INFO}")


# ---------- HOMEPAGE "/" ----------

@app.get("/", response_class=HTMLResponse)
def homepage() -> str:
    """Homepage with brief project info and link to prediction UI."""
    model_name = BEST_MODEL_INFO.get("best_model", "Unknown")
    acc = BEST_MODEL_INFO.get("accuracy", "N/A")
    f1 = BEST_MODEL_INFO.get("macro_f1", "N/A")

    # Safely format accuracy / f1 as strings
    if isinstance(acc, (float, int)):
        acc_str = f"{acc:.4f}"
    else:
        acc_str = str(acc)

    if isinstance(f1, (float, int)):
        f1_str = f"{f1:.4f}"
    else:
        f1_str = str(f1)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Mutual Fund Risk Classifier</title>
        <style>
            body {{
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                padding: 3rem 1.5rem 4rem;
            }}
            .card {{
                background: #020617;
                border-radius: 18px;
                padding: 2rem 2.2rem;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.75);
                border: 1px solid #1f2937;
            }}
            h1 {{
                font-size: 2rem;
                margin-bottom: 0.75rem;
                color: #f9fafb;
            }}
            p {{
                font-size: 0.95rem;
                line-height: 1.6;
                color: #9ca3af;
            }}
            ul {{
                font-size: 0.95rem;
                color: #e5e7eb;
            }}
            .btn {{
                display: inline-block;
                padding: 0.7rem 1.4rem;
                border-radius: 999px;
                background: #4f46e5;
                color: #fff;
                text-decoration: none;
                font-size: 0.95rem;
                font-weight: 500;
                margin-top: 1.5rem;
            }}
            .btn:hover {{
                background: #6366f1;
            }}
            .small {{
                font-size: 0.8rem;
                color: #9ca3af;
                margin-top: 1.5rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Mutual Fund Risk Classifier</h1>
                <p>
                    This application uses machine learning to analyse mutual fund characteristics
                    (NAV, AUM, minimum investment, quarter-wise AUM) and predict the
                    <code>Scheme_Type</code> &mdash; such as Open Ended, Close Ended, or Interval Fund.
                </p>

                <h2 style="font-size:1.15rem;margin-top:1.5rem;">Best Model (Training Summary)</h2>
                <ul>
                    <li><strong>Model:</strong> {model_name}</li>
                    <li><strong>Test Accuracy:</strong> {acc_str}</li>
                    <li><strong>Test Macro F1:</strong> {f1_str}</li>
                </ul>

                <a href="/predict-ui" class="btn">Go to Prediction Page →</a>

                <p class="small">
                    For API-based access, you can also call the JSON endpoint at <code>/predict</code>,
                    or view documentation at <code>/docs</code>.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


# ---------- PREDICTION UI "/predict-ui" ----------

@app.get("/predict-ui", response_class=HTMLResponse)
def predict_ui() -> str:
    """HTML prediction form."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Mutual Fund Risk Classifier – Predict</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 700px;
                margin: 0 auto;
                padding: 3rem 1.5rem 4rem;
            }
            .card {
                background: #020617;
                border-radius: 18px;
                padding: 2rem 2.2rem;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.75);
                border: 1px solid #1f2937;
            }
            h1 {
                font-size: 1.9rem;
                margin-bottom: 0.5rem;
                color: #f9fafb;
            }
            p {
                font-size: 0.93rem;
                line-height: 1.6;
                color: #9ca3af;
            }
            label {
                display: block;
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
                color: #d1d5db;
            }
            input, select {
                width: 100%;
                padding: 0.5rem 0.6rem;
                border-radius: 10px;
                border: 1px solid #374151;
                background: #020617;
                color: #e5e7eb;
                font-size: 0.9rem;
                margin-bottom: 0.9rem;
            }
            button {
                margin-top: 0.5rem;
                padding: 0.6rem 1.3rem;
                border-radius: 999px;
                border: 1px solid transparent;
                background: #4f46e5;
                color: white;
                font-size: 0.9rem;
                font-weight: 500;
                cursor: pointer;
            }
            button:hover {
                background: #6366f1;
            }
            pre {
                background: #020617;
                border-radius: 10px;
                padding: 0.8rem 1rem;
                font-size: 0.8rem;
                border: 1px solid #111827;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .small {
                font-size: 0.8rem;
                color: #9ca3af;
                margin-top: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Mutual Fund Risk Classifier</h1>
                <p>
                    Enter the feature values below and click <strong>Predict</strong>.
                    The model will return the predicted <code>Scheme_Type</code> and the
                    probability for each class.
                </p>

                <form id="predict-form">
                    <label for="nav">NAV</label>
                    <input type="number" step="0.01" id="nav" name="nav" value="25.3" required />

                    <label for="aum">Average AUM (Cr)</label>
                    <input type="number" step="0.01" id="aum" name="aum" value="1500" required />

                    <label for="min_amt">Scheme Minimum Amount</label>
                    <input type="number" step="1" id="min_amt" name="min_amt" value="5000" required />

                    <label for="quarter">AAUM Quarter</label>
                    <select id="quarter" name="quarter">
                        <option value="Q1">Q1</option>
                        <option value="Q2">Q2</option>
                        <option value="Q3">Q3</option>
                        <option value="Q4">Q4</option>
                    </select>

                    <button type="submit">Predict</button>
                </form>

                <h2 style="margin-top:1.5rem;font-size:1rem;">Response</h2>
                <pre id="response-box">{ "message": "Submit the form to see prediction" }</pre>
                <p class="small">
                    API docs are available at <code>/docs</code> and <code>/redoc</code> if needed.
                </p>
            </div>
        </div>

        <script>
            const form = document.getElementById("predict-form");
            const responseBox = document.getElementById("response-box");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();

                const payload = {
                    NAV: parseFloat(document.getElementById("nav").value),
                    Average_AUM_Cr: parseFloat(document.getElementById("aum").value),
                    Scheme_Min_Amt: parseFloat(document.getElementById("min_amt").value),
                    AAUM_Quarter: document.getElementById("quarter").value
                };

                responseBox.textContent = "Sending request...";

                try {
                    const res = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });

                    const text = await res.text();
                    let display;

                    try {
                        const json = JSON.parse(text);
                        display = JSON.stringify(json, null, 2);
                    } catch (parseErr) {
                        display = `HTTP ${res.status} response (not JSON):\\n` + text;
                    }

                    responseBox.textContent = display;
                } catch (err) {
                    responseBox.textContent = "Network error: " + err;
                }
            });
        </script>
    </body>
    </html>
    """


# ---------- PREDICTION API "/predict" ----------

@app.post("/predict")
def predict(features: FundFeatures) -> Dict[str, Any]:
    """
    Predict the Scheme_Type for a given mutual fund, and return class probabilities.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Run training first and check startup logs.",
        )

    try:
        data = pd.DataFrame([{
            "NAV": features.NAV,
            "Average_AUM_Cr": features.Average_AUM_Cr,
            "Scheme_Min_Amt": features.Scheme_Min_Amt,
            "AAUM_Quarter": features.AAUM_Quarter,
        }])

        # ensure correct dtypes
        for col in NUMERIC_FEATURES:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        for col in CATEGORICAL_FEATURES_OPT:
            if col in data.columns:
                data[col] = data[col].astype(str)

        ordered_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES_OPT)
        data = data[ordered_cols]

        preds = MODEL.predict(data)

        if not hasattr(MODEL, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba().")

        proba = MODEL.predict_proba(data)[0]
        classes = getattr(MODEL, "classes_", None)
        if classes is None:
            raise ValueError("Loaded model has no 'classes_' attribute.")

        probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}

        return {
            "prediction": str(preds[0]),
            "probabilities": probabilities,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
