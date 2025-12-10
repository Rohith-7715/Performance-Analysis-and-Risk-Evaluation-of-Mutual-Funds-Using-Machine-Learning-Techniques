from typing import Dict
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Mutual Fund Risk Prediction Service")

# CORS (optional, handy if you later call from JS frontend / notebooks)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory of the project (folder where serve.py lives)
BASE_DIR = Path(__file__).resolve().parent

# Templates (HTML files)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Static files (CSS, images, JS)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

# -----------------------------------------------------------------------------
# Paths and data loading
# -----------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"
DATA_PATH = BASE_DIR / "data" / "mutual-fund-data.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Train the model first with train_risk_model.py."
    )

# Load trained model
risk_model = joblib.load(MODEL_PATH)

# Load metadata for dropdowns (AMC, AAUM quarter) – read once at startup
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found at {DATA_PATH}. "
        "Make sure mutual-fund-data.csv is in the data/ folder."
    )

df_meta = pd.read_csv(DATA_PATH)

AMC_OPTIONS = sorted(df_meta["AMC"].dropna().unique().tolist())
AAUM_OPTIONS = sorted(df_meta["AAUM_Quarter"].dropna().unique().tolist())

# Features used by the model (must match train_risk_model.py)
MODEL_FEATURES = [
    "NAV",
    "Average_AUM_Cr",
    "Scheme_Min_Amt_num",
    "Fund_Age_Years",
    "AMC",
    "Scheme_Type",
    "Scheme_NAV_Name",
    "AAUM_Quarter",
]

# Basic model info (update metrics if you re-train)
MODEL_INFO = {
    "algorithm": "RandomForestClassifier",
    "task": "Mutual Fund Risk Classification (Low / Medium / High)",
    "features": MODEL_FEATURES,
    "metrics": {
        "accuracy": 0.88,
        "f1_low": 0.83,
        "f1_medium": 0.92,
        "f1_high": 0.78,
    },
}

# -----------------------------------------------------------------------------
# Pydantic models for JSON API
# -----------------------------------------------------------------------------
class RiskInput(BaseModel):
    NAV: float
    Average_AUM_Cr: float
    Scheme_Min_Amt_num: float
    Fund_Age_Years: float
    AMC: str
    Scheme_Type: str
    Scheme_NAV_Name: str
    AAUM_Quarter: str


class RiskOutput(BaseModel):
    predicted_risk: str
    probabilities: Dict[str, float]
    explanation: str


# -----------------------------------------------------------------------------
# Utility prediction function
# -----------------------------------------------------------------------------
def predict_risk_from_dict(data: dict):
    """
    Takes a dict with keys matching MODEL_FEATURES.
    Returns (label, prob_dict, explanation).
    """
    # Ensure correct column order
    row = {k: data.get(k) for k in MODEL_FEATURES}
    df = pd.DataFrame([row])

    # Predicted label
    pred = risk_model.predict(df)[0]

    # Class probabilities
    if hasattr(risk_model, "predict_proba"):
        proba = risk_model.predict_proba(df)[0]
        # risk_model.classes_ contains the class labels in the order of proba
        prob_dict: Dict[str, float] = {
            str(cls): float(p) for cls, p in zip(risk_model.classes_, proba)
        }
        conf = prob_dict[str(pred)]
        explanation = (
            f"The model predicts this fund is '{pred}' risk with "
            f"approximately {conf * 100:.1f}% confidence."
        )
    else:
        prob_dict = {}
        explanation = "This model does not expose class probabilities."

    return str(pred), prob_dict, explanation


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """
    Homepage: overview, graphics, links to prediction page + model info.
    """
    context = {
        "request": request,
        "model_info": MODEL_INFO,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    """
    Show HTML form to input mutual fund details and predict risk.
    """
    context = {
        "request": request,
        "result": None,
        "probabilities": None,
        "explanation": None,
        "error": None,
        "input_data": {},
        "amc_options": AMC_OPTIONS,
        "aaum_options": AAUM_OPTIONS,
    }
    return templates.TemplateResponse("predict.html", context)


@app.post("/predict", response_class=HTMLResponse)
async def post_predict_form(
    request: Request,
    NAV: float = Form(...),
    Average_AUM_Cr: float = Form(...),
    Scheme_Min_Amt_num: float = Form(...),
    Fund_Age_Years: float = Form(...),
    AMC: str = Form(...),
    Scheme_Type: str = Form(...),
    Scheme_NAV_Name: str = Form(...),
    AAUM_Quarter: str = Form(...),
):
    """
    Handle form submit, call model, show prediction + probabilities + explanation.
    """
    form_data = {
        "NAV": NAV,
        "Average_AUM_Cr": Average_AUM_Cr,
        "Scheme_Min_Amt_num": Scheme_Min_Amt_num,
        "Fund_Age_Years": Fund_Age_Years,
        "AMC": AMC,
        "Scheme_Type": Scheme_Type,
        "Scheme_NAV_Name": Scheme_NAV_Name,
        "AAUM_Quarter": AAUM_Quarter,
    }

    try:
        label, prob_dict, explanation = predict_risk_from_dict(form_data)
        context = {
            "request": request,
            "result": label,
            "probabilities": prob_dict,
            "explanation": explanation,
            "error": None,
            "input_data": form_data,
            "amc_options": AMC_OPTIONS,
            "aaum_options": AAUM_OPTIONS,
        }
    except Exception as e:
        context = {
            "request": request,
            "result": None,
            "probabilities": None,
            "explanation": None,
            "error": str(e),
            "input_data": form_data,
            "amc_options": AMC_OPTIONS,
            "aaum_options": AAUM_OPTIONS,
        }

    return templates.TemplateResponse("predict.html", context)


@app.post("/api/predict", response_model=RiskOutput)
async def api_predict(input_data: RiskInput):
    """
    JSON API endpoint: returns risk label, probabilities and explanation.
    """
    label, prob_dict, explanation = predict_risk_from_dict(input_data.dict())
    return RiskOutput(
        predicted_risk=label,
        probabilities=prob_dict,
        explanation=explanation,
    )


@app.get("/model-info", response_class=HTMLResponse)
async def model_info(request: Request):
    """
    Page showing model algorithm, metrics and feature list.
    """
    context = {
        "request": request,
        "model_info": MODEL_INFO,
    }
    return templates.TemplateResponse("model_info.html", context)

