import json
import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "models" /
    "loanrisk_model.joblib"
)

METADATA_PATH = (
    ROOT /
    "models" /
    "model_metadata.json"
)

FRONTEND_DIR = (
    ROOT /
    "frontend"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "loanrisk_model.joblib not found."
    )


model = joblib.load(
    MODEL_PATH
)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


THRESHOLD = float(
    metadata["threshold"]
)


app = FastAPI(
    title="LoanRisk Pro API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class LoanApplication(BaseModel):

    Age: int = Field(ge=18, le=100)

    Income: float = Field(gt=0)

    LoanAmount: float = Field(gt=0)

    CreditScore: int = Field(
        ge=300,
        le=850
    )

    MonthsEmployed: int = Field(
        ge=0
    )

    NumCreditLines: int = Field(
        ge=1
    )

    InterestRate: float = Field(
        gt=0
    )

    LoanTerm: int = Field(
        gt=0
    )

    DTIRatio: float = Field(
        ge=0,
        le=1
    )

    Education: str

    EmploymentType: str

    MaritalStatus: str

    HasMortgage: str

    HasDependents: str

    LoanPurpose: str

    HasCoSigner: str


def risk_level(probability):

    if probability < THRESHOLD:
        return "Low"

    if probability < 0.30:
        return "Medium"

    return "High"


@app.get("/api/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "model": metadata["model_name"],
        "threshold": THRESHOLD
    }


@app.get("/api/model-info")
def model_info():

    return {
        "success": True,
        "model": metadata
    }


@app.post("/api/predict")
def predict(application: LoanApplication):

    data = application.model_dump()

    df = pd.DataFrame([data])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    level = risk_level(
        probability
    )

    if prediction == 1:

        message = (
            "Elevated repayment-failure "
            "risk detected."
        )

    else:

        message = (
            "Repayment-failure risk is "
            "below the selected model threshold."
        )

    return {
        "success": True,

        "default_probability":
            round(
                probability,
                4
            ),

        "risk_percentage":
            round(
                probability * 100,
                2
            ),

        "prediction":
            prediction,

        "risk_level":
            level,

        "threshold":
            THRESHOLD,

        "message":
            message,

        "disclaimer":
            (
                "Educational portfolio project only. "
                "This model uses synthetic data and "
                "must not be used for real lending decisions."
            )
    }


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        FRONTEND_DIR /
        "index.html"
    )
