import json
import joblib
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "loanrisk_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])


def get_risk_level(probability):

    if probability < THRESHOLD:
        return "Low"

    if probability < 0.30:
        return "Medium"

    return "High"


def predict_default(data):

    input_df = pd.DataFrame([data])

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    return {
        "default_probability": round(probability, 4),
        "risk_percentage": round(probability * 100, 2),
        "prediction": prediction,
        "risk_level": get_risk_level(probability),
        "threshold": THRESHOLD
    }


if __name__ == "__main__":

    sample = {
        "Age": 35,
        "Income": 55000,
        "LoanAmount": 120000,
        "CreditScore": 620,
        "MonthsEmployed": 36,
        "NumCreditLines": 3,
        "InterestRate": 14.5,
        "LoanTerm": 36,
        "DTIRatio": 0.45,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "Yes"
    }

    print(predict_default(sample))
