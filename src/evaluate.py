import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

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


def main():

    matches = [
        p for p in DATA_DIR.rglob("Loan_default.csv")
        if p.is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            "Dataset not found."
        )

    df = pd.read_csv(matches[0])

    X = df.drop(
        columns=["LoanID", "Default"]
    )

    y = df["Default"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    threshold = float(
        metadata["threshold"]
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    print("===== FINAL MODEL EVALUATION =====")

    print(
        "Accuracy :",
        round(
            accuracy_score(
                y_test,
                prediction
            ),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall   :",
        round(
            recall_score(
                y_test,
                prediction
            ),
            4
        )
    )

    print(
        "F1 Score :",
        round(
            f1_score(
                y_test,
                prediction
            ),
            4
        )
    )

    print(
        "ROC-AUC  :",
        round(
            roc_auc_score(
                y_test,
                probability
            ),
            4
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            prediction,
            digits=4
        )
    )


if __name__ == "__main__":
    main()
