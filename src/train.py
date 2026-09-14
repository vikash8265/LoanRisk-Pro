import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import create_preprocessor


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset():

    files = [
        p for p in DATA_DIR.rglob("Loan_default.csv")
        if p.is_file()
    ]

    if not files:
        raise FileNotFoundError(
            "Loan_default.csv not found."
        )

    return files[0]


def main():

    print("=" * 55)
    print("LOANRISK PRO - TRAINING")
    print("=" * 55)

    dataset_path = find_dataset()

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["LoanID", "Default"])
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    random_state=42
                )
            )
        ]
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))
    print("Training model...")

    model.fit(X_train, y_train)

    threshold = 0.175

    probability = model.predict_proba(X_test)[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    model_path = (
        MODELS_DIR /
        "loanrisk_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {
        "model_name": "HistGradientBoostingClassifier",
        "threshold": threshold,
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "dataset_type": "Synthetic / simulated loan default dataset"
    }

    with open(
        MODELS_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\n===== RESULT =====")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nModel saved:")
    print(model_path)


if __name__ == "__main__":
    main()
