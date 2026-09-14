# LoanRisk Pro

LoanRisk Pro is an end-to-end machine learning project for estimating loan repayment-failure risk.

## Problem

Binary classification:

- 0 = repayment failure did not occur
- 1 = repayment failure occurred

## Dataset

Kaggle Loan Default Prediction Dataset.

The dataset is synthetic/simulated and contains approximately 255,000 records.

## Final Model

HistGradientBoostingClassifier

- learning_rate = 0.05
- max_iter = 200
- max_leaf_nodes = 15
- threshold = 0.175

## Final Evaluation

- Accuracy: 80.30%
- Precision: 29.54%
- Recall: 50.26%
- F1 Score: 37.21%
- ROC-AUC: 75.85%

Because the dataset is imbalanced, model evaluation was not based on accuracy alone.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Joblib
- HTML
- CSS
- JavaScript
- Jupyter Notebook
- VS Code

## Run

Install dependencies:

    pip install -r requirements.txt

Start server:

    uvicorn api.app:app --reload

Open in browser:

    http://127.0.0.1:8000

API documentation:

    http://127.0.0.1:8000/docs

## Disclaimer

This project uses a synthetic dataset and is intended for learning, demonstration, and portfolio purposes.

It should not be used for real-world lending, credit approval, or financial decision making.