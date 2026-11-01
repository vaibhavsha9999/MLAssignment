from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ROOT_DIR / "model"


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "test_data.csv")


def get_model_paths() -> list[tuple[str, Path]]:
    return [
        ("logistic_regression", MODEL_DIR / "logistic_regression.joblib"),
        ("decision_tree", MODEL_DIR / "decision_tree.joblib"),
        ("knn", MODEL_DIR / "knn.joblib"),
        ("naive_bayes", MODEL_DIR / "naive_bayes.joblib"),
        ("svm", MODEL_DIR / "svm.joblib"),
    ]


def get_positive_probabilities(model, X_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        return decision

    return None


def evaluate_model(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    positive_scores = get_positive_probabilities(model, X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, positive_scores) if positive_scores is not None else None,
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    return metrics


def evaluate_all_models() -> pd.DataFrame:
    test_df = load_test_data()
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    results = []
    for model_name, model_path in get_model_paths():
        model = joblib.load(model_path)
        results.append(evaluate_model(model_name, model, X_test, y_test))

    evaluation_df = pd.DataFrame(results)
    evaluation_df = evaluation_df[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    evaluation_df.to_csv(ARTIFACTS_DIR / "model_evaluation_summary.csv", index=False)

    print(evaluation_df.to_string(index=False))
    return evaluation_df


if __name__ == "__main__":
    evaluate_all_models()
