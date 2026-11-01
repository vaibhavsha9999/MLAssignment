from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

MODEL_OPTIONS = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "decision_tree": MODEL_DIR / "decision_tree.joblib",
    "knn": MODEL_DIR / "knn.joblib",
    "naive_bayes": MODEL_DIR / "naive_bayes.joblib",
    "svm": MODEL_DIR / "svm.joblib",
}


def load_model(model_name: str):
    model_path = MODEL_OPTIONS[model_name]
    return joblib.load(model_path)


def get_positive_scores(model, features: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(features)

    return None


def compute_metrics(model, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = model.predict(features)
    positive_scores = get_positive_scores(model, features)

    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, positive_scores) if positive_scores is not None else None,
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 - Classification Model Evaluation")

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("Uploaded file must contain a target column named 'target'.")
    else:
        features = data.drop(columns=["target"])
        target = data["target"]

        model_name = st.selectbox(
            "Select model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
        )

        selected_model = load_model(model_name)
        metrics = compute_metrics(selected_model, features, target)

        st.subheader("Evaluation Metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        predictions = selected_model.predict(features)
        cm = confusion_matrix(target, predictions)

        st.subheader("Confusion Matrix")
        st.write(cm)

        st.subheader("Classification Summary")
        st.write(
            pd.Series(
                {
                    "Predicted Positive": (predictions == 1).sum(),
                    "Predicted Negative": (predictions == 0).sum(),
                    "Actual Positive": (target == 1).sum(),
                    "Actual Negative": (target == 0).sum(),
                }
            )
        )

else:
    st.info("Please upload a CSV file to evaluate the selected model.")
