from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ROOT_DIR / "model"


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "train_data.csv")


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "naive_bayes": GaussianNB(),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", random_state=42)),
            ]
        ),
    }


def train_and_save_models() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    train_df = load_training_data()
    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    for model_name, model in build_models().items():
        model.fit(X, y)
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
        print(f"Saved {model_name} to {MODEL_DIR / f'{model_name}.joblib'}")


if __name__ == "__main__":
    train_and_save_models()
