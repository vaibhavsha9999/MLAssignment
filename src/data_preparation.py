from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


DATASET_NAME = "breast_cancer_wisconsin"
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def build_dataset() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    if "target" in df.columns:
        df = df.drop(columns=["target"])

    df["target"] = data.target.astype(int)
    return df


def save_split_data(dataframe: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    train_df, test_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=dataframe["target"],
    )

    train_df.to_csv(ARTIFACTS_DIR / "train_data.csv", index=False)
    test_df.to_csv(ARTIFACTS_DIR / "test_data.csv", index=False)

    summary = {
        "dataset_name": DATASET_NAME,
        "total_rows": len(dataframe),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": len(dataframe.columns) - 1,
        "target_distribution": dataframe["target"].value_counts().to_dict(),
    }

    pd.Series(summary).to_json(ARTIFACTS_DIR / "data_split_summary.json")

    print(f"Dataset rows: {summary['total_rows']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"Files saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    df = build_dataset()
    save_split_data(df)
