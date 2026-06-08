from io import StringIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://raw.githubusercontent.com/thieu1995/csv-files/main/data/pandas/hepatitis.csv"
RANDOM_STATE = 42


def load_hepatitis_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the hepatitis CSV, including the compact one-line raw format."""
    raw_text = urlopen(url, timeout=30).read().decode("utf-8")

    # The source can appear as one long whitespace-separated CSV stream:
    # header row, then each data row as one comma-separated token.
    tokens = raw_text.strip().split()
    if tokens:
        header = tokens[0].split(",")
        rows = [token.split(",") for token in tokens[1:]]
        if rows and all(len(row) == len(header) for row in rows):
            return pd.DataFrame(rows, columns=header)

    return pd.read_csv(StringIO(raw_text))


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_hepatitis_data()
    df = df.replace("", np.nan)

    target_column = "class"
    numeric_features = [
        "age",
        "bilirubin",
        "alk_phosphate",
        "sgot",
        "albumin",
        "protime",
    ]

    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    categorical_features = [column for column in X.columns if column not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts())
    print()
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["die", "live"]))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
