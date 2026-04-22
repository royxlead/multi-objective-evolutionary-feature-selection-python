from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class DatasetBundle:
    """Container for a single dataset and metadata."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    source: str
    notes: str = ""


class TrainOnlyFeatureExpander:
    """Fit feature expansion on train split only, then transform train/test consistently."""

    def __init__(
        self,
        enable_feature_expansion: bool,
        max_features: int,
        dataset_name: str,
        logger: logging.Logger,
    ) -> None:
        self.enable_feature_expansion = enable_feature_expansion
        self.max_features = max_features
        self.dataset_name = dataset_name
        self.logger = logger

        self.output_feature_names: list[str] = []

        self._poly: PolynomialFeatures | None = None
        self._selected_indices: np.ndarray | None = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, feature_names: Sequence[str]) -> "TrainOnlyFeatureExpander":
        X = np.asarray(X_train, dtype=float)
        base_feature_names = [str(name) for name in feature_names]

        if not self.enable_feature_expansion:
            self._selected_indices = np.arange(X.shape[1], dtype=int)
            self.output_feature_names = base_feature_names
            self._fitted = True
            self.logger.info(
                "Feature expansion disabled for %s; using %d original features",
                self.dataset_name,
                X.shape[1],
            )
            return self

        self._poly = PolynomialFeatures(degree=2, include_bias=False)
        expanded = self._poly.fit_transform(X)
        expanded_names = list(self._poly.get_feature_names_out(np.asarray(base_feature_names, dtype=str)))

        if expanded.shape[1] > self.max_features:
            variance = np.var(expanded, axis=0)
            selected = np.argsort(variance)[::-1][: self.max_features]
            selected = np.sort(selected)
        else:
            selected = np.arange(expanded.shape[1], dtype=int)

        self._selected_indices = selected.astype(int)
        self.output_feature_names = [expanded_names[index] for index in self._selected_indices]
        self._fitted = True

        self.logger.info(
            "Expanded %s feature space from %d to %d columns using train-only fitting",
            self.dataset_name,
            len(base_feature_names),
            len(self.output_feature_names),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TrainOnlyFeatureExpander must be fitted before transform.")

        array = np.asarray(X, dtype=float)
        if not self.enable_feature_expansion:
            return array

        if self._poly is None or self._selected_indices is None:
            raise RuntimeError("Feature expander internal state is not initialized.")

        expanded = self._poly.transform(array)
        return expanded[:, self._selected_indices]

    def fit_transform(self, X_train: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        self.fit(X_train=X_train, feature_names=feature_names)
        return self.transform(X_train)


PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def _download_csv(url: str, file_path: Path, **read_csv_kwargs: object) -> pd.DataFrame:
    df = pd.read_csv(url, **read_csv_kwargs)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def load_wisconsin_breast_cancer(logger: logging.Logger) -> DatasetBundle:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.astype(int).rename("target")
    logger.info("Loaded Wisconsin Breast Cancer dataset with shape %s", X.shape)
    return DatasetBundle(
        name="Wisconsin Breast Cancer",
        X=X,
        y=y,
        source="scikit-learn built-in (UCI Wisconsin Diagnostic Breast Cancer)",
        notes="Binary classification: malignant vs benign.",
    )


def load_pima_diabetes(raw_data_dir: Path, logger: logging.Logger) -> DatasetBundle:
    csv_path = raw_data_dir / "pima_diabetes.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info("Loaded cached Pima Diabetes dataset from %s", csv_path)
    else:
        logger.info("Downloading Pima Diabetes dataset from %s", PIMA_URL)
        df = _download_csv(PIMA_URL, csv_path, header=None)

    if df.shape[1] == 9:
        df.columns = PIMA_COLUMNS

    df = _coerce_numeric(df)

    # Domain-informed cleanup: zero values in these fields are often placeholders for missing data.
    nullable_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in nullable_zero_columns:
        if column in df.columns:
            df[column] = df[column].replace(0, np.nan)

    y = df["Outcome"].fillna(0).astype(int)
    X = df.drop(columns=["Outcome"])

    logger.info("Loaded Pima Diabetes dataset with shape %s", X.shape)
    return DatasetBundle(
        name="Pima Indians Diabetes",
        X=X,
        y=y,
        source="UCI/Kaggle mirror (Pima Indians Diabetes)",
        notes="Binary classification: diabetes onset within 5 years.",
    )


def load_heart_disease(raw_data_dir: Path, logger: logging.Logger) -> DatasetBundle:
    csv_path = raw_data_dir / "heart_disease_cleveland.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info("Loaded cached Heart Disease dataset from %s", csv_path)
    else:
        logger.info("Downloading Heart Disease dataset from %s", HEART_URL)
        df = _download_csv(HEART_URL, csv_path, header=None)

    if df.shape[1] == 14:
        df.columns = HEART_COLUMNS

    df = df.replace("?", np.nan)
    df = _coerce_numeric(df)

    y = (df["target"].fillna(0) > 0).astype(int)
    X = df.drop(columns=["target"])

    logger.info("Loaded Heart Disease dataset with shape %s", X.shape)
    return DatasetBundle(
        name="Heart Disease (Cleveland)",
        X=X,
        y=y,
        source="UCI Cleveland Heart Disease",
        notes="Multiclass target collapsed to binary (disease present vs absent).",
    )


def load_medical_datasets(
    raw_data_dir: Path,
    logger: logging.Logger,
    dataset_limit: int | None = None,
) -> list[DatasetBundle]:
    """Load required medical datasets."""

    loaders = [
        lambda: load_wisconsin_breast_cancer(logger),
        lambda: load_pima_diabetes(raw_data_dir, logger),
        lambda: load_heart_disease(raw_data_dir, logger),
    ]

    if dataset_limit is not None:
        loaders = loaders[:dataset_limit]

    return [loader() for loader in loaders]
