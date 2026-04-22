from __future__ import annotations

import logging

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from moefs.baselines import run_rf_importance_baseline


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test.baselines")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def test_rf_importance_baseline_schema() -> None:
    X, y = make_classification(
        n_samples=140,
        n_features=14,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_names = [f"f{i}" for i in range(X.shape[1])]

    outcome = run_rf_importance_baseline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        seed=42,
        n_jobs=1,
        quick_mode=True,
    )

    assert outcome.method_name == "RF Importance + RandomForest"
    assert len(outcome.selected_feature_indices) > 0
    assert len(outcome.selected_feature_names) == len(outcome.selected_feature_indices)
    assert {"accuracy", "precision", "recall", "f1_score", "n_features"}.issubset(outcome.metrics.keys())
