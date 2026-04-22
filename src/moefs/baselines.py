from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_classification_metrics


@dataclass
class BaselineOutcome:
    method_name: str
    metrics: dict[str, float]
    selected_feature_indices: list[int]
    selected_feature_names: list[str]
    best_params: dict[str, Any]
    estimator: Any
    cv_scores: np.ndarray


def _base_rf(seed: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=seed,
        class_weight="balanced",
        n_jobs=n_jobs,
    )


def _candidate_feature_counts(total_features: int) -> list[int]:
    raw_counts = {
        max(2, int(total_features * 0.1)),
        max(2, int(total_features * 0.25)),
        max(2, int(total_features * 0.5)),
        max(2, int(total_features * 0.75)),
        total_features,
    }
    return sorted(raw_counts)


def _evaluation_cv(seed: int, quick_mode: bool) -> StratifiedKFold:
    return StratifiedKFold(n_splits=3 if quick_mode else 5, shuffle=True, random_state=seed)


def run_grid_search_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_jobs: int,
    quick_mode: bool = False,
) -> BaselineOutcome:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if quick_mode:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.7],
        }
    else:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.8],
        }

    grid = GridSearchCV(
        estimator=_base_rf(seed, n_jobs),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        n_features=X_train.shape[1],
    )

    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=_evaluation_cv(seed, quick_mode),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    return BaselineOutcome(
        method_name="Grid Search RF",
        metrics=metrics,
        selected_feature_indices=list(range(X_train.shape[1])),
        selected_feature_names=feature_names,
        best_params=grid.best_params_,
        estimator=best_model,
        cv_scores=np.asarray(cv_scores, dtype=float),
    )


def run_random_search_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_jobs: int,
    quick_mode: bool = False,
) -> BaselineOutcome:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    param_dist = {
        "n_estimators": randint(80, 450),
        "max_depth": randint(2, 28),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": uniform(loc=0.2, scale=0.8),
    }

    random_search = RandomizedSearchCV(
        estimator=_base_rf(seed, n_jobs),
        param_distributions=param_dist,
        n_iter=10 if quick_mode else 30,
        scoring="accuracy",
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        n_features=X_train.shape[1],
    )

    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=_evaluation_cv(seed, quick_mode),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    return BaselineOutcome(
        method_name="Random Search RF",
        metrics=metrics,
        selected_feature_indices=list(range(X_train.shape[1])),
        selected_feature_names=feature_names,
        best_params=random_search.best_params_,
        estimator=best_model,
        cv_scores=np.asarray(cv_scores, dtype=float),
    )


def run_pca_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_jobs: int,
    quick_mode: bool = False,
) -> BaselineOutcome:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=seed)),
            ("clf", _base_rf(seed, n_jobs)),
        ]
    )

    pca_grid = [0.9, 0.95] if quick_mode else [0.8, 0.9, 0.95]
    n_estimators_grid = [100] if quick_mode else [100, 200]
    max_depth_grid = [None, 10] if quick_mode else [None, 10, 20]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "pca__n_components": pca_grid,
            "clf__n_estimators": n_estimators_grid,
            "clf__max_depth": max_depth_grid,
        },
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    predictions = best_pipeline.predict(X_test)

    pca_step: PCA = best_pipeline.named_steps["pca"]
    reduced_feature_count = int(getattr(pca_step, "n_components_", X_train.shape[1]))

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        n_features=reduced_feature_count,
    )

    cv_scores = cross_val_score(
        best_pipeline,
        X_train,
        y_train,
        cv=_evaluation_cv(seed, quick_mode),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    return BaselineOutcome(
        method_name="PCA + RandomForest",
        metrics=metrics,
        selected_feature_indices=list(range(reduced_feature_count)),
        selected_feature_names=[f"PC_{i + 1}" for i in range(reduced_feature_count)],
        best_params=grid.best_params_,
        estimator=best_pipeline,
        cv_scores=np.asarray(cv_scores, dtype=float),
    )


def run_rfe_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_jobs: int,
    quick_mode: bool = False,
) -> BaselineOutcome:
    total_features = X_train.shape[1]
    candidate_counts = _candidate_feature_counts(total_features)
    if quick_mode:
        candidate_counts = sorted({candidate_counts[1], candidate_counts[2], candidate_counts[-1]})

    selector_estimator = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        random_state=seed,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", RFE(estimator=selector_estimator, step=0.2)),
            ("clf", _base_rf(seed, n_jobs)),
        ]
    )

    n_estimators_grid = [100] if quick_mode else [100, 200]
    max_depth_grid = [None, 10] if quick_mode else [None, 10, 20]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "selector__n_features_to_select": candidate_counts,
            "clf__n_estimators": n_estimators_grid,
            "clf__max_depth": max_depth_grid,
        },
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    predictions = best_pipeline.predict(X_test)

    support_mask = best_pipeline.named_steps["selector"].support_
    selected_indices = np.where(support_mask)[0].tolist()
    selected_names = [feature_names[index] for index in selected_indices]

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        n_features=len(selected_indices),
    )

    cv_scores = cross_val_score(
        best_pipeline,
        X_train,
        y_train,
        cv=_evaluation_cv(seed, quick_mode),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    return BaselineOutcome(
        method_name="RFE + RandomForest",
        metrics=metrics,
        selected_feature_indices=selected_indices,
        selected_feature_names=selected_names,
        best_params=grid.best_params_,
        estimator=best_pipeline,
        cv_scores=np.asarray(cv_scores, dtype=float),
    )


def run_rf_importance_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_jobs: int,
    quick_mode: bool = False,
) -> BaselineOutcome:
    model = _base_rf(seed, n_jobs)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    ranking = np.argsort(importances)[::-1]

    candidate_counts = _candidate_feature_counts(X_train.shape[1])
    if quick_mode:
        candidate_counts = sorted({candidate_counts[1], candidate_counts[2], candidate_counts[-1]})
    best_count = candidate_counts[0]
    best_score = -np.inf

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for count in candidate_counts:
        indices = ranking[:count]
        candidate_model = _base_rf(seed, n_jobs)
        score = cross_val_score(
            candidate_model,
            X_train[:, indices],
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
        ).mean()

        if score > best_score:
            best_score = float(score)
            best_count = count

    selected_indices = ranking[:best_count].tolist()
    selected_names = [feature_names[index] for index in selected_indices]

    best_model = _base_rf(seed, n_jobs)
    best_model.fit(X_train[:, selected_indices], y_train)
    predictions = best_model.predict(X_test[:, selected_indices])

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        n_features=len(selected_indices),
    )

    cv_scores = cross_val_score(
        best_model,
        X_train[:, selected_indices],
        y_train,
        cv=_evaluation_cv(seed, quick_mode),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    return BaselineOutcome(
        method_name="RF Importance + RandomForest",
        metrics=metrics,
        selected_feature_indices=selected_indices,
        selected_feature_names=selected_names,
        best_params={"top_k_features": len(selected_indices)},
        estimator=best_model,
        cv_scores=np.asarray(cv_scores, dtype=float),
    )


def run_all_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_jobs: int,
    logger: logging.Logger,
    quick_mode: bool = False,
) -> dict[str, BaselineOutcome]:
    """Run all baseline methods and return structured outcomes."""

    logger.info("Running baseline: PCA + RandomForest")
    pca_result = run_pca_baseline(X_train, X_test, y_train, y_test, seed, n_jobs, quick_mode)

    logger.info("Running baseline: RFE + RandomForest")
    rfe_result = run_rfe_baseline(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        seed,
        n_jobs,
        quick_mode,
    )

    logger.info("Running baseline: RF importance + RandomForest")
    rf_importance_result = run_rf_importance_baseline(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        seed,
        n_jobs,
        quick_mode,
    )

    logger.info("Running baseline: Grid Search RF")
    grid_result = run_grid_search_baseline(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        seed,
        n_jobs,
        quick_mode,
    )

    logger.info("Running baseline: Random Search RF")
    random_result = run_random_search_baseline(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        seed,
        n_jobs,
        quick_mode,
    )

    return {
        pca_result.method_name: pca_result,
        rfe_result.method_name: rfe_result,
        rf_importance_result.method_name: rf_importance_result,
        grid_result.method_name: grid_result,
        random_result.method_name: random_result,
    }
