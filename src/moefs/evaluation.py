from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold


def _get_average_mode(y_true: np.ndarray) -> str:
    return "binary" if np.unique(y_true).shape[0] == 2 else "weighted"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> dict[str, float]:
    """Compute a standard set of classification metrics."""

    average_mode = _get_average_mode(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average_mode, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average_mode, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average_mode, zero_division=0)),
        "n_features": float(n_features),
    }


def repeated_cv_accuracy(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray | list[int] | None = None,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> np.ndarray:
    """Compute repeated stratified CV accuracy scores for significance testing."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    X_array = np.asarray(X)
    y_array = np.asarray(y)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X_array, y_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        if feature_indices is not None:
            selected_idx = np.asarray(feature_indices, dtype=int)
            X_train = X_train[:, selected_idx]
            X_test = X_test[:, selected_idx]

        estimator = clone(model)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        scores.append(float(accuracy_score(y_test, predictions)))

    return np.asarray(scores, dtype=float)


def paired_cohens_d(reference_scores: np.ndarray, candidate_scores: np.ndarray) -> float:
    """Compute paired Cohen's d effect size."""

    differences = np.asarray(reference_scores) - np.asarray(candidate_scores)
    std = differences.std(ddof=1)
    if std == 0:
        return 0.0
    return float(differences.mean() / std)


def paired_mean_difference_ci(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute paired mean difference and confidence interval."""

    differences = np.asarray(reference_scores, dtype=float) - np.asarray(candidate_scores, dtype=float)
    n = differences.size

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean_diff = float(np.mean(differences))

    if n < 2:
        return mean_diff, float("nan"), float("nan")

    std = float(np.std(differences, ddof=1))
    if std == 0.0:
        return mean_diff, mean_diff, mean_diff

    sem = std / np.sqrt(n)
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    margin = critical * sem
    return mean_diff, mean_diff - margin, mean_diff + margin


def holm_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Compute Holm-Bonferroni adjusted p-values."""

    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for index, p_value in enumerate(sorted_p):
        candidate = (m - index) * p_value
        running_max = max(running_max, candidate)
        adjusted_sorted[index] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def compare_against_reference(
    dataset_name: str,
    reference_name: str,
    reference_scores: np.ndarray,
    method_scores: dict[str, np.ndarray],
    alpha: float,
    min_pairs: int = 8,
) -> pd.DataFrame:
    """Build a significance table versus a reference method."""

    rows: list[dict[str, float | str | bool]] = []

    comparable_row_indices: list[int] = []
    comparable_p_values: list[float] = []

    for method_name, scores in method_scores.items():
        if method_name == reference_name:
            continue

        n = min(len(reference_scores), len(scores))
        aligned_reference = np.asarray(reference_scores[:n], dtype=float)
        aligned_candidate = np.asarray(scores[:n], dtype=float)

        row: dict[str, float | str | bool] = {
            "dataset": dataset_name,
            "reference_method": reference_name,
            "comparison_method": method_name,
            "n_pairs": float(n),
            "enough_pairs": bool(n >= min_pairs),
            "mean_diff_reference_minus_candidate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "t_statistic": float("nan"),
            "t_p_value": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "raw_p_value": float("nan"),
            "holm_adjusted_p_value": float("nan"),
            "cohens_d": float("nan"),
            "significant_at_alpha": False,
        }

        if n >= min_pairs:
            mean_diff, ci_low, ci_high = paired_mean_difference_ci(aligned_reference, aligned_candidate)
            t_stat, t_p_value = ttest_rel(aligned_reference, aligned_candidate, nan_policy="omit")
            effect_size = paired_cohens_d(aligned_reference, aligned_candidate)

            try:
                w_stat, w_p_value = wilcoxon(aligned_reference, aligned_candidate, zero_method="wilcox")
            except ValueError:
                w_stat, w_p_value = float("nan"), float("nan")

            raw_p_value = float(w_p_value if not np.isnan(w_p_value) else t_p_value)

            row.update(
                {
                    "mean_diff_reference_minus_candidate": float(mean_diff),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "t_statistic": float(t_stat),
                    "t_p_value": float(t_p_value),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_p_value": float(w_p_value),
                    "raw_p_value": raw_p_value,
                    "cohens_d": float(effect_size),
                }
            )

            comparable_row_indices.append(len(rows))
            comparable_p_values.append(raw_p_value)

        rows.append(row)

    if comparable_p_values:
        p_values = np.asarray(comparable_p_values, dtype=float)
        adjusted = holm_adjusted_pvalues(p_values)

        for local_index, row_index in enumerate(comparable_row_indices):
            significant = bool(adjusted[local_index] < alpha)
            rows[row_index]["holm_adjusted_p_value"] = float(adjusted[local_index])
            rows[row_index]["significant_at_alpha"] = significant

    return pd.DataFrame(rows)
