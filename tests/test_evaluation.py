from __future__ import annotations

import numpy as np

from moefs.evaluation import compare_against_reference, holm_adjusted_pvalues


def test_holm_adjusted_pvalues_are_valid() -> None:
    p_values = np.array([0.01, 0.03, 0.02], dtype=float)
    adjusted = holm_adjusted_pvalues(p_values)

    assert adjusted.shape == p_values.shape
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1.0)


def test_compare_against_reference_enforces_min_pairs() -> None:
    reference = np.array([0.80, 0.82, 0.79, 0.81], dtype=float)
    candidate = np.array([0.75, 0.76, 0.74, 0.75], dtype=float)

    result = compare_against_reference(
        dataset_name="Synthetic",
        reference_name="NSGA",
        reference_scores=reference,
        method_scores={"Baseline": candidate},
        alpha=0.05,
        min_pairs=8,
    )

    row = result.iloc[0]
    assert not bool(row["enough_pairs"])
    assert not bool(row["significant_at_alpha"])


def test_compare_against_reference_reports_adjusted_pvalues() -> None:
    reference = np.array([0.91, 0.90, 0.92, 0.93, 0.91, 0.90, 0.92, 0.93], dtype=float)
    baseline_a = np.array([0.85, 0.84, 0.86, 0.87, 0.86, 0.84, 0.85, 0.86], dtype=float)
    baseline_b = np.array([0.90, 0.89, 0.91, 0.91, 0.90, 0.89, 0.90, 0.91], dtype=float)

    result = compare_against_reference(
        dataset_name="Synthetic",
        reference_name="NSGA",
        reference_scores=reference,
        method_scores={"Baseline A": baseline_a, "Baseline B": baseline_b},
        alpha=0.05,
        min_pairs=8,
    )

    assert {"raw_p_value", "holm_adjusted_p_value", "wilcoxon_p_value", "t_p_value"}.issubset(result.columns)
    assert result["holm_adjusted_p_value"].notna().all()
