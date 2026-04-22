from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from moefs.datasets import TrainOnlyFeatureExpander, _download_csv


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test.datasets")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def test_train_only_feature_expander_reduces_dimensions() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(40, 6))
    X_test = rng.normal(size=(10, 6))

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=True,
        max_features=15,
        dataset_name="Synthetic",
        logger=_test_logger(),
    )

    transformed_train = expander.fit_transform(X_train, [f"f{i}" for i in range(X_train.shape[1])])
    transformed_test = expander.transform(X_test)

    assert transformed_train.shape[1] <= 15
    assert transformed_test.shape[1] == transformed_train.shape[1]
    assert len(expander.output_feature_names) == transformed_train.shape[1]


def test_feature_expander_disabled_keeps_original_columns() -> None:
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(30, 4))
    X_test = rng.normal(size=(8, 4))

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=False,
        max_features=50,
        dataset_name="NoExpansion",
        logger=_test_logger(),
    )

    transformed_train = expander.fit_transform(X_train, ["a", "b", "c", "d"])
    transformed_test = expander.transform(X_test)

    assert transformed_train.shape == X_train.shape
    assert transformed_test.shape == X_test.shape
    assert expander.output_feature_names == ["a", "b", "c", "d"]


def test_download_csv_writes_cache_file(tmp_path, monkeypatch) -> None:
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("https://example.com"):
            return expected.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    output_path = tmp_path / "cached.csv"
    result = _download_csv("https://example.com/mock.csv", output_path)

    assert output_path.exists()
    pd.testing.assert_frame_equal(result, expected)
    cached = real_read_csv(output_path)
    pd.testing.assert_frame_equal(cached, expected)
