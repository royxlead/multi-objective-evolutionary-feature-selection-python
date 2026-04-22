from __future__ import annotations

import logging

from moefs.config import ExperimentConfig
from moefs.evolution import NSGA2FeatureHyperOptimizer


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test.evolution")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def test_classifier_gene_decoding_defaults_to_rf_and_svm() -> None:
    config = ExperimentConfig(population_size=6, generations=1, n_jobs=1)
    optimizer = NSGA2FeatureHyperOptimizer(config=config, logger=_test_logger())

    assert optimizer._decode_classifier(0.0) == "random_forest"
    assert optimizer._decode_classifier(0.999) == "svm"


def test_hyperparameter_decoding_contains_classifier_specific_fields() -> None:
    config = ExperimentConfig(population_size=6, generations=1, n_jobs=1)
    optimizer = NSGA2FeatureHyperOptimizer(config=config, logger=_test_logger())

    rf_genes = [0.0] + [0.5] * (optimizer.hyper_gene_count - 1)
    rf_params = optimizer._decode_hyperparameters(rf_genes)

    assert rf_params["classifier"] == "random_forest"
    assert "n_estimators" in rf_params
    assert "max_features" in rf_params

    svm_genes = [0.999] + [0.5] * (optimizer.hyper_gene_count - 1)
    svm_params = optimizer._decode_hyperparameters(svm_genes)

    assert svm_params["classifier"] == "svm"
    assert "C" in svm_params
    assert "gamma" in svm_params


def test_mutation_restores_empty_feature_mask() -> None:
    config = ExperimentConfig(population_size=6, generations=1, n_jobs=1)
    optimizer = NSGA2FeatureHyperOptimizer(config=config, logger=_test_logger())

    optimizer._n_features = 5
    individual = [0, 0, 0, 0, 0] + [0.4] * optimizer.hyper_gene_count

    mutated, = optimizer._mutate(individual)

    assert sum(int(g) for g in mutated[: optimizer._n_features]) >= 1
