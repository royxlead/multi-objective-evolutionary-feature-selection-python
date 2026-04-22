from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Any

import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from .config import ExperimentConfig

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    HAS_XGBOOST = False


FITNESS_CLASS_NAME = "FitnessMultiMOEFS"
INDIVIDUAL_CLASS_NAME = "IndividualMOEFS"


@dataclass
class EvolutionResult:
    """Artifacts produced by the NSGA-II optimization run."""

    dataset_name: str
    best_accuracy: float
    best_feature_count: int
    best_classifier: str
    selected_feature_indices: list[int]
    selected_feature_names: list[str]
    best_hyperparameters: dict[str, float | int | str]
    pareto_records: list[dict[str, Any]]
    generation_log: list[dict[str, float]]


class NSGA2FeatureHyperOptimizer:
    """Joint feature selection and hyperparameter optimization via NSGA-II."""

    classifier_gene_count = 1
    rf_gene_count = 5
    svm_gene_count = 3
    xgb_gene_count = 5
    hyper_gene_count = classifier_gene_count + rf_gene_count + svm_gene_count + xgb_gene_count

    def __init__(self, config: ExperimentConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        self._rng = random.Random(config.seed)
        self._evaluation_cache: dict[tuple[float | int, ...], tuple[float, float]] = {}
        self._available_classifiers = self._resolve_available_classifiers()

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._n_features = 0
        self._cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.seed,
        )

        self._ensure_deap_creators()

    def _resolve_available_classifiers(self) -> list[str]:
        classifiers = ["random_forest", "svm"]

        if self.config.enable_xgboost:
            if HAS_XGBOOST:
                classifiers.append("xgboost")
            else:
                self.logger.warning(
                    "enable_xgboost=True but xgboost is not installed; continuing with RF and SVM only."
                )

        return classifiers

    @staticmethod
    def _ensure_deap_creators() -> None:
        if FITNESS_CLASS_NAME not in creator.__dict__:
            creator.create(FITNESS_CLASS_NAME, base.Fitness, weights=(1.0, -1.0))
        if INDIVIDUAL_CLASS_NAME not in creator.__dict__:
            creator.create(INDIVIDUAL_CLASS_NAME, list, fitness=getattr(creator, FITNESS_CLASS_NAME))

    @staticmethod
    def build_classifier(
        params: dict[str, float | int | str],
        seed: int,
        n_jobs: int,
    ):
        classifier_name = str(params["classifier"])

        if classifier_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=float(params["max_features"]),
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            )

        if classifier_name == "svm":
            return SVC(
                C=float(params["C"]),
                gamma=float(params["gamma"]),
                kernel=str(params["kernel"]),
                class_weight="balanced",
            )

        if classifier_name == "xgboost":
            if not HAS_XGBOOST or XGBClassifier is None:
                raise RuntimeError("xgboost is not available but xgboost classifier was requested.")

            return XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=n_jobs,
            )

        raise ValueError(f"Unsupported classifier: {classifier_name}")

    def _decode_classifier(self, classifier_gene: float) -> str:
        index = min(int(classifier_gene * len(self._available_classifiers)), len(self._available_classifiers) - 1)
        return self._available_classifiers[index]

    def _decode_hyperparameters(self, hyper_genes: list[float]) -> dict[str, float | int | str]:
        rf_bounds = self.config.rf_bounds
        svm_bounds = self.config.svm_bounds
        xgb_bounds = self.config.xgb_bounds

        def scale_int(gene: float, lower: int, upper: int) -> int:
            value = lower + float(np.clip(gene, 0.0, 1.0)) * (upper - lower)
            return int(round(value))

        def scale_float(gene: float, lower: float, upper: float) -> float:
            return float(lower + float(np.clip(gene, 0.0, 1.0)) * (upper - lower))

        classifier_name = self._decode_classifier(hyper_genes[0])

        rf_genes = hyper_genes[1 : 1 + self.rf_gene_count]
        svm_genes = hyper_genes[1 + self.rf_gene_count : 1 + self.rf_gene_count + self.svm_gene_count]
        xgb_genes = hyper_genes[-self.xgb_gene_count :]

        if classifier_name == "random_forest":
            return {
                "classifier": classifier_name,
                "n_estimators": scale_int(rf_genes[0], rf_bounds.n_estimators[0], rf_bounds.n_estimators[1]),
                "max_depth": scale_int(rf_genes[1], rf_bounds.max_depth[0], rf_bounds.max_depth[1]),
                "min_samples_split": scale_int(
                    rf_genes[2],
                    rf_bounds.min_samples_split[0],
                    rf_bounds.min_samples_split[1],
                ),
                "min_samples_leaf": scale_int(
                    rf_genes[3],
                    rf_bounds.min_samples_leaf[0],
                    rf_bounds.min_samples_leaf[1],
                ),
                "max_features": scale_float(rf_genes[4], rf_bounds.max_features[0], rf_bounds.max_features[1]),
            }

        if classifier_name == "svm":
            c_log10 = scale_float(svm_genes[0], svm_bounds.c_log10[0], svm_bounds.c_log10[1])
            gamma_log10 = scale_float(svm_genes[1], svm_bounds.gamma_log10[0], svm_bounds.gamma_log10[1])
            kernel = "rbf" if svm_genes[2] >= 0.5 else "linear"
            return {
                "classifier": classifier_name,
                "C": float(10.0**c_log10),
                "gamma": float(10.0**gamma_log10),
                "kernel": kernel,
            }

        return {
            "classifier": classifier_name,
            "n_estimators": scale_int(xgb_genes[0], xgb_bounds.n_estimators[0], xgb_bounds.n_estimators[1]),
            "max_depth": scale_int(xgb_genes[1], xgb_bounds.max_depth[0], xgb_bounds.max_depth[1]),
            "learning_rate": scale_float(
                xgb_genes[2],
                xgb_bounds.learning_rate[0],
                xgb_bounds.learning_rate[1],
            ),
            "subsample": scale_float(xgb_genes[3], xgb_bounds.subsample[0], xgb_bounds.subsample[1]),
            "colsample_bytree": scale_float(
                xgb_genes[4],
                xgb_bounds.colsample_bytree[0],
                xgb_bounds.colsample_bytree[1],
            ),
        }

    def _individual_cache_key(self, individual: list[float | int]) -> tuple[float | int, ...]:
        mask_key = tuple(int(g) for g in individual[: self._n_features])
        hyper_key = tuple(round(float(g), 4) for g in individual[self._n_features :])
        return mask_key + hyper_key

    def _evaluate(self, individual: list[float | int]) -> tuple[float, float]:
        cache_key = self._individual_cache_key(individual)
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        assert self._X is not None
        assert self._y is not None

        mask = np.asarray(individual[: self._n_features], dtype=int)
        selected_indices = np.where(mask == 1)[0]
        selected_count = int(selected_indices.size)

        if selected_count == 0:
            fitness = (0.0, float(self._n_features + 1))
            self._evaluation_cache[cache_key] = fitness
            return fitness

        X_selected = self._X[:, selected_indices]
        params = self._decode_hyperparameters(list(individual[self._n_features :]))
        classifier = self.build_classifier(params=params, seed=self.config.seed, n_jobs=self.config.n_jobs)

        scores = cross_val_score(
            classifier,
            X_selected,
            self._y,
            cv=self._cv,
            scoring="accuracy",
            n_jobs=self.config.n_jobs,
            error_score=0.0,
        )

        fitness = (float(np.mean(scores)), float(selected_count))
        self._evaluation_cache[cache_key] = fitness
        return fitness

    def _mate(self, ind1: list[float | int], ind2: list[float | int]) -> tuple[list[float | int], list[float | int]]:
        # Uniform crossover on binary mask genes.
        for index in range(self._n_features):
            if self._rng.random() < 0.5:
                ind1[index], ind2[index] = ind2[index], ind1[index]

        # Blend crossover on normalized hyperparameter genes.
        alpha = 0.3
        for index in range(self._n_features, self._n_features + self.hyper_gene_count):
            if self._rng.random() < 0.5:
                x1 = float(ind1[index])
                x2 = float(ind2[index])
                gamma = (1.0 + 2.0 * alpha) * self._rng.random() - alpha
                child1 = (1.0 - gamma) * x1 + gamma * x2
                child2 = gamma * x1 + (1.0 - gamma) * x2
                ind1[index] = float(np.clip(child1, 0.0, 1.0))
                ind2[index] = float(np.clip(child2, 0.0, 1.0))

        return ind1, ind2

    def _mutate(self, individual: list[float | int]) -> tuple[list[float | int]]:
        for index in range(self._n_features):
            if self._rng.random() < self.config.mask_mutation_prob:
                individual[index] = 1 - int(individual[index])

        for index in range(self._n_features, self._n_features + self.hyper_gene_count):
            if self._rng.random() < self.config.hyper_mutation_prob:
                mutated_value = float(individual[index]) + self._rng.gauss(0.0, 0.1)
                individual[index] = float(np.clip(mutated_value, 0.0, 1.0))

        if sum(int(g) for g in individual[: self._n_features]) == 0:
            random_index = self._rng.randrange(self._n_features)
            individual[random_index] = 1

        return (individual,)

    def _build_toolbox(self) -> base.Toolbox:
        toolbox = base.Toolbox()
        individual_cls = getattr(creator, INDIVIDUAL_CLASS_NAME)

        def init_individual() -> list[float | int]:
            mask = [self._rng.randint(0, 1) for _ in range(self._n_features)]
            if sum(mask) == 0:
                mask[self._rng.randrange(self._n_features)] = 1
            hyper = [self._rng.random() for _ in range(self.hyper_gene_count)]
            return mask + hyper

        toolbox.register("individual", tools.initIterate, individual_cls, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._mate)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", self._evaluate)
        return toolbox

    @staticmethod
    def _select_knee_solution(pareto_records: list[dict[str, Any]]) -> dict[str, Any]:
        accuracies = np.asarray([item["accuracy"] for item in pareto_records], dtype=float)
        feature_counts = np.asarray([item["n_features"] for item in pareto_records], dtype=float)

        acc_min, acc_max = accuracies.min(), accuracies.max()
        feat_min, feat_max = feature_counts.min(), feature_counts.max()

        acc_norm = (accuracies - acc_min) / (acc_max - acc_min + 1e-12)
        feat_norm = (feature_counts - feat_min) / (feat_max - feat_min + 1e-12)

        distances = np.sqrt((1.0 - acc_norm) ** 2 + (feat_norm**2))
        best_index = int(np.argmin(distances))
        return pareto_records[best_index]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        dataset_name: str,
    ) -> EvolutionResult:
        self._evaluation_cache.clear()

        self._X = np.asarray(X)
        self._y = np.asarray(y)
        self._n_features = self._X.shape[1]

        toolbox = self._build_toolbox()
        population = toolbox.population(n=self.config.population_size)

        invalid_individuals = [individual for individual in population if not individual.fitness.valid]
        initial_fitness = list(map(toolbox.evaluate, invalid_individuals))
        for individual, fitness in zip(invalid_individuals, initial_fitness):
            individual.fitness.values = fitness

        population = toolbox.select(population, len(population))
        pareto_front = tools.ParetoFront()
        pareto_front.update(population)

        generation_log: list[dict[str, float]] = []

        def record_generation(generation_number: int, individuals: list[list[float | int]]) -> None:
            fitness_values = np.asarray([ind.fitness.values for ind in individuals], dtype=float)
            metrics = {
                "generation": float(generation_number),
                "accuracy_max": float(fitness_values[:, 0].max()),
                "accuracy_mean": float(fitness_values[:, 0].mean()),
                "features_min": float(fitness_values[:, 1].min()),
                "features_mean": float(fitness_values[:, 1].mean()),
            }
            generation_log.append(metrics)

        record_generation(0, population)

        for generation in range(1, self.config.generations + 1):
            offspring = algorithms.varAnd(
                population,
                toolbox,
                cxpb=self.config.crossover_prob,
                mutpb=self.config.mutation_prob,
            )

            invalid_individuals = [individual for individual in offspring if not individual.fitness.valid]
            fitness_values = list(map(toolbox.evaluate, invalid_individuals))
            for individual, fitness in zip(invalid_individuals, fitness_values):
                individual.fitness.values = fitness

            population = toolbox.select(population + offspring, self.config.population_size)
            pareto_front.update(population)
            record_generation(generation, population)

            if generation % 5 == 0 or generation == self.config.generations:
                latest = generation_log[-1]
                self.logger.info(
                    "[%s] Gen %d | acc_max=%.4f acc_mean=%.4f feat_min=%.0f",
                    dataset_name,
                    generation,
                    latest["accuracy_max"],
                    latest["accuracy_mean"],
                    latest["features_min"],
                )

        pareto_records: list[dict[str, Any]] = []
        for individual in pareto_front:
            mask = np.asarray(individual[: self._n_features], dtype=int)
            selected_indices = np.where(mask == 1)[0].tolist()
            params = self._decode_hyperparameters(list(individual[self._n_features :]))
            pareto_records.append(
                {
                    "accuracy": float(individual.fitness.values[0]),
                    "n_features": int(len(selected_indices)),
                    "selected_indices": selected_indices,
                    "classifier": str(params["classifier"]),
                    "hyperparameters": params,
                }
            )

        pareto_records.sort(key=lambda item: (-item["accuracy"], item["n_features"]))
        selected_solution = self._select_knee_solution(pareto_records)

        selected_indices = list(selected_solution["selected_indices"])
        selected_feature_names = [feature_names[index] for index in selected_indices]

        return EvolutionResult(
            dataset_name=dataset_name,
            best_accuracy=float(selected_solution["accuracy"]),
            best_feature_count=int(selected_solution["n_features"]),
            best_classifier=str(selected_solution["hyperparameters"]["classifier"]),
            selected_feature_indices=selected_indices,
            selected_feature_names=selected_feature_names,
            best_hyperparameters=dict(selected_solution["hyperparameters"]),
            pareto_records=pareto_records,
            generation_log=generation_log,
        )
