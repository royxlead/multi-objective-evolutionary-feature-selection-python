from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RandomForestBounds:
    """Bounds for evolving RandomForest hyperparameters."""

    n_estimators: tuple[int, int] = (60, 400)
    max_depth: tuple[int, int] = (2, 24)
    min_samples_split: tuple[int, int] = (2, 20)
    min_samples_leaf: tuple[int, int] = (1, 10)
    max_features: tuple[float, float] = (0.15, 1.0)


@dataclass(frozen=True)
class SVMBounds:
    """Bounds for evolving SVM hyperparameters in log space."""

    c_log10: tuple[float, float] = (-3.0, 3.0)
    gamma_log10: tuple[float, float] = (-4.0, 1.0)


@dataclass(frozen=True)
class XGBoostBounds:
    """Bounds for evolving XGBoost hyperparameters."""

    n_estimators: tuple[int, int] = (80, 450)
    max_depth: tuple[int, int] = (2, 12)
    learning_rate: tuple[float, float] = (0.01, 0.3)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.4, 1.0)


@dataclass
class ExperimentConfig:
    """Global configuration for the experiment pipeline."""

    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1

    # NSGA-II configuration
    population_size: int = 64
    generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    mask_mutation_prob: float = 0.02
    hyper_mutation_prob: float = 0.3

    # Feature-space expansion to emulate high-dimensional settings
    enable_feature_expansion: bool = True
    feature_space_max: int = 250

    # Optional model families
    enable_xgboost: bool = False

    # Significance testing
    significance_alpha: float = 0.05
    significance_cv_splits: int = 5
    significance_cv_repeats: int = 3
    significance_min_pairs: int = 8

    # Interpretability
    permutation_repeats: int = 30
    permutation_top_k: int = 20

    # Runtime profile
    quick_mode: bool = False

    # Dataset control
    dataset_limit: int | None = None

    # Paths
    project_root: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    rf_bounds: RandomForestBounds = field(default_factory=RandomForestBounds)
    svm_bounds: SVMBounds = field(default_factory=SVMBounds)
    xgb_bounds: XGBoostBounds = field(default_factory=XGBoostBounds)

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "processed"

    @property
    def resolved_plots_dir(self) -> Path:
        return self.project_root / self.plots_dir

    @property
    def resolved_results_dir(self) -> Path:
        return self.project_root / self.results_dir

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        if self.generations < 1:
            raise ValueError("generations must be at least 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.feature_space_max < 10:
            raise ValueError("feature_space_max must be >= 10.")
        if self.significance_min_pairs < 2:
            raise ValueError("significance_min_pairs must be at least 2.")
        if self.permutation_repeats < 2:
            raise ValueError("permutation_repeats must be at least 2.")
        if self.permutation_top_k < 1:
            raise ValueError("permutation_top_k must be at least 1.")
        if self.dataset_limit is not None and self.dataset_limit < 1:
            raise ValueError("dataset_limit must be at least 1 when provided.")
