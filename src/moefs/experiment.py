from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .baselines import BaselineOutcome, run_all_baselines
from .config import ExperimentConfig
from .datasets import TrainOnlyFeatureExpander, load_medical_datasets
from .evaluation import compare_against_reference, compute_classification_metrics, repeated_cv_accuracy
from .evolution import NSGA2FeatureHyperOptimizer
from .utils import (
    collect_environment_metadata,
    configure_logging,
    dataframe_to_markdown,
    ensure_directories,
    get_package_versions,
    make_json_serializable,
    save_dataframe,
    save_json,
    set_global_seed,
    slugify,
)
from .visualization import (
    plot_dataset_comparison,
    plot_feature_importance_stability,
    plot_generation_history,
    plot_overall_accuracy,
    plot_pareto_front,
)


def _build_result_row(
    dataset_name: str,
    method_name: str,
    classifier_family: str,
    metrics: dict[str, float],
    train_cv_accuracy: float,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "method": method_name,
        "classifier_family": classifier_family,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "n_features": int(metrics["n_features"]),
        "train_cv_accuracy": float(train_cv_accuracy),
        "hyperparameters": json.dumps(hyperparameters),
    }


def _baseline_significance_scores(
    outcome: BaselineOutcome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    if outcome.method_name == "RF Importance + RandomForest":
        return repeated_cv_accuracy(
            model=outcome.estimator,
            X=X_train,
            y=y_train,
            feature_indices=outcome.selected_feature_indices,
            seed=config.seed,
            n_splits=config.significance_cv_splits,
            n_repeats=config.significance_cv_repeats,
        )

    return repeated_cv_accuracy(
        model=outcome.estimator,
        X=X_train,
        y=y_train,
        feature_indices=None,
        seed=config.seed,
        n_splits=config.significance_cv_splits,
        n_repeats=config.significance_cv_repeats,
    )


def _nsga_permutation_stability(
    optimizer: NSGA2FeatureHyperOptimizer,
    hyperparameters: dict[str, float | int | str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    selected_feature_names: list[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Estimate feature importance stability for the selected NSGA-II solution."""

    if selected_indices.size == 0:
        return pd.DataFrame()

    X_selected = X_train[:, selected_indices]
    n_splits = 2 if config.quick_mode else min(5, config.cv_folds)
    n_repeats = 1 if config.quick_mode else 2
    permutation_repeats = max(5, config.permutation_repeats // 3) if config.quick_mode else config.permutation_repeats

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=config.seed,
    )

    importance_runs: list[np.ndarray] = []
    top_frequency = np.zeros(X_selected.shape[1], dtype=float)

    for fold_id, (fold_train, fold_test) in enumerate(splitter.split(X_selected, y_train)):
        model = optimizer.build_classifier(
            params=hyperparameters,
            seed=config.seed + fold_id,
            n_jobs=config.n_jobs,
        )
        model.fit(X_selected[fold_train], y_train[fold_train])

        permutation = permutation_importance(
            model,
            X_selected[fold_test],
            y_train[fold_test],
            scoring="accuracy",
            n_repeats=permutation_repeats,
            random_state=config.seed + fold_id,
            n_jobs=config.n_jobs,
        )

        means = permutation.importances_mean
        importance_runs.append(means)

        top_k = min(config.permutation_top_k, means.size)
        top_indices = np.argsort(means)[::-1][:top_k]
        top_frequency[top_indices] += 1.0

    stacked = np.vstack(importance_runs)
    mean_importance = stacked.mean(axis=0)
    std_importance = stacked.std(axis=0, ddof=1 if stacked.shape[0] > 1 else 0)

    variability = std_importance / (np.abs(mean_importance) + 1e-12)
    stability_index = 1.0 / (1.0 + variability)
    top_selection_frequency = top_frequency / stacked.shape[0]

    df = pd.DataFrame(
        {
            "feature": selected_feature_names,
            "importance_mean": mean_importance,
            "importance_std": std_importance,
            "stability_index": stability_index,
            "top_selection_frequency": top_selection_frequency,
        }
    )

    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def run_full_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run NSGA-II and baseline experiments across all required datasets."""

    config.validate()
    start_time = time.perf_counter()

    ensure_directories(
        [
            config.resolved_plots_dir,
            config.resolved_results_dir,
        ]
    )

    logger = configure_logging(config.resolved_results_dir / "experiment.log")
    set_global_seed(config.seed)

    logger.info("Starting experiment with configuration: %s", config)

    datasets = load_medical_datasets(
        raw_data_dir=config.raw_data_dir,
        logger=logger,
        dataset_limit=config.dataset_limit,
    )

    if config.dataset_limit is not None:
        logger.info("Dataset limit applied: processing first %d dataset(s)", config.dataset_limit)

    comparison_rows: list[dict[str, object]] = []
    significance_tables: list[pd.DataFrame] = []
    dataset_overview_rows: list[dict[str, object]] = []

    for dataset in datasets:
        dataset_name = dataset.name
        dataset_slug = slugify(dataset_name)

        logger.info("=" * 80)
        logger.info("Processing dataset: %s", dataset_name)

        X = dataset.X.values
        y = dataset.y.values
        base_feature_names = list(dataset.X.columns)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.seed,
            stratify=y,
        )

        # Use train-only statistics to avoid leakage.
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)

        expander = TrainOnlyFeatureExpander(
            enable_feature_expansion=config.enable_feature_expansion,
            max_features=config.feature_space_max,
            dataset_name=dataset_name,
            logger=logger,
        )
        X_train = expander.fit_transform(X_train, base_feature_names)
        X_test = expander.transform(X_test)
        feature_names = expander.output_feature_names

        optimizer = NSGA2FeatureHyperOptimizer(config=config, logger=logger)
        evolution_result = optimizer.fit(
            X=X_train,
            y=y_train,
            feature_names=feature_names,
            dataset_name=dataset_name,
        )

        selected_indices = np.asarray(evolution_result.selected_feature_indices, dtype=int)
        nsga_model = optimizer.build_classifier(
            params=evolution_result.best_hyperparameters,
            seed=config.seed,
            n_jobs=config.n_jobs,
        )
        nsga_model.fit(X_train[:, selected_indices], y_train)
        nsga_predictions = nsga_model.predict(X_test[:, selected_indices])

        nsga_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=nsga_predictions,
            n_features=len(selected_indices),
        )

        nsga_scores = repeated_cv_accuracy(
            model=nsga_model,
            X=X_train,
            y=y_train,
            feature_indices=selected_indices,
            seed=config.seed,
            n_splits=config.significance_cv_splits,
            n_repeats=config.significance_cv_repeats,
        )

        comparison_rows.append(
            _build_result_row(
                dataset_name=dataset_name,
                method_name="NSGA-II (DEAP)",
                classifier_family=evolution_result.best_classifier,
                metrics=nsga_metrics,
                train_cv_accuracy=evolution_result.best_accuracy,
                hyperparameters=evolution_result.best_hyperparameters,
            )
        )

        permutation_df = _nsga_permutation_stability(
            optimizer=optimizer,
            hyperparameters=evolution_result.best_hyperparameters,
            X_train=X_train,
            y_train=y_train,
            selected_indices=selected_indices,
            selected_feature_names=evolution_result.selected_feature_names,
            config=config,
        )

        if not permutation_df.empty:
            save_dataframe(
                permutation_df,
                config.resolved_results_dir / f"{dataset_slug}_nsga_permutation_importance.csv",
            )
            plot_feature_importance_stability(
                importance_df=permutation_df,
                dataset_name=dataset_name,
                output_path=config.resolved_plots_dir / f"{dataset_slug}_nsga_feature_stability.png",
                top_k=min(config.permutation_top_k, 20),
            )

        baseline_results = run_all_baselines(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            seed=config.seed,
            n_jobs=config.n_jobs,
            logger=logger,
            quick_mode=config.quick_mode,
        )

        method_scores: dict[str, np.ndarray] = {"NSGA-II (DEAP)": nsga_scores}

        for baseline in baseline_results.values():
            comparison_rows.append(
                _build_result_row(
                    dataset_name=dataset_name,
                    method_name=baseline.method_name,
                    classifier_family="random_forest",
                    metrics=baseline.metrics,
                    train_cv_accuracy=float(np.mean(baseline.cv_scores)),
                    hyperparameters=baseline.best_params,
                )
            )
            method_scores[baseline.method_name] = _baseline_significance_scores(
                outcome=baseline,
                X_train=X_train,
                y_train=y_train,
                config=config,
            )

        significance_df = compare_against_reference(
            dataset_name=dataset_name,
            reference_name="NSGA-II (DEAP)",
            reference_scores=nsga_scores,
            method_scores=method_scores,
            alpha=config.significance_alpha,
            min_pairs=config.significance_min_pairs,
        )
        significance_tables.append(significance_df)

        dataset_overview_rows.append(
            {
                "dataset": dataset_name,
                "n_samples": int(X.shape[0]),
                "base_features": int(len(base_feature_names)),
                "expanded_features": int(len(feature_names)),
                "selected_features_nsga": int(len(selected_indices)),
                "selected_classifier_nsga": evolution_result.best_classifier,
            }
        )

        dataset_comparison_df = pd.DataFrame([row for row in comparison_rows if row["dataset"] == dataset_name])

        save_dataframe(
            dataset_comparison_df,
            config.resolved_results_dir / f"{dataset_slug}_comparison.csv",
        )
        save_dataframe(
            pd.DataFrame(evolution_result.pareto_records),
            config.resolved_results_dir / f"{dataset_slug}_pareto_front.csv",
        )
        save_dataframe(
            pd.DataFrame(evolution_result.generation_log),
            config.resolved_results_dir / f"{dataset_slug}_evolution_log.csv",
        )

        plot_pareto_front(
            pareto_records=evolution_result.pareto_records,
            selected_accuracy=evolution_result.best_accuracy,
            selected_features=evolution_result.best_feature_count,
            dataset_name=dataset_name,
            output_path=config.resolved_plots_dir / f"{dataset_slug}_pareto_front.png",
        )

        plot_generation_history(
            generation_log=evolution_result.generation_log,
            dataset_name=dataset_name,
            output_path=config.resolved_plots_dir / f"{dataset_slug}_evolution_history.png",
        )

        plot_dataset_comparison(
            dataset_comparison=dataset_comparison_df,
            dataset_name=dataset_name,
            output_path=config.resolved_plots_dir / f"{dataset_slug}_method_comparison.png",
        )

    comparison_df = pd.DataFrame(comparison_rows)
    dataset_overview_df = pd.DataFrame(dataset_overview_rows)
    significance_df = (
        pd.concat(significance_tables, ignore_index=True) if significance_tables else pd.DataFrame()
    )

    comparison_summary_df = (
        comparison_df.groupby("method", as_index=False)[
            ["accuracy", "precision", "recall", "f1_score", "n_features"]
        ]
        .mean()
        .sort_values("accuracy", ascending=False)
    )

    pivot_accuracy_df = comparison_df.pivot_table(
        index="dataset",
        columns="method",
        values="accuracy",
        aggfunc="mean",
    ).reset_index()

    save_dataframe(comparison_df, config.resolved_results_dir / "all_results.csv")
    save_dataframe(dataset_overview_df, config.resolved_results_dir / "dataset_overview.csv")
    save_dataframe(comparison_summary_df, config.resolved_results_dir / "summary_by_method.csv")
    save_dataframe(pivot_accuracy_df, config.resolved_results_dir / "final_comparison_table.csv")
    save_dataframe(significance_df, config.resolved_results_dir / "significance_tests.csv")

    summary_markdown = dataframe_to_markdown(comparison_summary_df)
    (config.resolved_results_dir / "summary_by_method.md").write_text(summary_markdown, encoding="utf-8")

    comparison_markdown = dataframe_to_markdown(pivot_accuracy_df)
    (config.resolved_results_dir / "final_comparison_table.md").write_text(comparison_markdown, encoding="utf-8")

    if not significance_df.empty:
        significance_markdown = dataframe_to_markdown(significance_df)
        (config.resolved_results_dir / "significance_tests.md").write_text(
            significance_markdown,
            encoding="utf-8",
        )

    overview_markdown = dataframe_to_markdown(dataset_overview_df)
    (config.resolved_results_dir / "dataset_overview.md").write_text(overview_markdown, encoding="utf-8")

    plot_overall_accuracy(
        comparison_df=comparison_df,
        output_path=config.resolved_plots_dir / "overall_accuracy_boxplot.png",
    )

    end_time = time.perf_counter()
    metadata_payload = {
        "config": make_json_serializable(config),
        "runtime_seconds": float(end_time - start_time),
        "datasets_processed": [dataset.name for dataset in datasets],
        "environment": collect_environment_metadata(),
        "package_versions": get_package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "deap",
                "scipy",
                "matplotlib",
                "seaborn",
                "xgboost",
            ]
        ),
    }
    save_json(metadata_payload, config.resolved_results_dir / "experiment_metadata.json")

    logger.info("Experiment completed successfully.")
    logger.info("Results saved to: %s", config.resolved_results_dir.resolve())
    logger.info("Plots saved to: %s", config.resolved_plots_dir.resolve())

    return comparison_df, significance_df
