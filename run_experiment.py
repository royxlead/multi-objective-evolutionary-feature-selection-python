from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from moefs.config import ExperimentConfig  # noqa: E402
from moefs.experiment import run_full_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective feature selection and classifier optimization for medical datasets.",
    )
    parser.add_argument("--population-size", type=int, default=64, help="NSGA-II population size.")
    parser.add_argument("--generations", type=int, default=30, help="Number of NSGA-II generations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--disable-feature-expansion",
        action="store_true",
        help="Disable polynomial feature-space expansion.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a lightweight configuration for quick smoke testing.",
    )
    parser.add_argument(
        "--enable-xgboost",
        action="store_true",
        help="Enable XGBoost as an additional evolved classifier (requires xgboost package).",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional limit for number of datasets to run (useful for CI and smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        enable_feature_expansion=not args.disable_feature_expansion,
        enable_xgboost=args.enable_xgboost,
        dataset_limit=args.dataset_limit,
    )

    if args.quick:
        config.population_size = min(config.population_size, 16)
        config.generations = min(config.generations, 4)
        config.cv_folds = 3
        config.significance_cv_splits = 2
        config.significance_cv_repeats = 1
        config.significance_min_pairs = 2
        config.permutation_repeats = 10
        config.permutation_top_k = 10
        config.n_jobs = 1
        config.quick_mode = True

    comparison_df, significance_df = run_full_experiment(config)

    print("\n=== Final Mean Performance by Method ===")
    summary = (
        comparison_df.groupby("method", as_index=False)[["accuracy", "f1_score", "n_features"]]
        .mean()
        .sort_values("accuracy", ascending=False)
        .round(4)
    )
    print(summary.to_string(index=False))

    if not significance_df.empty:
        print("\n=== Statistical Significance (NSGA-II vs Baselines) ===")
        print(significance_df.round(4).to_string(index=False))

    print(f"\nArtifacts generated in: {PROJECT_ROOT / 'results'} and {PROJECT_ROOT / 'plots'}")


if __name__ == "__main__":
    main()
