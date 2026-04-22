from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def plot_pareto_front(
    pareto_records: list[dict[str, object]],
    selected_accuracy: float,
    selected_features: int,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot and save Pareto front for a dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(pareto_records)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="n_features",
        y="accuracy",
        s=80,
        alpha=0.75,
        color="#1f77b4",
        edgecolor="white",
    )

    plt.scatter(
        [selected_features],
        [selected_accuracy],
        color="#d62728",
        s=180,
        marker="*",
        label="Selected Knee Solution",
        zorder=10,
    )

    plt.title(f"Pareto Front - {dataset_name}")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Cross-Validated Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_generation_history(
    generation_log: list[dict[str, float]],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot evolutionary progress over generations."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(generation_log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=df, x="generation", y="accuracy_max", ax=axes[0], color="#2ca02c", label="Max")
    sns.lineplot(data=df, x="generation", y="accuracy_mean", ax=axes[0], color="#1f77b4", label="Mean")
    axes[0].set_title(f"Accuracy Progress - {dataset_name}")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Accuracy")

    sns.lineplot(data=df, x="generation", y="features_min", ax=axes[1], color="#d62728", label="Min")
    sns.lineplot(data=df, x="generation", y="features_mean", ax=axes[1], color="#9467bd", label="Mean")
    axes[1].set_title(f"Feature Count Progress - {dataset_name}")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Number of Features")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_dataset_comparison(
    dataset_comparison: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot per-dataset method comparison for accuracy and feature counts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = dataset_comparison.copy().sort_values("accuracy", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(
        data=df,
        x="accuracy",
        y="method",
        hue="method",
        dodge=False,
        legend=False,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title(f"Accuracy by Method - {dataset_name}")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_ylabel("Method")

    sns.barplot(
        data=df,
        x="n_features",
        y="method",
        hue="method",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="magma",
    )
    axes[1].set_title(f"Selected Features by Method - {dataset_name}")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("Method")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overall_accuracy(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create overall boxplot of accuracy grouped by method."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=comparison_df,
        x="method",
        y="accuracy",
        hue="method",
        dodge=False,
        legend=False,
        palette="Set2",
    )
    sns.stripplot(data=comparison_df, x="method", y="accuracy", color="black", alpha=0.6, size=5)
    plt.title("Overall Accuracy Distribution Across Datasets")
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance_stability(
    importance_df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Plot top feature importances with stability indicators."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if importance_df.empty:
        return

    top_df = importance_df.head(top_k).copy().iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.0, 1.2]})

    axes[0].barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"], color="#1f77b4")
    axes[0].set_title(f"Top Feature Importances - {dataset_name}")
    axes[0].set_xlabel("Permutation Importance (Mean +/- Std)")
    axes[0].set_ylabel("Feature")

    sns.barplot(
        data=top_df,
        x="stability_index",
        y="feature",
        hue="feature",
        dodge=False,
        legend=False,
        palette="crest",
        ax=axes[1],
    )
    axes[1].set_title(f"Feature Stability Index - {dataset_name}")
    axes[1].set_xlabel("Stability Index")
    axes[1].set_ylabel("Feature")
    axes[1].set_xlim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
